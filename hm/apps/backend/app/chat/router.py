import logging
import uuid

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth.dependencies import optional_user
from ..auth.models import User
from ..catalog.models import Product
from ..catalog.schemas import ProductOut
from ..core.config import settings
from ..core.database import get_db
from ..preferences.service import forget, load_events, persist_events
from ..recommendations.service import _recent_product_ids, _skus_for_product_ids
from .schemas import ChatIn, ChatOut, ChatProductOut, EvidenceOut

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

# First request after start may wait for the query encoder; the LLM adds seconds on top.
RAG_TIMEOUT_S = 120


def _keyword_fallback(db: Session, message: str, session_id: str | None, reason: str) -> ChatOut:
    """Used only when the RAG service is down: plain catalog keyword search, clearly labelled."""
    words = [w for w in message.split() if len(w) > 2][:4]
    query = select(Product).where(Product.is_active.is_(True))
    for word in words:
        query = query.where(Product.name.ilike(f"%{word}%"))
    products = list(db.scalars(query.limit(5))) if words else []
    reply = (
        "Trợ lý AI đang tạm thời không khả dụng, đây là kết quả tìm theo từ khóa trong tên sản phẩm."
        if products
        else "Trợ lý AI đang tạm thời không khả dụng và mình chưa tìm thấy sản phẩm khớp từ khóa. Bạn thử lại sau ít phút nhé."
    )
    return ChatOut(
        session_id=session_id or uuid.uuid4().hex,
        reply=reply,
        action="search",
        products=[ChatProductOut(product=ProductOut.model_validate(p, from_attributes=True)) for p in products],
        meta={"answer_source": "keyword_fallback", "rag_unavailable": reason},
        warnings=["rag_unavailable"],
    )


@router.get("/health")
async def chat_health():
    if not settings.datn_rag_url:
        return {"enabled": False}
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            data = (await client.get(f"{settings.datn_rag_url.rstrip('/')}/health")).json()
        return {"enabled": True, "reachable": True, **data}
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "reachable": False, "error": str(exc)[:120]}


@router.post("", response_model=ChatOut)
async def chat(payload: ChatIn, user: User | None = Depends(optional_user), db: Session = Depends(get_db)):
    if not payload.message.strip() and not payload.image_base64 and payload.action is None:
        raise HTTPException(422, "Provide a message, an image or an action")
    if not settings.datn_rag_url:
        return _keyword_fallback(db, payload.message, payload.session_id, "DATN_RAG_URL not configured")

    if user and payload.action and payload.action.type == "forget":
        # The shop DB is the durable copy: delete it first so it is not sent back; the RAG session drops its own copy.
        forget(db, user, payload.action.product_id)

    # Shop-side purchase intent (cart / recent orders) becomes the model's interaction history.
    history_skus = _skus_for_product_ids(db, _recent_product_ids(db, user, limit=10)) if user else []
    body = {
        "message": payload.message,
        "session_id": payload.session_id,
        "image_base64": payload.image_base64,
        "history_skus": history_skus,
        # Durable taste memory (likes/dislikes/clicks from earlier sessions); the RAG service weighs and decays it.
        "events": load_events(db, user) if user else [],
        "action": payload.action.model_dump(exclude_none=True) if payload.action else None,
        "filters": payload.filters.model_dump(exclude_none=True) if payload.filters else None,
    }
    try:
        async with httpx.AsyncClient(timeout=RAG_TIMEOUT_S) as client:
            resp = await client.post(f"{settings.datn_rag_url.rstrip('/')}/chat", json=body)
        if resp.status_code in (413, 422):
            raise HTTPException(resp.status_code, resp.json().get("detail", "Invalid request"))
        resp.raise_for_status()
        data = resp.json()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        return _keyword_fallback(db, payload.message, payload.session_id, str(exc)[:120])

    if user and data.get("events"):
        try:
            persist_events(db, user, data["events"])
        except Exception:  # noqa: BLE001 - remembering feedback must never break the chat answer
            db.rollback()
            log.exception("could not persist preference events")

    ids = [p["product_id"] for p in data.get("products", [])]
    rows = {p.id: p for p in db.scalars(select(Product).where(Product.id.in_(ids), Product.is_active.is_(True)))} if ids else {}
    products = []
    for item in data.get("products", []):
        row = rows.get(item["product_id"])
        if row is None:  # deleted/deactivated since indexing
            continue
        products.append(
            ChatProductOut(
                product=ProductOut.model_validate(row, from_attributes=True),  # live price/stock from Postgres
                score=item.get("score", 0.0),
                brand=item.get("brand"),
                price_estimated=item.get("price_estimated", False),
                avg_rating=item.get("avg_rating"),
                review_count=item.get("review_count", 0),
                reviews_mock=item.get("reviews_mock", False),
                audience=item.get("audience"),
                colour=item.get("colour"),
                product_type=item.get("product_type"),
                reasons=item.get("reasons", []),
                evidence=[EvidenceOut(**{k: e[k] for k in EvidenceOut.model_fields if k in e}) for e in item.get("evidence", [])],
            )
        )
    return ChatOut(
        session_id=data["session_id"],
        reply=data["reply"],
        action=data.get("action", "search"),
        lang=data.get("lang", "vi"),
        products=products,
        filter_chips=data.get("filter_chips", []),
        suggestions=data.get("suggestions", []),
        citations=data.get("citations", []),
        meta=data.get("meta", {}),
        warnings=data.get("warnings", []),
    )
