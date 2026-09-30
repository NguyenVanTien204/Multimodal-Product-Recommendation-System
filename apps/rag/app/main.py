from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
from contextlib import asynccontextmanager

from datn.agent.orchestrator import ChatAgent, ChatRequest, ChatResponse
from datn.agent.recommender_client import RecommenderClient
from datn.agent.session import SessionStore
from datn.rag.evidence import ReviewRetriever
from datn.rag.llm import OpenAICompatLLM
from datn.retrieval import schema as S
from datn.retrieval.encoder import JinaClipEncoder
from datn.retrieval.filters import SearchFilters
from datn.retrieval.search import HybridSearcher
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from qdrant_client import QdrantClient

from .config import settings
from .schemas import (
    ChatIn,
    ChatOut,
    CompareIn,
    ExplainIn,
    FiltersIn,
    HealthOut,
    ProductOut,
    RecommendIn,
    RefineIn,
    SearchIn,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rag")

state: dict[str, object] = {}
Image.MAX_IMAGE_PIXELS = 40_000_000  # decompression-bomb guard


def _agent() -> ChatAgent:
    agent = state.get("agent")
    if agent is None:
        raise HTTPException(503, "RAG service is starting")
    return agent  # type: ignore[return-value]


@asynccontextmanager
async def lifespan(_: FastAPI):
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=30)
    searcher = HybridSearcher(client)
    encoder = JinaClipEncoder(device=settings.device) if settings.load_encoder else None
    agent = ChatAgent(
        searcher=searcher,
        encoder=encoder,
        reviews=ReviewRetriever(client),
        recommender=RecommenderClient(settings.recommender_url),
        llm=OpenAICompatLLM(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout_s=settings.llm_timeout_s,
            reasoning_effort=settings.llm_reasoning_effort,
            max_tokens_floor=settings.llm_max_tokens,
            max_retries=settings.llm_max_retries,
        ),
        store=SessionStore(ttl_s=settings.session_ttl_s),
    )
    state["agent"] = agent
    # Loading Jina CLIP v2 takes tens of seconds: do it off the event loop so /health answers immediately.
    warm = asyncio.create_task(asyncio.to_thread(agent.warm_up))
    yield
    warm.cancel()
    state.clear()


app = FastAPI(title="DATN RAG Chat Service", version="1.0.0", lifespan=lifespan)


# ---- helpers ----------------------------------------------------------------------
def _filters(f: FiltersIn | None) -> SearchFilters | None:
    return SearchFilters.from_dict(f.model_dump()) if f else None


def _decode_image(data: str | None) -> Image.Image | None:
    if not data:
        return None
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        raw = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, "image_base64 is not valid base64") from exc
    if len(raw) > settings.max_image_mb * 1024 * 1024:
        raise HTTPException(413, f"image larger than {settings.max_image_mb:g} MB")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "image_base64 is not a readable image") from exc
    return img.convert("RGB")


def _out(resp: ChatResponse) -> ChatOut:
    products = []
    for r in resp.products:
        p = r.payload
        products.append(
            ProductOut(
                product_id=r.product_id,
                sku=r.sku,
                title=str(p.get(S.P_TITLE, "")),
                brand=p.get(S.P_BRAND),
                category=p.get(S.P_CATEGORY),
                price=p.get(S.P_PRICE),
                price_estimated=bool(p.get(S.P_PRICE_ESTIMATED, False)),
                image_url=p.get(S.P_IMAGE_URL),
                avg_rating=p.get(S.P_AVG_RATING),
                review_count=int(p.get(S.P_REVIEW_COUNT, 0) or 0),
                score=r.score,
                reasons=r.reasons,
                evidence=r.evidence,
                signals=r.signals,
            )
        )
    return ChatOut(
        session_id=resp.session_id,
        reply=resp.reply,
        action=resp.action,
        lang=resp.lang,
        products=products,
        filters=resp.filters,
        filter_chips=resp.filter_chips,
        suggestions=resp.suggestions,
        citations=resp.citations,
        meta=resp.meta,
        warnings=resp.warnings,
    )


def _k(k: int | None) -> int:
    return min(k or settings.default_k, settings.max_k)


# ---- endpoints --------------------------------------------------------------------
@app.get("/health", response_model=HealthOut)
async def health():
    agent = state.get("agent")
    if agent is None:
        return HealthOut(status="starting", encoder_loaded=False, encoder_ok=False, llm="disabled", llm_status={"state": "disabled"}, collections={}, recommender={}, sessions=0)
    collections = await asyncio.to_thread(agent.searcher.ready)
    ready = collections.get(S.PRODUCTS_COLLECTION, 0) > 0
    return HealthOut(
        status="ok" if ready else "degraded",
        encoder_loaded=bool(agent.encoder and agent.encoder.loaded),
        encoder_ok=agent.encoder_ok,
        llm=agent.llm.name,
        llm_status=agent.llm.status(),
        collections=collections,
        recommender=await agent.recommender.health(),
        sessions=len(agent.store),
    )


@app.post("/chat", response_model=ChatOut)
async def chat(payload: ChatIn):
    action = payload.action.model_dump(exclude_none=True) if payload.action else None
    resp = await _agent().handle(
        ChatRequest(
            message=payload.message.strip(),
            session_id=payload.session_id,
            image=_decode_image(payload.image_base64),
            history_skus=payload.history_skus,
            action=action,
            filters=_filters(payload.filters),
            k=_k(payload.k),
        )
    )
    return _out(resp)


@app.post("/search", response_model=ChatOut)
async def search(payload: SearchIn):
    image = _decode_image(payload.image_base64)
    if not payload.query.strip() and image is None:
        raise HTTPException(422, "provide a query and/or an image")
    resp = await _agent().handle(
        ChatRequest(
            message=payload.query.strip(),
            session_id=payload.session_id,
            image=image,
            history_skus=payload.history_skus,
            filters=_filters(payload.filters),
            force="search",
            k=_k(payload.k),
        )
    )
    return _out(resp)


@app.post("/recommend", response_model=ChatOut)
async def recommend(payload: RecommendIn):
    resp = await _agent().handle(
        ChatRequest(
            session_id=payload.session_id,
            history_skus=payload.history_skus,
            filters=_filters(payload.filters),
            force="recommend",
            k=_k(payload.k),
        )
    )
    return _out(resp)


@app.post("/refine", response_model=ChatOut)
async def refine(payload: RefineIn):
    agent = _agent()
    if agent.store.get(payload.session_id) is None:
        raise HTTPException(404, "unknown session_id (start with /search or /chat)")
    resp = await agent.handle(
        ChatRequest(message=payload.message.strip(), session_id=payload.session_id, force="refine", k=_k(payload.k))
    )
    return _out(resp)


@app.post("/explain", response_model=ChatOut)
async def explain(payload: ExplainIn):
    resp = await _agent().handle(
        ChatRequest(
            message=payload.question.strip(),
            session_id=payload.session_id,
            action={"type": "explain", "product_id": payload.product_id},
        )
    )
    return _out(resp)


@app.post("/compare", response_model=ChatOut)
async def compare(payload: CompareIn):
    resp = await _agent().handle(
        ChatRequest(
            message=payload.question.strip(),
            session_id=payload.session_id,
            action={"type": "compare", "product_ids": payload.product_ids},
        )
    )
    return _out(resp)


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    _agent().store.delete(session_id)
