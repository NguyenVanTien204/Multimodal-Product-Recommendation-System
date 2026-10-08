from __future__ import annotations

import time
import uuid

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..auth.models import User
from ..catalog.models import Product
from .models import PreferenceEvent

KINDS = ("view", "click", "like", "cart", "purchase", "dislike")
WEB_KINDS = ("view", "click", "like", "dislike")  # what the storefront may log itself; cart/orders already feed the model
MAX_LOADED = 300  # newest events sent to the chat service per request
RETENTION_DAYS = 90


def load_events(db: Session, user: User, limit: int = MAX_LOADED) -> list[dict]:
    """The user's newest events as chat-service payloads (sku-keyed), oldest first."""
    since = time.time() - RETENTION_DAYS * 86400
    rows = db.execute(
        select(PreferenceEvent, Product.sku)
        .join(Product, Product.id == PreferenceEvent.product_id)
        .where(PreferenceEvent.user_id == user.id, PreferenceEvent.ts >= since)
        .order_by(PreferenceEvent.ts.desc())
        .limit(limit)
    ).all()
    return [
        {"sku": sku, "kind": e.kind, "ts": e.ts, "event_id": e.event_id, "source": e.source}
        for e, sku in reversed(rows)
    ]


def persist_events(db: Session, user: User, events: list[dict]) -> int:
    """Store events returned by the chat service (sku-keyed). Unknown SKUs and already-stored ids are skipped."""
    events = [e for e in events if e.get("kind") in KINDS and e.get("sku")]
    if not events:
        return 0
    product_by_sku = {sku: pid for pid, sku in db.execute(select(Product.id, Product.sku).where(Product.sku.in_({e["sku"] for e in events}))).all()}
    known = set(db.scalars(select(PreferenceEvent.event_id).where(PreferenceEvent.user_id == user.id, PreferenceEvent.event_id.in_([e["event_id"] for e in events if e.get("event_id")]))))
    added = 0
    for e in events:
        pid = product_by_sku.get(e["sku"])
        event_id = e.get("event_id") or uuid.uuid4().hex
        if pid is None or event_id in known:
            continue
        known.add(event_id)
        db.add(PreferenceEvent(user_id=user.id, product_id=pid, event_id=event_id, kind=e["kind"], ts=float(e.get("ts") or time.time()), source=str(e.get("source") or "chat")[:16]))
        added += 1
    db.commit()
    return added


def explicit_state(db: Session, user: User) -> dict[int, str]:
    """product_id -> latest explicit verdict ('like' | 'dislike'); the newest explicit event wins."""
    rows = db.execute(
        select(PreferenceEvent.product_id, PreferenceEvent.kind)
        .where(PreferenceEvent.user_id == user.id, PreferenceEvent.kind.in_(("like", "dislike")))
        .order_by(PreferenceEvent.ts.asc(), PreferenceEvent.id.asc())
    ).all()
    return {pid: kind for pid, kind in rows}


def forget(db: Session, user: User, product_id: int | None = None) -> int:
    stmt = delete(PreferenceEvent).where(PreferenceEvent.user_id == user.id)
    if product_id is not None:
        stmt = stmt.where(PreferenceEvent.product_id == product_id)
    removed = db.execute(stmt).rowcount or 0
    db.commit()
    return removed
