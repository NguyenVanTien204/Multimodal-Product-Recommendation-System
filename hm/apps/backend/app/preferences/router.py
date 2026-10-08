import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth.dependencies import current_user
from ..auth.models import User
from ..catalog.models import Product
from ..catalog.schemas import ProductOut
from ..core.database import get_db
from .models import PreferenceEvent
from .service import WEB_KINDS, explicit_state, forget

router = APIRouter(prefix="/me/preferences", tags=["preferences"])


class EventIn(BaseModel):
    product_id: int
    kind: str = Field(pattern="^(" + "|".join(WEB_KINDS) + ")$")


class PreferencesOut(BaseModel):
    liked: list[ProductOut] = Field(default_factory=list)
    disliked: list[ProductOut] = Field(default_factory=list)
    event_count: int = 0


@router.post("/events", status_code=status.HTTP_201_CREATED)
def log_event(payload: EventIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    product = db.get(Product, payload.product_id)
    if product is None or not product.is_active:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    db.add(PreferenceEvent(user_id=user.id, product_id=product.id, event_id=uuid.uuid4().hex, kind=payload.kind, ts=time.time(), source="web"))
    db.commit()
    return {"ok": True}


@router.get("", response_model=PreferencesOut)
def my_preferences(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """What the assistant currently remembers: items the user liked and disliked (newest verdict per item)."""
    state = explicit_state(db, user)
    products = {p.id: p for p in db.scalars(select(Product).where(Product.id.in_(list(state)), Product.is_active.is_(True)))} if state else {}
    count = len(db.scalars(select(PreferenceEvent.id).where(PreferenceEvent.user_id == user.id)).all())
    return PreferencesOut(
        liked=[products[pid] for pid, kind in state.items() if kind == "like" and pid in products],
        disliked=[products[pid] for pid, kind in state.items() if kind == "dislike" and pid in products],
        event_count=count,
    )


@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def forget_product(product_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    forget(db, user, product_id)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def forget_everything(user: User = Depends(current_user), db: Session = Depends(get_db)):
    forget(db, user)
