from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base


class PreferenceEvent(Base):
    """One piece of taste feedback (like / dislike / click / view) about a product.

    The chat service decides what these mean (weights, decay, negatives); the shop only stores them durably
    so the assistant remembers a user across sessions, and lets the user inspect and delete them.
    """

    __tablename__ = "preference_events"
    __table_args__ = (UniqueConstraint("user_id", "event_id"), Index("ix_preference_events_user_ts", "user_id", "ts"))
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id", ondelete="CASCADE"), index=True)
    event_id: Mapped[str] = mapped_column(String(64))  # idempotency key: the same event relayed twice is stored once
    kind: Mapped[str] = mapped_column(String(16))
    ts: Mapped[float] = mapped_column(Float)  # unix seconds
    source: Mapped[str] = mapped_column(String(16), default="shop")
