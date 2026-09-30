from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from . import schema as S


@dataclass(frozen=True)
class SearchFilters:
    """Hard, structured constraints applied on top of semantic retrieval.

    Prices are in VND (what the shop displays). The same object drives three
    things so they cannot drift apart: the Qdrant payload filter, an in-memory
    filter over an already-retrieved candidate pool (refinement without a new
    vector query), and the human-readable summary shown back to the user.
    """

    min_price: float | None = None
    max_price: float | None = None
    brands: tuple[str, ...] = ()
    category_slugs: tuple[str, ...] = ()
    min_rating: float | None = None
    exclude_product_ids: tuple[int, ...] = ()

    def is_empty(self) -> bool:
        return not (
            self.min_price is not None
            or self.max_price is not None
            or self.brands
            or self.category_slugs
            or self.min_rating is not None
            or self.exclude_product_ids
        )

    def with_(self, **changes: Any) -> "SearchFilters":
        return replace(self, **changes)

    def without_exclusions(self) -> "SearchFilters":
        return replace(self, exclude_product_ids=())

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SearchFilters":
        if not data:
            return cls()
        return cls(
            min_price=data.get("min_price"),
            max_price=data.get("max_price"),
            brands=tuple(data.get("brands") or ()),
            category_slugs=tuple(data.get("category_slugs") or ()),
            min_rating=data.get("min_rating"),
            exclude_product_ids=tuple(int(x) for x in (data.get("exclude_product_ids") or ())),
        )

    def matches(self, payload: Mapping[str, Any], product_id: int | None = None) -> bool:
        """In-memory equivalent of `to_qdrant`, used to refine a cached pool."""
        price = payload.get(S.P_PRICE)
        if self.min_price is not None and (price is None or price < self.min_price):
            return False
        if self.max_price is not None and (price is None or price > self.max_price):
            return False
        if self.brands:
            brand = (payload.get(S.P_BRAND) or "").lower()
            if brand not in {b.lower() for b in self.brands}:
                return False
        if self.category_slugs and payload.get(S.P_CATEGORY_SLUG) not in self.category_slugs:
            return False
        if self.min_rating is not None:
            rating = payload.get(S.P_AVG_RATING)
            if rating is None or rating < self.min_rating:
                return False
        pid = product_id if product_id is not None else payload.get(S.P_PRODUCT_ID)
        if pid is not None and pid in self.exclude_product_ids:
            return False
        return True

    def to_qdrant(self):
        from qdrant_client.http import models as qm

        must: list[Any] = []
        must_not: list[Any] = []
        if self.min_price is not None or self.max_price is not None:
            must.append(qm.FieldCondition(key=S.P_PRICE, range=qm.Range(gte=self.min_price, lte=self.max_price)))
        if self.brands:
            must.append(qm.FieldCondition(key=S.P_BRAND, match=qm.MatchAny(any=list(self.brands))))
        if self.category_slugs:
            must.append(qm.FieldCondition(key=S.P_CATEGORY_SLUG, match=qm.MatchAny(any=list(self.category_slugs))))
        if self.min_rating is not None:
            must.append(qm.FieldCondition(key=S.P_AVG_RATING, range=qm.Range(gte=self.min_rating)))
        if self.exclude_product_ids:
            must_not.append(qm.HasIdCondition(has_id=list(self.exclude_product_ids)))
        if not must and not must_not:
            return None
        return qm.Filter(must=must or None, must_not=must_not or None)

    def describe(self, currency: str = "₫") -> list[str]:
        """Short human-readable constraint list (Vietnamese) for UI chips."""
        out: list[str] = []
        if self.min_price is not None and self.max_price is not None:
            out.append(f"Giá {_fmt(self.min_price)}–{_fmt(self.max_price)}{currency}")
        elif self.max_price is not None:
            out.append(f"Giá ≤ {_fmt(self.max_price)}{currency}")
        elif self.min_price is not None:
            out.append(f"Giá ≥ {_fmt(self.min_price)}{currency}")
        if self.brands:
            out.append("Thương hiệu: " + ", ".join(self.brands))
        if self.category_slugs:
            out.append("Danh mục: " + ", ".join(self.category_slugs))
        if self.min_rating is not None:
            out.append(f"Đánh giá ≥ {self.min_rating:g}★")
        return out


def _fmt(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", ".")


@dataclass
class Hit:
    """One retrieved product. `scores` keeps the raw per-signal similarities so
    explanations can report *why* it matched rather than only a fused number."""

    product_id: int
    payload: dict[str, Any]
    score: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def sku(self) -> str:
        return str(self.payload.get(S.P_ITEM_ID, ""))

    @property
    def title(self) -> str:
        return str(self.payload.get(S.P_TITLE, ""))
