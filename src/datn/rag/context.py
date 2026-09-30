from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence

from ..retrieval import schema as S
from .evidence import Review


def format_vnd(value: float | None) -> str:
    if value is None:
        return "chưa có giá"
    return f"{int(round(value)):,}".replace(",", ".") + "₫"


@dataclass
class ProductFact:
    """Everything the model may say about one product. Nothing outside this
    object (and the review snippets attached to it) is admissible evidence."""

    tag: str  # "P1", "P2", ... -- the handle the LLM must cite
    product_id: int
    sku: str
    title: str
    brand: str | None
    category: str | None
    price: float | None
    price_estimated: bool
    avg_rating: float | None
    review_count: int
    description: str | None
    features: str | None
    image_url: str | None = None

    @classmethod
    def from_payload(cls, tag: str, product_id: int, payload: Mapping[str, Any]) -> "ProductFact":
        return cls(
            tag=tag,
            product_id=product_id,
            sku=str(payload.get(S.P_ITEM_ID, "")),
            title=str(payload.get(S.P_TITLE, "")),
            brand=payload.get(S.P_BRAND),
            category=payload.get(S.P_CATEGORY),
            price=payload.get(S.P_PRICE),
            price_estimated=bool(payload.get(S.P_PRICE_ESTIMATED, False)),
            avg_rating=payload.get(S.P_AVG_RATING),
            review_count=int(payload.get(S.P_REVIEW_COUNT, 0) or 0),
            description=payload.get(S.P_DESCRIPTION),
            features=payload.get(S.P_FEATURES),
            image_url=payload.get(S.P_IMAGE_URL),
        )

    def price_text(self) -> str:
        if self.price is None:
            return "chưa có giá"
        return format_vnd(self.price) + (" (giá tham khảo, chưa xác thực)" if self.price_estimated else "")

    def rating_text(self) -> str:
        if not self.review_count or self.avg_rating is None:
            return "chưa có đánh giá"
        return f"{self.avg_rating:.1f}/5 từ {self.review_count} đánh giá"


@dataclass
class EvidenceContext:
    """Numbered facts + review snippets with stable citation tags."""

    products: list[ProductFact] = field(default_factory=list)
    reviews: dict[str, list[tuple[str, Review]]] = field(default_factory=dict)  # product tag -> [(review tag, review)]
    extras: list[str] = field(default_factory=list)  # e.g. "Bộ lọc: giá ≤ 500.000₫"
    user_thresholds: list[float] = field(default_factory=list)  # numbers the *user* supplied (e.g. max price)

    @classmethod
    def build(
        cls,
        products: Sequence[tuple[int, Mapping[str, Any]]],
        reviews_by_product: Mapping[int, Sequence[Review]] | None = None,
        extras: Sequence[str] = (),
        user_thresholds: Sequence[float] = (),
    ) -> "EvidenceContext":
        ctx = cls(extras=list(extras), user_thresholds=list(user_thresholds))
        for i, (pid, payload) in enumerate(products, start=1):
            fact = ProductFact.from_payload(f"P{i}", pid, payload)
            ctx.products.append(fact)
            revs = (reviews_by_product or {}).get(pid, [])
            ctx.reviews[fact.tag] = [(f"R{i}.{j}", r) for j, r in enumerate(revs, start=1)]
        return ctx

    # ---- prompt rendering --------------------------------------------------------
    def render(self, max_desc: int = 450, max_feat: int = 300) -> str:
        lines: list[str] = []
        for note in self.extras:
            lines.append(f"[NOTE] {note}")
        for fact in self.products:
            lines.append(f"[{fact.tag}] {fact.title}")
            lines.append(f"  - Thương hiệu: {fact.brand or 'không rõ'}")
            lines.append(f"  - Loại: {fact.category or 'không rõ'}")
            lines.append(f"  - Giá: {fact.price_text()}")
            lines.append(f"  - Đánh giá tổng hợp: {fact.rating_text()}")
            if fact.features:
                lines.append(f"  - Đặc điểm: {fact.features[:max_feat]}")
            if fact.description:
                lines.append(f"  - Mô tả: {fact.description[:max_desc]}")
            for rtag, rev in self.reviews.get(fact.tag, []):
                title = f"{rev.title} — " if rev.title else ""
                lines.append(f"  - [{rtag}] ({rev.rating:.0f}★, {rev.helpful_vote} lượt hữu ích) {title}{rev.snippet(300)}")
            if not self.reviews.get(fact.tag):
                lines.append("  - (chưa có nhận xét nào của người mua trong dữ liệu)")
        return "\n".join(lines)

    # ---- grounding vocabulary ----------------------------------------------------
    def valid_tags(self) -> set[str]:
        tags = {f.tag for f in self.products}
        for revs in self.reviews.values():
            tags.update(t for t, _ in revs)
        return tags

    def allowed_amounts(self) -> list[float]:
        prices = [f.price for f in self.products if f.price is not None]
        allowed = list(prices) + list(self.user_thresholds)
        allowed += [abs(a - b) for a, b in combinations(prices, 2)]  # "rẻ hơn X₫"
        return allowed


# ---- grounding checks ------------------------------------------------------------
_TAG_RE = re.compile(r"\[(P\d+|R\d+\.\d+)\]")
_LINK_RE = re.compile(r"https?://\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|vn|io|co|info|example|xyz|shop|store)\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(₫|đồng|đ\b|vnđ|vnd|nghìn|ngàn|k\b|triệu|tr\b)",
    re.IGNORECASE,
)


def extract_citations(text: str) -> set[str]:
    return set(_TAG_RE.findall(text))


def _to_vnd(number: str, unit: str) -> float | None:
    unit = unit.lower()
    if unit in {"₫", "đ", "đồng", "vnđ", "vnd"}:
        digits = re.sub(r"[.,]", "", number)
        return float(digits) if digits.isdigit() else None
    try:
        value = float(number.replace(",", "."))
    except ValueError:
        return None
    if unit in {"nghìn", "ngàn", "k"}:
        return value * 1_000
    return value * 1_000_000  # triệu / tr


def check_grounding(answer: str, ctx: EvidenceContext, tolerance: float = 0.015) -> list[str]:
    """Return a list of grounding problems (empty == acceptable).

    Cheap, deterministic checks that catch the failure modes that matter for
    a shop assistant: (1) citing a product/review handle that was never provided,
    (2) stating a price that is not any provided price (or a difference between
    two provided prices, or a threshold the user themselves gave), (3) emitting a
    link/domain that is not in the provided data (typical prompt-injection payload).
    """
    issues: list[str] = []
    unknown = extract_citations(answer) - ctx.valid_tags()
    if unknown:
        issues.append(f"trích dẫn không tồn tại: {sorted(unknown)}")
    allowed = ctx.allowed_amounts()
    for number, unit in _AMOUNT_RE.findall(answer):
        amount = _to_vnd(number, unit)
        if amount is None:
            continue
        if not any(abs(amount - a) <= max(tolerance * a, 1_000) for a in allowed):
            issues.append(f"số tiền không có trong dữ liệu: {number} {unit}")
    context_text = ctx.render().lower()
    for link in _LINK_RE.findall(answer):
        if link.lower().rstrip(".,)") not in context_text:
            issues.append(f"đường dẫn/tên miền không có trong dữ liệu: {link}")
    return issues
