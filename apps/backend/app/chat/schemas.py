from typing import Any

from pydantic import BaseModel, Field

from ..catalog.schemas import ProductOut


class ChatFilters(BaseModel):
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)
    brands: list[str] = Field(default_factory=list)
    min_rating: float | None = Field(default=None, ge=1, le=5)
    audiences: list[str] = Field(default_factory=list)
    colours: list[str] = Field(default_factory=list)


class ChatAction(BaseModel):
    type: str = Field(pattern="^(explain|compare|similar|recommend|feedback|forget)$")
    product_id: int | None = None
    product_ids: list[int] = Field(default_factory=list)
    kind: str | None = Field(default=None, pattern="^(like|dislike)$", description="feedback only")


class ChatIn(BaseModel):
    message: str = Field(default="", max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)
    image_base64: str | None = Field(default=None, max_length=12_000_000)
    action: ChatAction | None = None
    filters: ChatFilters | None = None


class EvidenceOut(BaseModel):
    tag: str | None = None
    review_id: int
    rating: float
    helpful_vote: int
    title: str | None = None
    text: str
    is_mock: bool = False


class ChatProductOut(BaseModel):
    product: ProductOut
    score: float = 0.0
    brand: str | None = None
    price_estimated: bool = False
    avg_rating: float | None = None
    review_count: int = 0
    reviews_mock: bool = False
    audience: str | None = None
    colour: str | None = None
    product_type: str | None = None
    reasons: list[str] = Field(default_factory=list)
    evidence: list[EvidenceOut] = Field(default_factory=list)


class ChatOut(BaseModel):
    session_id: str
    reply: str
    action: str
    lang: str = "vi"
    products: list[ChatProductOut] = Field(default_factory=list)
    filter_chips: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
