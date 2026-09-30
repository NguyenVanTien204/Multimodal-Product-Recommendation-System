from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FiltersIn(BaseModel):
    min_price: float | None = Field(default=None, ge=0, description="VND")
    max_price: float | None = Field(default=None, ge=0, description="VND")
    brands: list[str] = Field(default_factory=list)
    category_slugs: list[str] = Field(default_factory=list)
    min_rating: float | None = Field(default=None, ge=1, le=5)
    exclude_product_ids: list[int] = Field(default_factory=list)


class ChatAction(BaseModel):
    type: str = Field(pattern="^(explain|compare|similar|recommend)$")
    product_id: int | None = None
    product_ids: list[int] = Field(default_factory=list)


class ChatIn(BaseModel):
    message: str = Field(default="", max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)
    image_base64: str | None = Field(default=None, description="Raw base64 or a data: URL of the uploaded photo")
    history_skus: list[str] = Field(default_factory=list, description="Shop cart/order SKUs, oldest first")
    action: ChatAction | None = None
    filters: FiltersIn | None = None
    k: int | None = Field(default=None, ge=1, le=12)


class SearchIn(BaseModel):
    query: str = Field(default="", max_length=500)
    image_base64: str | None = None
    session_id: str | None = None
    filters: FiltersIn | None = None
    history_skus: list[str] = Field(default_factory=list)
    k: int | None = Field(default=None, ge=1, le=12)


class RecommendIn(BaseModel):
    history_skus: list[str] = Field(default_factory=list)
    session_id: str | None = None
    filters: FiltersIn | None = None
    k: int | None = Field(default=None, ge=1, le=12)


class RefineIn(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=500)
    k: int | None = Field(default=None, ge=1, le=12)


class ExplainIn(BaseModel):
    product_id: int
    question: str = Field(default="", max_length=500)
    session_id: str | None = None


class CompareIn(BaseModel):
    product_ids: list[int] = Field(min_length=2, max_length=4)
    question: str = Field(default="", max_length=500)
    session_id: str | None = None


class EvidenceOut(BaseModel):
    tag: str | None = None
    review_id: int
    rating: float
    helpful_vote: int
    title: str | None = None
    text: str
    relevance: float = 0.0


class ProductOut(BaseModel):
    product_id: int
    sku: str
    title: str
    brand: str | None = None
    category: str | None = None
    price: float | None = None
    price_estimated: bool = False
    image_url: str | None = None
    avg_rating: float | None = None
    review_count: int = 0
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    evidence: list[EvidenceOut] = Field(default_factory=list)
    signals: dict[str, Any] = Field(default_factory=dict)


class ChatOut(BaseModel):
    session_id: str
    reply: str
    action: str
    lang: str
    products: list[ProductOut] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    filter_chips: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class HealthOut(BaseModel):
    status: str
    encoder_loaded: bool
    encoder_ok: bool
    llm: str
    llm_status: dict[str, Any] = Field(default_factory=dict)
    collections: dict[str, int]
    recommender: dict[str, Any]
    sessions: int
