from pydantic import BaseModel, Field


class RecommendIn(BaseModel):
    history: list[str] = Field(default_factory=list, description="Catalog SKUs (ASIN), most-recent last")
    k: int = Field(default=10, ge=1, le=100)


class RecommendationOut(BaseModel):
    item_id: str
    score: float


class RecommendOut(BaseModel):
    source: str
    model_version: str
    recommendations: list[RecommendationOut]


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    catalog_size: int | None = None
