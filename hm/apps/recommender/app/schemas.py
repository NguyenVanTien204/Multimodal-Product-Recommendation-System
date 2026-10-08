from pydantic import BaseModel, Field


class RecommendIn(BaseModel):
    history: list[str] = Field(default_factory=list, description="Catalog SKUs (Amazon ASIN or H&M article_id), most-recent last")
    k: int = Field(default=10, ge=1, le=100)
    exclude_history: bool = Field(default=True, description="H&M engine: drop items already in the history (repeat purchases are valid in the data)")
    cold_every: int | None = Field(default=None, ge=0, le=50, description="H&M engine: override the server default of reserving every Nth slot for never-sold items")


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
