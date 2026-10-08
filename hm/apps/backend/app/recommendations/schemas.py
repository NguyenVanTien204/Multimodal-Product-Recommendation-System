from pydantic import BaseModel, Field


class ProductVectorIn(BaseModel):
    product_id: int
    vector: list[float] = Field(min_length=1, max_length=4096)


class SimilarProductOut(BaseModel):
    product_id: int
    score: float
