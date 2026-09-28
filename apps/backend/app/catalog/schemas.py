from decimal import Decimal
from pydantic import BaseModel, ConfigDict, Field


class CategoryIn(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    slug: str = Field(pattern=r"^[a-z0-9-]+$")


class CategoryOut(CategoryIn):
    model_config = ConfigDict(from_attributes=True)
    id: int


class ProductIn(BaseModel):
    sku: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=2, max_length=255)
    description: str = ""
    price: Decimal = Field(gt=0)
    stock_quantity: int = Field(ge=0)
    image_url: str | None = None
    category_id: int | None = None


class ProductOut(ProductIn):
    model_config = ConfigDict(from_attributes=True)
    id: int
    is_active: bool


class PaginatedProductsOut(BaseModel):
    items: list[ProductOut]
    total: int
    page: int
    page_size: int
    total_pages: int
