from decimal import Decimal
from pydantic import BaseModel, Field

from ..catalog.schemas import ProductOut


class CartItemIn(BaseModel):
    product_id: int
    quantity: int = Field(ge=1, le=100)


class CartItemOut(BaseModel):
    product: ProductOut
    quantity: int


class CartOut(BaseModel):
    items: list[CartItemOut]
    total_amount: Decimal
