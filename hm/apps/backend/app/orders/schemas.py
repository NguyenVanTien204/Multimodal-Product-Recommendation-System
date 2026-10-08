from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field


class CheckoutIn(BaseModel):
    shipping_address: str = Field(min_length=10, max_length=1000)


class OrderStatusIn(BaseModel):
    status: str = Field(pattern=r"^(PENDING|PAID|PROCESSING|SHIPPED|CANCELLED)$")


class OrderItemOut(BaseModel):
    product_id: int
    product_name: str
    unit_price: Decimal
    quantity: int


class OrderOut(BaseModel):
    id: int
    status: str
    shipping_address: str
    total_amount: Decimal
    created_at: datetime
    items: list[OrderItemOut]
