from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=2, max_length=120)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: EmailStr
    full_name: str
    is_admin: bool


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



class CartItemIn(BaseModel):
    product_id: int
    quantity: int = Field(ge=1, le=100)


class CartItemOut(BaseModel):
    product: ProductOut
    quantity: int


class CartOut(BaseModel):
    items: list[CartItemOut]
    total_amount: Decimal


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


class ProductVectorIn(BaseModel):
    product_id: int
    vector: list[float] = Field(min_length=1, max_length=4096)


class SimilarProductOut(BaseModel):
    product_id: int
    score: float
