from __future__ import annotations

from decimal import Decimal
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from .database import get_db
from .dependencies import admin_user, current_user
from .models import CartItem, Category, Order, OrderItem, Product, User
from .schemas import (
    CartItemIn, CartItemOut, CartOut, CategoryIn, CategoryOut,
    CheckoutIn, LoginIn, OrderItemOut, OrderOut, OrderStatusIn,
    PaginatedProductsOut, ProductIn, ProductOut, ProductVectorIn,
    RegisterIn, SimilarProductOut, TokenOut, UserOut
)
from .security import create_access_token, hash_password, verify_password
from .services import (
    cart_total, get_ai_recommendations_for_user, get_or_create_cart,
    model_recommendation, qdrant_health, similar_product_ids, upsert_product_vector
)


auth_router = APIRouter(prefix="/auth", tags=["auth"])
catalog_router = APIRouter(tags=["catalog"])
cart_router = APIRouter(prefix="/cart", tags=["cart"])
order_router = APIRouter(prefix="/orders", tags=["orders"])
recommendation_router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@auth_router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    if db.scalar(select(User).where(User.email == payload.email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Email is already registered")
    user = User(email=str(payload.email), full_name=payload.full_name, password_hash=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@auth_router.post("/login", response_model=TokenOut)
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == payload.email))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    return TokenOut(access_token=create_access_token(user.id))


@auth_router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user


@catalog_router.get("/categories", response_model=list[CategoryOut])
def categories(db: Session = Depends(get_db)):
    return list(db.scalars(select(Category).order_by(Category.name)))


@catalog_router.post("/categories", response_model=CategoryOut, status_code=201)
def create_category(payload: CategoryIn, db: Session = Depends(get_db), _: User = Depends(admin_user)):
    category = Category(**payload.model_dump())
    db.add(category)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Category name or slug already exists") from exc
    db.refresh(category)
    return category


@catalog_router.get("/products", response_model=PaginatedProductsOut)
def products(
    q: str | None = None,
    category_id: int | None = None,
    page: int = 1,
    page_size: int = 24,
    db: Session = Depends(get_db)
):
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 24

    base_query = select(Product).where(Product.is_active.is_(True))
    count_query = select(func.count(Product.id)).where(Product.is_active.is_(True))

    if q:
        search_filter = Product.name.ilike(f"%{q.strip()}%")
        base_query = base_query.where(search_filter)
        count_query = count_query.where(search_filter)

    if category_id:
        base_query = base_query.where(Product.category_id == category_id)
        count_query = count_query.where(Product.category_id == category_id)

    total = db.scalar(count_query) or 0
    total_pages = max(1, (total + page_size - 1) // page_size)

    offset = (page - 1) * page_size
    items = list(db.scalars(base_query.order_by(Product.id.asc()).offset(offset).limit(page_size)))

    return PaginatedProductsOut(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )




@catalog_router.get("/products/{product_id}", response_model=ProductOut)
def product_detail(product_id: int, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if product is None or not product.is_active:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    return product


@catalog_router.post("/products", response_model=ProductOut, status_code=201)
def create_product(payload: ProductIn, db: Session = Depends(get_db), _: User = Depends(admin_user)):
    if payload.category_id and db.get(Category, payload.category_id) is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Category does not exist")
    product = Product(**payload.model_dump())
    db.add(product)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "SKU already exists") from exc
    db.refresh(product)
    return product


@catalog_router.put("/products/{product_id}", response_model=ProductOut)
def update_product(product_id: int, payload: ProductIn, db: Session = Depends(get_db), _: User = Depends(admin_user)):
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    if payload.category_id and db.get(Category, payload.category_id) is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Category does not exist")
    for key, value in payload.model_dump().items():
        setattr(product, key, value)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "SKU already exists") from exc
    db.refresh(product)
    return product


@catalog_router.delete("/products/{product_id}", status_code=204)
def deactivate_product(product_id: int, db: Session = Depends(get_db), _: User = Depends(admin_user)):
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    product.is_active = False
    db.commit()


def _cart_out(cart) -> CartOut:
    return CartOut(items=[CartItemOut(product=item.product, quantity=item.quantity) for item in cart.items], total_amount=cart_total(cart))


@cart_router.get("", response_model=CartOut)
def get_cart(user: User = Depends(current_user), db: Session = Depends(get_db)):
    cart = get_or_create_cart(db, user)
    db.commit()
    db.refresh(cart, attribute_names=["items"])
    for item in cart.items:
        _ = item.product
    return _cart_out(cart)


@cart_router.post("/items", response_model=CartOut)
def add_cart_item(payload: CartItemIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    product = db.get(Product, payload.product_id)
    if product is None or not product.is_active:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    if payload.quantity > product.stock_quantity:
        raise HTTPException(status.HTTP_409_CONFLICT, "Requested quantity exceeds stock")
    cart = get_or_create_cart(db, user)
    item = db.scalar(select(CartItem).where(CartItem.cart_id == cart.id, CartItem.product_id == product.id))
    if item:
        if item.quantity + payload.quantity > product.stock_quantity:
            raise HTTPException(status.HTTP_409_CONFLICT, "Requested quantity exceeds stock")
        item.quantity += payload.quantity
    else:
        db.add(CartItem(cart_id=cart.id, product_id=product.id, quantity=payload.quantity))
    db.commit()
    return get_cart(user, db)


@cart_router.delete("/items/{product_id}", status_code=204)
def remove_cart_item(product_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    cart = get_or_create_cart(db, user)
    item = db.scalar(select(CartItem).where(CartItem.cart_id == cart.id, CartItem.product_id == product_id))
    if item is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Cart item not found")
    db.delete(item)
    db.commit()


@order_router.post("/checkout", response_model=OrderOut, status_code=201)
def checkout(payload: CheckoutIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    cart = get_or_create_cart(db, user)
    items = list(db.scalars(select(CartItem).where(CartItem.cart_id == cart.id).options(selectinload(CartItem.product))))
    if not items:
        raise HTTPException(status.HTTP_409_CONFLICT, "Cart is empty")
    product_ids = [item.product_id for item in items]
    locked = {product.id: product for product in db.scalars(select(Product).where(Product.id.in_(product_ids)).with_for_update())}
    for item in items:
        product = locked[item.product_id]
        if not product.is_active or product.stock_quantity < item.quantity:
            raise HTTPException(status.HTTP_409_CONFLICT, f"Insufficient stock for product {product.id}")
    total = sum((locked[item.product_id].price * item.quantity for item in items), start=Decimal("0"))
    order = Order(user_id=user.id, shipping_address=payload.shipping_address, total_amount=total)
    db.add(order)
    db.flush()
    for item in items:
        product = locked[item.product_id]
        product.stock_quantity -= item.quantity
        db.add(OrderItem(order_id=order.id, product_id=product.id, product_name=product.name, unit_price=product.price, quantity=item.quantity))
        db.delete(item)
    db.commit()
    db.refresh(order, attribute_names=["items"])
    return OrderOut(id=order.id, status=order.status, shipping_address=order.shipping_address, total_amount=order.total_amount, created_at=order.created_at, items=[OrderItemOut.model_validate(item, from_attributes=True) for item in order.items])


@order_router.get("", response_model=list[OrderOut])
def orders(user: User = Depends(current_user), db: Session = Depends(get_db)):
    result = db.scalars(select(Order).where(Order.user_id == user.id).options(selectinload(Order.items)).order_by(Order.created_at.desc()))
    return [OrderOut(id=o.id, status=o.status, shipping_address=o.shipping_address, total_amount=o.total_amount, created_at=o.created_at, items=[OrderItemOut.model_validate(i, from_attributes=True) for i in o.items]) for o in result]


@order_router.patch("/{order_id}/status", response_model=OrderOut)
def update_order_status(order_id: int, payload: OrderStatusIn, db: Session = Depends(get_db), _: User = Depends(admin_user)):
    order = db.scalar(select(Order).where(Order.id == order_id).options(selectinload(Order.items)))
    if order is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Order not found")
    order.status = payload.status
    db.commit()
    db.refresh(order, attribute_names=["items"])
    return OrderOut(id=order.id, status=order.status, shipping_address=order.shipping_address, total_amount=order.total_amount, created_at=order.created_at, items=[OrderItemOut.model_validate(i, from_attributes=True) for i in order.items])


@recommendation_router.get("/health")
def recommendation_health():
    return {"qdrant_available": qdrant_health()}


@recommendation_router.put("/vectors", status_code=204)
def index_product_vector(payload: ProductVectorIn, db: Session = Depends(get_db), _: User = Depends(admin_user)):
    if db.get(Product, payload.product_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    upsert_product_vector(payload.product_id, payload.vector)


@recommendation_router.get("/products/{product_id}/similar", response_model=list[SimilarProductOut])
def similar_products(product_id: int, limit: int = 10):
    if not 1 <= limit <= 50:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "limit must be between 1 and 50")
    return [SimilarProductOut(product_id=item_id, score=score) for item_id, score in similar_product_ids(product_id, limit)]


@recommendation_router.get("/for-you")
def recommendations_for_you(limit: int = 8, db: Session = Depends(get_db)):
    """Personalized or diverse AI recommendations powered by Qdrant vector space."""
    recs = get_ai_recommendations_for_user(db=db, limit=limit)
    return [
        {
            "product": ProductOut.model_validate(item["product"], from_attributes=True),
            "score": round(item["score"], 4),
        }
        for item in recs
    ]


@recommendation_router.post("/sequential")
async def sequential_recommendation(payload: dict, db: Session = Depends(get_db)):
    """Gateway to Amazon sequential recommender service with Qdrant vector fallback."""
    return await model_recommendation(payload, db)

