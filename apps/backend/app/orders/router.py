from decimal import Decimal
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..auth.dependencies import admin_user, current_user
from ..auth.models import User
from ..cart.models import CartItem
from ..cart.service import get_or_create_cart
from ..catalog.models import Product
from ..core.database import get_db
from .models import Order, OrderItem
from .schemas import CheckoutIn, OrderItemOut, OrderOut, OrderStatusIn

router = APIRouter(prefix="/orders", tags=["orders"])


@router.post("/checkout", response_model=OrderOut, status_code=201)
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


@router.get("", response_model=list[OrderOut])
def orders(user: User = Depends(current_user), db: Session = Depends(get_db)):
    result = db.scalars(select(Order).where(Order.user_id == user.id).options(selectinload(Order.items)).order_by(Order.created_at.desc()))
    return [OrderOut(id=o.id, status=o.status, shipping_address=o.shipping_address, total_amount=o.total_amount, created_at=o.created_at, items=[OrderItemOut.model_validate(i, from_attributes=True) for i in o.items]) for o in result]


@router.patch("/{order_id}/status", response_model=OrderOut)
def update_order_status(order_id: int, payload: OrderStatusIn, db: Session = Depends(get_db), _: User = Depends(admin_user)):
    order = db.scalar(select(Order).where(Order.id == order_id).options(selectinload(Order.items)))
    if order is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Order not found")
    order.status = payload.status
    db.commit()
    db.refresh(order, attribute_names=["items"])
    return OrderOut(id=order.id, status=order.status, shipping_address=order.shipping_address, total_amount=order.total_amount, created_at=order.created_at, items=[OrderItemOut.model_validate(i, from_attributes=True) for i in order.items])
