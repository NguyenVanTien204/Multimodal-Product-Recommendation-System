from __future__ import annotations

from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth.models import User
from .models import Cart


def get_or_create_cart(db: Session, user: User) -> Cart:
    cart = db.scalar(select(Cart).where(Cart.user_id == user.id))
    if cart is None:
        cart = Cart(user_id=user.id)
        db.add(cart)
        db.flush()
    return cart


def cart_total(cart: Cart) -> Decimal:
    return sum((item.product.price * item.quantity for item in cart.items), start=Decimal("0"))
