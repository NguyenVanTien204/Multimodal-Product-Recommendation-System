from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth.dependencies import current_user
from ..auth.models import User
from ..catalog.models import Product
from ..core.database import get_db
from .models import CartItem
from .schemas import CartItemIn, CartItemOut, CartOut
from .service import cart_total, get_or_create_cart

router = APIRouter(prefix="/cart", tags=["cart"])


def _cart_out(cart) -> CartOut:
    return CartOut(items=[CartItemOut(product=item.product, quantity=item.quantity) for item in cart.items], total_amount=cart_total(cart))


@router.get("", response_model=CartOut)
def get_cart(user: User = Depends(current_user), db: Session = Depends(get_db)):
    cart = get_or_create_cart(db, user)
    db.commit()
    db.refresh(cart, attribute_names=["items"])
    for item in cart.items:
        _ = item.product
    return _cart_out(cart)


@router.post("/items", response_model=CartOut)
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


@router.delete("/items/{product_id}", status_code=204)
def remove_cart_item(product_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    cart = get_or_create_cart(db, user)
    item = db.scalar(select(CartItem).where(CartItem.cart_id == cart.id, CartItem.product_id == product_id))
    if item is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Cart item not found")
    db.delete(item)
    db.commit()
