from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..auth.dependencies import admin_user
from ..auth.models import User
from ..core.database import get_db
from .models import Category, Product
from .schemas import CategoryIn, CategoryOut, PaginatedProductsOut, ProductIn, ProductOut

router = APIRouter(tags=["catalog"])


@router.get("/categories", response_model=list[CategoryOut])
def categories(db: Session = Depends(get_db)):
    return list(db.scalars(select(Category).order_by(Category.name)))


@router.post("/categories", response_model=CategoryOut, status_code=201)
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


@router.get("/products", response_model=PaginatedProductsOut)
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


@router.get("/products/{product_id}", response_model=ProductOut)
def product_detail(product_id: int, db: Session = Depends(get_db)):
    product = db.get(Product, product_id)
    if product is None or not product.is_active:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    return product


@router.post("/products", response_model=ProductOut, status_code=201)
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


@router.put("/products/{product_id}", response_model=ProductOut)
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


@router.delete("/products/{product_id}", status_code=204)
def deactivate_product(product_id: int, db: Session = Depends(get_db), _: User = Depends(admin_user)):
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    product.is_active = False
    db.commit()
