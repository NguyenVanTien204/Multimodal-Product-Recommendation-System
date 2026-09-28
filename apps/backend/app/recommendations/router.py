from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..auth.dependencies import admin_user, optional_user
from ..auth.models import User
from ..catalog.models import Product
from ..catalog.schemas import ProductOut
from ..core.database import get_db
from .schemas import ProductVectorIn, SimilarProductOut
from .service import (
    get_ai_recommendations_for_user, model_recommendation,
    qdrant_health, recommender_health, similar_product_ids, upsert_product_vector
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/health")
async def recommendation_health():
    return {"qdrant_available": qdrant_health(), "recommender_available": await recommender_health()}


@router.put("/vectors", status_code=204)
def index_product_vector(payload: ProductVectorIn, db: Session = Depends(get_db), _: User = Depends(admin_user)):
    if db.get(Product, payload.product_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    upsert_product_vector(payload.product_id, payload.vector)


@router.get("/products/{product_id}/similar", response_model=list[SimilarProductOut])
def similar_products(product_id: int, limit: int = 10):
    if not 1 <= limit <= 50:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "limit must be between 1 and 50")
    return [SimilarProductOut(product_id=item_id, score=score) for item_id, score in similar_product_ids(product_id, limit)]


@router.get("/for-you")
async def recommendations_for_you(limit: int = 8, user: User | None = Depends(optional_user), db: Session = Depends(get_db)):
    """Personalized recommendations: retrieval+reranking model first, Qdrant/catalog fallback."""
    recs = await get_ai_recommendations_for_user(db=db, user=user, limit=limit)
    return [
        {
            "product": ProductOut.model_validate(item["product"], from_attributes=True),
            "score": round(item["score"], 4),
        }
        for item in recs
    ]


@router.post("/sequential")
async def sequential_recommendation(payload: dict, db: Session = Depends(get_db)):
    """Gateway to the retrieval+reranking model service, with Qdrant vector fallback."""
    return await model_recommendation(payload, db)
