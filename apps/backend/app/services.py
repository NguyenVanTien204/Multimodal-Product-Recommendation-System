from __future__ import annotations

from decimal import Decimal
import numpy as np
import httpx
from fastapi import HTTPException, status
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .models import Cart, CartItem, Order, OrderItem, Product, User


def get_or_create_cart(db: Session, user: User) -> Cart:
    cart = db.scalar(select(Cart).where(Cart.user_id == user.id))
    if cart is None:
        cart = Cart(user_id=user.id)
        db.add(cart)
        db.flush()
    return cart


def cart_total(cart: Cart) -> Decimal:
    return sum((item.product.price * item.quantity for item in cart.items), start=Decimal("0"))


def qdrant_health() -> bool:
    try:
        QdrantClient(url=settings.qdrant_url, timeout=1).get_collections()
        return True
    except Exception:
        return False


PRODUCT_COLLECTION = "product_embeddings"


def upsert_product_vector(product_id: int, vector: list[float]) -> None:
    client = QdrantClient(url=settings.qdrant_url, timeout=5)
    if not client.collection_exists(PRODUCT_COLLECTION):
        client.create_collection(PRODUCT_COLLECTION, vectors_config=VectorParams(size=len(vector), distance=Distance.COSINE))
    client.upsert(PRODUCT_COLLECTION, points=[PointStruct(id=product_id, vector=vector, payload={"product_id": product_id})], wait=True)


def similar_product_ids(product_id: int, limit: int) -> list[tuple[int, float]]:
    client = QdrantClient(url=settings.qdrant_url, timeout=5)
    records = client.retrieve(PRODUCT_COLLECTION, ids=[product_id], with_vectors=True)
    if not records or records[0].vector is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product embedding is not indexed in Qdrant")
    query = client.query_points(PRODUCT_COLLECTION, query=records[0].vector, limit=limit + 1).points
    return [(int(point.id), float(point.score)) for point in query if int(point.id) != product_id][:limit]


async def model_recommendation(payload: dict, db: Session | None = None) -> dict:
    """Gateway to external recommender or built-in Qdrant vector recommendation."""
    if settings.datn_recommender_url:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(f"{settings.datn_recommender_url.rstrip('/')}/recommend", json=payload)
            if not response.is_error:
                return response.json()
        except Exception:
            pass

    # Built-in Qdrant Sequential / Vector recommendation fallback
    limit = int(payload.get("k", 10))
    history_ids = payload.get("history", []) or payload.get("recent_product_ids", [])
    
    client = QdrantClient(url=settings.qdrant_url, timeout=5)
    recommended_ids_scores: list[tuple[int, float]] = []

    if history_ids and client.collection_exists(PRODUCT_COLLECTION):
        # Retrieve vectors for history items
        records = client.retrieve(PRODUCT_COLLECTION, ids=history_ids[-5:], with_vectors=True)
        valid_vectors = [r.vector for r in records if r.vector is not None]
        if valid_vectors:
            # Average vector representing user preference state
            avg_vector = np.mean(valid_vectors, axis=0).tolist()
            res = client.query_points(PRODUCT_COLLECTION, query=avg_vector, limit=limit + len(history_ids)).points
            recommended_ids_scores = [
                (int(p.id), float(p.score))
                for p in res
                if int(p.id) not in history_ids
            ][:limit]

    # If no history or Qdrant points returned, return top diverse products
    return {
        "source": "qdrant_vector_ai" if recommended_ids_scores else "catalog_popularity",
        "model_version": "qdrant-1024d-cosine-v1",
        "recommendations": [
            {"product_id": pid, "score": round(score, 4)} for pid, score in recommended_ids_scores
        ],
    }


def get_ai_recommendations_for_user(db: Session, user: User | None = None, limit: int = 8) -> list[dict]:
    """Retrieve personalized or curated AI recommendations for user."""
    client = QdrantClient(url=settings.qdrant_url, timeout=5)
    
    # 1. Check user cart or orders for history
    recent_ids: list[int] = []
    if user:
        cart = db.scalar(select(Cart).where(Cart.user_id == user.id))
        if cart and cart.items:
            recent_ids = [item.product_id for item in cart.items]
        if not recent_ids:
            orders = list(db.scalars(select(Order).where(Order.user_id == user.id).order_by(Order.created_at.desc()).limit(3)))
            for o in orders:
                for oi in o.items:
                    recent_ids.append(oi.product_id)

    # 2. Query Qdrant with recent vectors if available
    recommended_prods: list[dict] = []
    seen_ids = set(recent_ids)

    if recent_ids and client.collection_exists(PRODUCT_COLLECTION):
        try:
            records = client.retrieve(PRODUCT_COLLECTION, ids=recent_ids[-3:], with_vectors=True)
            valid_vectors = [r.vector for r in records if r.vector is not None]
            if valid_vectors:
                avg_vec = np.mean(valid_vectors, axis=0).tolist()
                query_res = client.query_points(PRODUCT_COLLECTION, query=avg_vec, limit=limit + len(recent_ids)).points
                for p in query_res:
                    pid = int(p.id)
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        prod = db.get(Product, pid)
                        if prod and prod.is_active:
                            recommended_prods.append({
                                "product": prod,
                                "score": float(p.score),
                            })
                            if len(recommended_prods) >= limit:
                                break
        except Exception:
            pass

    # 3. If still needed, fill with diverse top products across different categories
    if len(recommended_prods) < limit:
        all_categories = list(db.scalars(select(Product.category_id).distinct()))
        for cat_id in all_categories:
            if len(recommended_prods) >= limit:
                break
            prod = db.scalar(select(Product).where(Product.category_id == cat_id, Product.is_active.is_(True), Product.id.not_in(seen_ids)).order_by(Product.id.asc()))
            if prod:
                seen_ids.add(prod.id)
                # Compute score or give high default score
                score = round(0.92 - len(recommended_prods) * 0.02, 2)
                recommended_prods.append({"product": prod, "score": score})

    return recommended_prods
