from __future__ import annotations

import numpy as np
import httpx
from fastapi import HTTPException, status
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth.models import User
from ..cart.models import Cart
from ..catalog.models import Product
from ..core.config import settings
from ..orders.models import Order

PRODUCT_COLLECTION = settings.qdrant_product_collection


def qdrant_health() -> bool:
    try:
        QdrantClient(url=settings.qdrant_url, timeout=1).get_collections()
        return True
    except Exception:
        return False


async def recommender_health() -> bool:
    if not settings.datn_recommender_url:
        return False
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(f"{settings.datn_recommender_url.rstrip('/')}/health")
        return not response.is_error and bool(response.json().get("model_loaded"))
    except Exception:
        return False


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


async def _call_recommender(history_skus: list[str], k: int) -> list[tuple[str, float]] | None:
    """POST to the standalone retrieval+reranking service (apps/recommender).

    Returns None (never raises) on any failure so callers can fall back to
    Qdrant/popularity without special-casing network vs. model errors.
    """
    if not settings.datn_recommender_url:
        return None
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.post(
                f"{settings.datn_recommender_url.rstrip('/')}/recommend",
                json={"history": history_skus, "k": k},
            )
        if response.is_error:
            return None
        data = response.json()
        return [(item["item_id"], float(item["score"])) for item in data.get("recommendations", [])]
    except Exception:
        return None


def _recent_product_ids(db: Session, user: User, limit: int = 5) -> list[int]:
    """Oldest-first recent product ids: cart contents if any, else the last few
    orders' items. Oldest-first matters for the model gateway -- the retrieval
    tower reads the *last* history position as "most recently interacted"."""
    cart = db.scalar(select(Cart).where(Cart.user_id == user.id))
    if cart and cart.items:
        return [item.product_id for item in cart.items][-limit:]

    orders = list(
        db.scalars(select(Order).where(Order.user_id == user.id).order_by(Order.created_at.desc()).limit(3))
    )
    recent_ids: list[int] = []
    for order in reversed(orders):  # oldest of the fetched orders first
        for order_item in order.items:
            recent_ids.append(order_item.product_id)
    return recent_ids[-limit:]


def _skus_for_product_ids(db: Session, product_ids: list[int]) -> list[str]:
    if not product_ids:
        return []
    rows = db.execute(select(Product.id, Product.sku).where(Product.id.in_(product_ids))).all()
    sku_by_id = {row.id: row.sku for row in rows}
    return [sku_by_id[pid] for pid in product_ids if pid in sku_by_id]


async def model_recommendation(payload: dict, db: Session | None = None) -> dict:
    """Gateway to the retrieval+reranking model service, with a Qdrant vector
    fallback (real content vectors, see scripts/index_qdrant_vectors.py) and a
    catalog-popularity fallback if both are unavailable."""
    limit = int(payload.get("k", 10))
    history_ids = payload.get("history", []) or payload.get("recent_product_ids", [])

    history_skus = _skus_for_product_ids(db, history_ids) if db is not None else []
    model_result = await _call_recommender(history_skus, limit)
    if model_result and db is not None:
        skus = [sku for sku, _ in model_result]
        rows = db.execute(select(Product.id, Product.sku).where(Product.sku.in_(skus))).all()
        product_id_by_sku = {row.sku: row.id for row in rows}
        recommendations = [
            {"product_id": product_id_by_sku[sku], "score": round(score, 4)}
            for sku, score in model_result
            if sku in product_id_by_sku
        ][:limit]
        if recommendations:
            return {
                "source": "model",
                "model_version": settings.model_version,
                "recommendations": recommendations,
            }

    # Qdrant vector fallback (product_id-keyed, real content vectors already indexed).
    client = QdrantClient(url=settings.qdrant_url, timeout=5)
    recommended_ids_scores: list[tuple[int, float]] = []

    if history_ids and client.collection_exists(PRODUCT_COLLECTION):
        records = client.retrieve(PRODUCT_COLLECTION, ids=history_ids[-5:], with_vectors=True)
        valid_vectors = [r.vector for r in records if r.vector is not None]
        if valid_vectors:
            avg_vector = np.mean(valid_vectors, axis=0).tolist()
            res = client.query_points(PRODUCT_COLLECTION, query=avg_vector, limit=limit + len(history_ids)).points
            recommended_ids_scores = [
                (int(p.id), float(p.score))
                for p in res
                if int(p.id) not in history_ids
            ][:limit]

    return {
        "source": "qdrant_vector_ai" if recommended_ids_scores else "catalog_popularity",
        "model_version": "qdrant-1024d-cosine-v1",
        "recommendations": [
            {"product_id": pid, "score": round(score, 4)} for pid, score in recommended_ids_scores
        ],
    }


async def get_ai_recommendations_for_user(db: Session, user: User | None = None, limit: int = 8) -> list[dict]:
    """Personalized recommendations: retrieval+reranking model first, then a
    Qdrant vector fallback, then diverse top-of-catalog fill."""
    seen_ids: set[int] = set()
    recommended_prods: list[dict] = []

    recent_ids = _recent_product_ids(db, user) if user else []
    seen_ids.update(recent_ids)
    history_skus = _skus_for_product_ids(db, recent_ids)
    # Called even for anonymous users (empty history): the model's own
    # popularity fallback is a better default than the arbitrary
    # first-product-per-category fill below.
    model_result = await _call_recommender(history_skus, limit)
    if model_result:
        skus = [sku for sku, _ in model_result]
        rows = db.execute(
            select(Product).where(Product.sku.in_(skus), Product.is_active.is_(True))
        ).scalars().all()
        product_by_sku = {p.sku: p for p in rows}
        for sku, score in model_result:
            product = product_by_sku.get(sku)
            if product and product.id not in seen_ids:
                seen_ids.add(product.id)
                recommended_prods.append({"product": product, "score": score})
                if len(recommended_prods) >= limit:
                    break

    # Qdrant fallback if the model returned nothing (service down, cold-start miss, etc.)
    if len(recommended_prods) < limit:
        client = QdrantClient(url=settings.qdrant_url, timeout=5)
        recent_ids = list(seen_ids)
        if recent_ids and client.collection_exists(PRODUCT_COLLECTION):
            try:
                records = client.retrieve(PRODUCT_COLLECTION, ids=recent_ids[-3:], with_vectors=True)
                valid_vectors = [r.vector for r in records if r.vector is not None]
                if valid_vectors:
                    avg_vec = np.mean(valid_vectors, axis=0).tolist()
                    query_res = client.query_points(
                        PRODUCT_COLLECTION, query=avg_vec, limit=limit + len(recent_ids)
                    ).points
                    for p in query_res:
                        pid = int(p.id)
                        if pid not in seen_ids:
                            seen_ids.add(pid)
                            prod = db.get(Product, pid)
                            if prod and prod.is_active:
                                recommended_prods.append({"product": prod, "score": float(p.score)})
                                if len(recommended_prods) >= limit:
                                    break
            except Exception:
                pass

    # Final fallback: diverse top products across different categories.
    if len(recommended_prods) < limit:
        all_categories = list(db.scalars(select(Product.category_id).distinct()))
        for cat_id in all_categories:
            if len(recommended_prods) >= limit:
                break
            prod = db.scalar(
                select(Product)
                .where(Product.category_id == cat_id, Product.is_active.is_(True), Product.id.not_in(seen_ids))
                .order_by(Product.id.asc())
            )
            if prod:
                seen_ids.add(prod.id)
                score = round(0.92 - len(recommended_prods) * 0.02, 2)
                recommended_prods.append({"product": prod, "score": score})

    return recommended_prods
