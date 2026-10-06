from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from .config import settings
from .schemas import HealthOut, RecommendationOut, RecommendIn, RecommendOut

state: dict = {}


def _load_amazon():
    from datn.recommenders.reranker import RerankerPipeline
    from datn.recommenders.user_tower.content import ContentSource

    content_sources = [
        ContentSource("image", settings.image_embeddings_path, settings.image_metadata_path),
        ContentSource("text", settings.text_embeddings_path, settings.text_metadata_path),
    ]
    return RerankerPipeline.from_artifacts(
        user_tower_dir=settings.user_tower_dir,
        reranker_dir=settings.reranker_dir,
        items_path=settings.items_path,
        train_path=settings.train_path,
        valid_path=settings.valid_path,
        test_path=settings.test_path,
        content_sources=content_sources,
        device=settings.device,
    )


def _load_hm():
    from datn.recommenders.hm import HMRecommender

    d = settings.hm_path
    device = "cpu" if settings.device == "auto" else settings.device
    rec = HMRecommender.from_artifacts(
        tower_path=d / "tower.pt", ranker_path=d / "reranker_lgbm.txt", items_path=d / "items.parquet",
        text_path=d / "text_embeddings.npy", image_path=d / "image_embeddings.npy",
        daily_counts_path=d / "item_daily_counts.parquet", as_of=settings.hm_as_of, device=device,
    )
    # Optional: only recommend article_ids that exist in the shop (written by the catalog import).
    allowed = None
    ids_path = settings.hm_shop_ids_path
    if ids_path.is_file():
        shop = {line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        allowed = np.zeros(rec.n_items + 1, dtype=bool)
        for item_id in shop:
            idx = rec.item2idx.get(item_id)
            if idx:
                allowed[idx] = True
    state["hm_allowed"] = allowed
    return rec


def _version() -> str:
    return settings.hm_model_version if settings.engine == "hm" else settings.model_version


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.engine not in {"amazon", "hm"}:
        raise RuntimeError(f"DATN_ENGINE must be 'amazon' or 'hm', got {settings.engine!r}")
    state["pipeline"] = _load_hm() if settings.engine == "hm" else _load_amazon()
    yield
    state.clear()


app = FastAPI(title="DATN Recommender Service", version="1.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthOut)
def health():
    pipeline = state.get("pipeline")
    size = None
    if pipeline is not None:
        size = pipeline.n_items if settings.engine == "hm" else pipeline.vocab.num_items
    return HealthOut(
        status="ok" if pipeline else "loading",
        model_loaded=pipeline is not None,
        model_version=_version(),
        catalog_size=size,
    )


@app.post("/recommend", response_model=RecommendOut)
def recommend(payload: RecommendIn):
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(503, "Model is not loaded yet")

    k = min(payload.k, settings.max_k)
    history = payload.history[-settings.max_history :]
    if settings.engine == "hm":
        cold_every = settings.hm_cold_every if payload.cold_every is None else payload.cold_every
        result = pipeline.recommend(
            history, k=k, cold_every=cold_every, exclude_history=payload.exclude_history, allowed=state.get("hm_allowed")
        )
    else:
        result = pipeline.recommend(history, k=k)
    return RecommendOut(
        source=result.source,
        model_version=_version(),
        recommendations=[RecommendationOut(item_id=r.item_id, score=r.score) for r in result.recommendations],
    )
