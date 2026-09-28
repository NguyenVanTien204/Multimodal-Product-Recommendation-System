from contextlib import asynccontextmanager

from datn.recommenders.reranker import RerankerPipeline
from datn.recommenders.user_tower.content import ContentSource
from fastapi import FastAPI, HTTPException

from .config import settings
from .schemas import HealthOut, RecommendationOut, RecommendIn, RecommendOut

state: dict[str, RerankerPipeline] = {}


def _load_pipeline() -> RerankerPipeline:
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["pipeline"] = _load_pipeline()
    yield
    state.clear()


app = FastAPI(title="DATN Recommender Service", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthOut)
def health():
    pipeline = state.get("pipeline")
    return HealthOut(
        status="ok" if pipeline else "loading",
        model_loaded=pipeline is not None,
        model_version=settings.model_version,
        catalog_size=pipeline.vocab.num_items if pipeline else None,
    )


@app.post("/recommend", response_model=RecommendOut)
def recommend(payload: RecommendIn):
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(503, "Model is not loaded yet")

    k = min(payload.k, settings.max_k)
    history = payload.history[-settings.max_history :]
    result = pipeline.recommend(history, k=k)
    return RecommendOut(
        source=result.source,
        model_version=settings.model_version,
        recommendations=[RecommendationOut(item_id=r.item_id, score=r.score) for r in result.recommendations],
    )
