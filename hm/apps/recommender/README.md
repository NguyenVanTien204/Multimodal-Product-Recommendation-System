# DATN Recommender Service

Standalone FastAPI service that serves the trained retrieval (User Tower) and
reranking (`ResidualListwiseRanker`, `data/artifacts/reranker_v2`) models over
HTTP. It is the only piece of the web stack allowed to import `datn` (the
`hm/src/` package; `datn_legacy` from `legacy/src/` serves the Amazon engine) and read Parquet/checkpoint artifacts directly — see
`hm/docs/web/06-frontend-architecture.md` section 1. `hm/apps/backend` never imports
`datn`; it calls this service through `DATN_RECOMMENDER_URL` and falls back to
Qdrant/catalog popularity when it is unavailable.

Core inference logic lives in `legacy/src/datn_legacy/recommenders/reranker/` (candidate
generation, the reranker model, and the `RerankerPipeline` orchestration) so it
stays reusable from legacy/notebooks/offline evaluation, not just this service. This
app is a thin HTTP wrapper: `app/main.py` loads a `RerankerPipeline` once at
startup and exposes it as `POST /recommend`.

## API

- `GET /health` — `{status, model_loaded, model_version, catalog_size}`.
- `POST /recommend` — `{history: string[] (catalog SKUs/ASINs, most-recent last), k: int}`
  → `{source: "reranker" | "popularity_fallback", model_version, recommendations: [{item_id, score}]}`.
  Empty or fully-unknown `history` returns a popularity-ranked fallback instead
  of erroring, so callers never need to special-case cold start.

`item_id` is the catalog SKU (Amazon ASIN), matching `Product.sku` in the
marketplace database — `hm/apps/backend` is responsible for translating between
its own `Product.id` and this SKU on both sides of the call.

## Running via Docker Compose

Already wired into `hm/apps/backend/docker-compose.yml` as the `recommender`
service. `docker compose up --build` builds it with the **repo root** as
build context (so it can install `datn` from `hm/src/` and `datn_legacy` from `legacy/src/`) and mounts
`data/` read-only at `/data`. Set `RECOMMENDER_DEVICE=cuda` in `.env` to use a
GPU if the Docker host has one; default is `cpu`.

## Running locally (no Docker)

Requires the same Python environment used for training (`pip install -e
".[recommenders]"` from the repo root gives you `datn` + torch). From the repo
root:

```powershell
$env:DATN_DATA_DIR = "./data"
uvicorn app.main:app --app-dir hm/apps/recommender --host 127.0.0.1 --port 8100
```

Then point the marketplace backend's `.env` at it:
`DATN_RECOMMENDER_URL=http://localhost:8100`.

## Artifacts this service expects

| Setting | Default (relative to `DATN_DATA_DIR`) | Source |
|---|---|---|
| User Tower checkpoint | `artifacts/user_tower_balanced_v1/` | `datn-user-tower train` |
| Reranker checkpoint | `artifacts/reranker_v2/` | `legacy/notebooks/reranker_training.ipynb` |
| Item catalog / splits | `processed/balanced_u5_i2_v1/` | `datn-balanced-data` |
| Image/text embeddings | `embedding/{image,text}_embeddings.npy` + metadata | `legacy/notebooks/kaggle_*_embeddings.ipynb` |

All configurable via `DATN_*` environment variables — see `app/config.py`.
