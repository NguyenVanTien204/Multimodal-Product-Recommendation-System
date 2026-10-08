# Decision log

## 2026-09-22 — H&M notebook architecture

- Retain all pre-existing Amazon/Coveo code and notebooks; H&M is an additive benchmark.
- Use a deterministic activity-stratified customer sample based only on history before holdout windows.
- Use two seven-day development windows before validation/test so retrieval and reranker checkpoint selection do not inspect test labels.
- Use persistent Jina CLIP v2 512-dimensional embeddings, downloaded and cached by the dedicated Kaggle embedding notebook; never silently approximate vision features with random weights.

## 2026-10-05 — H&M becomes the primary thesis dataset; model and evaluation decisions

- Project owner decision: H&M replaces Amazon as the primary dataset for the thesis; all Amazon/Coveo docs and code are retained as reference. Open follow-ups (roles of the two datasets, RAG without reviews, web demo data, porting inference to `src/`) are listed in `docs/hm/05_thesis_plan.md`, section 6.
- Cold-start at model level is handled by ID-dropout (0.3) during training; inference-time ID zeroing is kept as an option but not selected (threshold 0). Reason: without ID-dropout, zeroing IDs collapses recall (-69%); with it the loss is ~8% and the selection-window sweep never prefers it.
- Add an inference-only serving rule `score + w*log(1+sales_7d) + b*[degree=0]` with optional `active_days` filter. Parameters are tuned on the validation window because the selection window contains no active cold items. Consequence accepted: tower validation numbers are not independent for these three parameters; test remains clean.
- Item-level sales statistics use all customers' transactions strictly before each window's cutoff (`item_daily_counts.parquet`), with the 50k-customer sample as fallback. Rationale: item-level, non-personalised, available in production; the 50k sample sees ~4% of transactions.
- Refit tower (train+selection+valid, fixed epochs) and ranker (selection+valid, 1.1x rounds) for test only.
- Ranker: LightGBM LambdaRank with 24 time-aware features; the 7-feature v1-style ranker is kept as a paired ablation.
- True cold items (never sold before cutoff) are served through a separate CLIP channel and fixed-position slot interleaving (every 10 or 5 positions), reported with its cost in overall accuracy. Reason: the ranker cannot learn to promote items with all-zero interaction features.
- Evaluation: full-ranking on test is the headline; sampled 1+99 (uniform and popularity negatives) is secondary and always shown next to random/popularity/recent-popularity; improvements are claimed only when the paired bootstrap 95% CI excludes zero.
- `scripts/build_hm_notebooks.py` is deprecated because the v2 notebooks are maintained by hand and the generator would overwrite them.

## 2026-10-06 — Split the repository by dataset (`hm/` vs `legacy/`)

- Project owner decision: gather the whole H&M system into one top-level directory with clear sub-divisions (option 2: split `src/datn` and `apps/*` too, not just notebooks/scripts/docs).
- Python package stays `datn` for H&M and shared code (no import churn in the services); Amazon/Coveo-only modules become package `datn_legacy`. `datn.retrieval.indexer/reviews` and `datn.evaluation` stay with H&M because `hm_indexer` and the RAG tooling import them.
- The recommender service keeps its `DATN_ENGINE=amazon` fallback by importing `datn_legacy`; the H&M engine never imports it.
- Large/ignored artifacts (`data/`, `*.pt`, `node_modules`, `.env`, `deploy/azure/config.env`) were not committed: `.gitignore` rules were re-pointed to the new paths. Historical logs keep old paths. The Azure VM was not touched: after the next `git pull` there, rebuild images (Dockerfile COPY paths changed) and move any VM-local files that lived under the old `checkpoints/`, `deploy/` paths.
