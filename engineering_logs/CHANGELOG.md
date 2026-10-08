# Changelog

## 2026-10-06

- Restructured the repository by dataset: all H&M-system code moved under `hm/` (`apps/`, `src/datn/`, `notebooks/`, `scripts/`, `configs/`, `tests/`, `docs/`, `reports/`, `checkpoints/`, `deploy/azure/`); Amazon/Coveo code moved under `legacy/` (package renamed `datn_legacy`: `data`, `features`, `experiments`, `recommenders/{user_tower,reranker,coveo}`). Old-to-new path table: `hm/README.md`. Moves done with `git mv` (history preserved).
- `pyproject.toml` now builds two packages (`hm/src/datn`, `legacy/src/datn_legacy`), pytest runs `hm/tests` + `legacy/tests`; Dockerfiles copy `hm/src` (+ `legacy/src` for the recommender's Amazon engine); `docker-compose.yml`, Azure overlay, `.gitignore`, `.dockerignore` updated. `data/`, `.env`, `docker-compose.yml` stay at the repo root so the Azure VM layout is unchanged except for build paths.
- Entries below this one keep their original (pre-move) paths on purpose; use the table in `hm/README.md`.

## 2026-10-05

- Rewrote `notebooks/hm/02_hm_retrieval.ipynb` and `03_hm_reranking_evaluation.ipynb` (v2) and extended `00_hm_data_preparation.ipynb`: ID-dropout and optional inference cold-fix, refit on train+selection+valid, time-aware serving rule, 7-source candidates, 24 features, LightGBM LambdaRank, CLIP cold channel with slot interleaving, sampled 1+99 evaluation, paired bootstrap, model export cell, global item-count table (`item_daily_counts.parquet`).
- Added `scripts/hm/` (`dataset_stats.py`, `probe_cold_clip.py`, `probe_serving_rule.py`); deprecated `scripts/build_hm_notebooks.py` (blocked without `--force`); rewrote `configs/hm.yaml` as a parameter reference.
- Added `docs/hm/` (README, 01–05, `results/`); replaced `docs/HM_NOTEBOOK_GUIDE.md`; removed `docs/hm_retrieval_diagnosis_and_business_metrics.md` (stale, unmeasured expectations); added dataset-scope notes to Amazon docs; updated `docs/README.md`, `docs/logs/README.md`, root `README.md`, `ROADMAP.md`, `AGENT.md` (H&M appendix); added `checkpoints/hm/*.pt|*.zip` to `.gitignore`.

## 2026-09-22

- Added an independent H&M personalized-fashion Kaggle retrieval and reranking notebook pipeline.
