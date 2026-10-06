# Changelog

## 2026-10-05

- Rewrote `notebooks/hm/02_hm_retrieval.ipynb` and `03_hm_reranking_evaluation.ipynb` (v2) and extended `00_hm_data_preparation.ipynb`: ID-dropout and optional inference cold-fix, refit on train+selection+valid, time-aware serving rule, 7-source candidates, 24 features, LightGBM LambdaRank, CLIP cold channel with slot interleaving, sampled 1+99 evaluation, paired bootstrap, model export cell, global item-count table (`item_daily_counts.parquet`).
- Added `scripts/hm/` (`dataset_stats.py`, `probe_cold_clip.py`, `probe_serving_rule.py`); deprecated `scripts/build_hm_notebooks.py` (blocked without `--force`); rewrote `configs/hm.yaml` as a parameter reference.
- Added `docs/hm/` (README, 01–05, `results/`); replaced `docs/HM_NOTEBOOK_GUIDE.md`; removed `docs/hm_retrieval_diagnosis_and_business_metrics.md` (stale, unmeasured expectations); added dataset-scope notes to Amazon docs; updated `docs/README.md`, `docs/logs/README.md`, root `README.md`, `ROADMAP.md`, `AGENT.md` (H&M appendix); added `checkpoints/hm/*.pt|*.zip` to `.gitignore`.

## 2026-09-22

- Added an independent H&M personalized-fashion Kaggle retrieval and reranking notebook pipeline.
