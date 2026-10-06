# Evaluation log

## H&M evaluation protocol (v2, executed 2026-10-05)

- Metrics: HitRate / Recall / NDCG / MAP at 12, 50, 100 (full-ranking over 105,542 items), candidate-union recall, catalog coverage; slices warm / strict-cold / never-sold / repeat / explore; secondary sampled 1+99 (uniform and popularity negatives) with cluster CIs; paired bootstrap (2,000 resamples) on HitRate@12 and NDCG@12. Definitions: `docs/hm/01_dataset_and_protocol.md`.
- Split: chronological 7-day windows train -> selection -> validation -> test; tower epoch chosen on selection; ranker early-stopped on validation; serving rule tuned on validation; test reported after refit.
- Baselines always reported: popularity, recent_popularity (7 days before the window), random.
- Result (Kaggle, test, 2,799 customers): popularity 0.0186 / 0.1165, recent_popularity 0.0757 / 0.2529, tower without serving rule 0.0564 / 0.1383, tower + serving rule 0.1208 / 0.3455, tower + LightGBM reranker (24 features) 0.1275 / 0.3651 (HitRate@12 / HitRate@100). Reranker minus recent_popularity: +0.0518 HitRate@12 [+0.0368, +0.0661]; minus 7-feature ranker: +0.0314 [+0.0204, +0.0418]. Never-sold items: reranker 0 HitRate@100; with slot interleaving every 10 / 5 positions 0.0212 / 0.0508 at a cost of -0.0143 / -0.0293 overall HitRate@100. Full tables: `docs/hm/03_experiments_and_results.md`; raw JSON: `docs/hm/results/`.
- Not yet evaluated: modality ablation (id/text/image/multimodal), multiple seeds, paired CI of reranker vs tower+serving rule, new-user cold start, `full` profile (300k customers).
