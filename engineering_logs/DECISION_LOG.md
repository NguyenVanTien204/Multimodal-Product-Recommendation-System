# Decision log

## 2026-09-22 — H&M notebook architecture

- Retain all pre-existing Amazon/Coveo code and notebooks; H&M is an additive benchmark.
- Use a deterministic activity-stratified customer sample based only on history before holdout windows.
- Use two seven-day development windows before validation/test so retrieval and reranker checkpoint selection do not inspect test labels.
- Use persistent Jina CLIP v2 512-dimensional embeddings, downloaded and cached by the dedicated Kaggle embedding notebook; never silently approximate vision features with random weights.
