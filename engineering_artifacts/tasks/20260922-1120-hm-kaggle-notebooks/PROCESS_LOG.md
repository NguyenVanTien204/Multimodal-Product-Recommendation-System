# Process log

- Inspected `AGENT.md`, existing Coveo notebook generator, and current retrieval/reranker conventions.
- Confirmed the existing working tree contains user changes; this task adds new files only.
- Chosen implementation: independent, Kaggle-portable notebook sequence with deterministic activity-stratified sampling.
- Revised at user request: split embedding extraction from retrieval and use a dedicated Jina CLIP v2 download/cache notebook rather than requiring a manually attached model.
- Fixed Kaggle embedding model loading after a user multi-GPU optimization: Jina remote configuration requires `torch_dtype` as the string `"float16"`, not `torch.float16`.
- Fixed the Pillow 12 `_Ink` import incompatibility by pinning/reinstalling Pillow 11.3.0 and compatible timm in the embedding install cell; Kaggle session restart is required after this dependency replacement.
- Fixed the wrapped `AutoImageProcessor` import error by pinning the Jina-compatible Transformers 4.x stack (`transformers==4.51.3`, below Jina's documented 4.52 upper bound) and related package versions.
- Made the text embedding cell restart-safe: it resolves `items.parquet` anew, prints the exact chosen artifact, and rebuilds it from the mounted `articles.csv` if it is genuinely absent.
- Corrected Jina CLIP text embedding task from unsupported `retrieval.passage` to `None`; product metadata is corpus content and must stay aligned with image vectors.
- Removed FP16/autocast from the user-optimized multi-GPU embedding notebook after Jina's image path mixed BF16 destinations with Float32 source tensors. Both T4 replicas now run fully in FP32 with smaller batches.
- Hardened the final image cell for unattended execution: explicit input-dataset resolution, FP32 and one-image preflight checks, per-batch memory-mapped checkpoints, resume support, and per-image corrupt-file isolation.
- Fixed cell 2's Transformers import failure caused by a mixed NumPy/SciPy binary stack. The install cell now pins NumPy 2.0.2, SciPy 1.14.1, and scikit-learn 1.5.2 together; the model cell verifies exact versions before importing Transformers.
