# H&M Kaggle pipeline: retrieval → reranking

The four notebooks under `notebooks/hm/` are independent from the Amazon and
Coveo pipelines. They write only to `/kaggle/working/hm_v1` by default.

## Kaggle setup

Create a Kaggle Notebook with GPU enabled, then attach the competition dataset
**H&M Personalized Fashion Recommendations** as an input. Its directory must
contain `transactions_train.csv`, `articles.csv`, `customers.csv`, and (for
image embeddings) `images/`. Execute notebooks in order:

1. `00_hm_data_preparation.ipynb` discovers the mounted input, applies the
   deterministic sample, then creates Parquet and the chronological split manifest.
2. `01_hm_jina_embeddings.ipynb` downloads Jina CLIP v2, then persistently saves
   aligned text/image embeddings and their manifest. This is the heaviest GPU job.
3. `02_hm_retrieval.ipynb` reads only those saved artifacts, trains the retriever,
   and selects its checkpoint on a separate pre-validation week (`rerank_train`).
4. `03_hm_reranking_evaluation.ipynb` creates the candidate union, trains the
   listwise reranker, selects on validation MAP@12, and reports final test metrics.

## Sampling and leakage policy

The source is too large for a typical Kaggle session to repeat every ablation.
The preparation notebook selects customers using **only events before the
pre-validation reranker-training week**. It stratifies by their log purchase-count bucket and uses a
seeded hash within each bucket. Afterwards it brings across their whole timeline,
limits only their oldest training history, and keeps both future 7-day windows
unaltered. Thus test labels are never used to choose users, features, candidates,
or checkpoints.

`quick` uses 50,000 customers; `full` uses 300,000. Set `HM_PROFILE=full` before
running notebook 00. A fully unbounded run is intentionally not the default:
31M transactions and image encoding make it impractical for a reproducible
student Kaggle run.

## Embedding modes

The embedding notebook downloads [`jinaai/jina-clip-v2`](https://huggingface.co/jinaai/jina-clip-v2)
on its first execution, so enable Kaggle Internet for that notebook. It caches the
download in `/kaggle/working/hm_v1/hf_cache` and saves normalized
`text_embeddings.npy`, `image_embeddings.npy`, and `embedding_manifest.json`.
Jina CLIP v2 supports 64–1024 dimensional Matryoshka outputs; this pipeline fixes
512 dimensions for a practical storage/quality trade-off. Do not run retrieval
until the embedding manifest reports the expected image coverage.

Article metadata is embedded with `task=None`; this Jina checkpoint accepts
`retrieval.query` only for query text and does not support `retrieval.passage`.

Jina CLIP runs in FP32 stable mode in this notebook. Its remote image code can
create Float32 tensors internally; forcing FP16/BF16 on T4 causes indexed-write
dtype errors. The two T4s each hold one FP32 model replica; text/image batches
are 64/16 respectively to keep memory bounded.

The image cell performs a one-image preflight, writes every successful batch to
a memory-mapped `.npy`, resumes non-zero rows after interruption, and isolates
corrupt JPEGs into `image_embedding_failures.json`. It never waits until the end
of the full catalog to persist progress.

The notebook pins a coherent binary stack (`numpy==2.0.2`, `scipy==1.14.1`,
`scikit-learn==1.5.2`) together with `transformers==4.51.3`, `einops==0.8.1`,
`timm==1.0.19`, and `Pillow==11.3.0`. Jina's published dependency range is
`transformers<4.52`; Pillow 12 also removed `_Ink`, breaking image imports in
some Kaggle combinations. After the install cell changes packages, restart the
Kaggle session once before loading Jina CLIP.

## Report metrics correctly

Report `MAP@12` and `Recall@12` on the 7-day validation/test windows. The
candidate-union recall in notebook 04 is an upper bound on reranker quality;
report it together with end-to-end MAP@12. Do not compare these values directly
with the current Amazon leave-last-out HR/NDCG experiment.
