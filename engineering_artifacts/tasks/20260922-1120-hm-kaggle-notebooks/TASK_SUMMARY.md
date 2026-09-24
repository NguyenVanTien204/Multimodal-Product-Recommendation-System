# Task summary

## Delivered

- Added four generated, standalone notebooks under `notebooks/hm/`: data preparation; persistent Jina CLIP embedding extraction; retrieval; candidate union plus reranking/evaluation.
- Added `configs/hm.yaml`, `docs/HM_NOTEBOOK_GUIDE.md`, and the reproducible generator `scripts/build_hm_notebooks.py`.
- Existing notebooks and source modules were not changed.

## Verification

- `python -m py_compile scripts/build_hm_notebooks.py` passed.
- Generated all four merged notebooks and compiled every Python code cell successfully with `compile(...)`.
- Verified the installed Polars accepts the grouped chronological rank expression used by the preparation notebook.

## Important operating detail

The local environment does not contain the H&M competition data, so no end-to-end training was executed here. The notebook discovers a Kaggle-mounted input; Jina CLIP v2 downloads from Hugging Face on its first run and therefore needs Kaggle Internet enabled.
