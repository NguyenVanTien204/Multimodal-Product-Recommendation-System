# Build context is the repo root, so the image installs the same `datn` package
# (retrieval/rag/agent) that the CLI and notebooks use, not a copy.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/hf_cache TOKENIZERS_PARALLELISM=false

# CPU-only torch/torchvision first: the default PyPI Linux wheels bundle CUDA runtimes (GBs).
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
RUN mkdir -p hm/src/datn legacy/src/datn_legacy && touch hm/src/datn/__init__.py legacy/src/datn_legacy/__init__.py && pip install --no-cache-dir ".[rag]"
COPY src ./src
RUN pip install --no-cache-dir --no-deps --force-reinstall .

COPY hm/apps/rag/requirements.txt ./rag-requirements.txt
RUN pip install --no-cache-dir -r rag-requirements.txt

COPY hm/apps/rag/app ./app

EXPOSE 8200
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8200"]
