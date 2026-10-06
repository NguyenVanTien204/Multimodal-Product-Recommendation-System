# Build context is the repo root (see apps/backend/docker-compose.yml), so this
# image can install the `datn` package from src/ with its `recommenders` extra
# (torch, numpy) -- the same code the training notebooks/CLI use, not a copy.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# CPU-only torch first: the default PyPI linux wheel bundles CUDA runtimes and
# is several GB larger than this service needs.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
RUN mkdir -p src/datn && touch src/datn/__init__.py && pip install --no-cache-dir ".[recommenders,hm]"
COPY src ./src
RUN pip install --no-cache-dir --no-deps --force-reinstall .

COPY apps/recommender/requirements.txt ./recommender-requirements.txt
RUN pip install --no-cache-dir -r recommender-requirements.txt

COPY apps/recommender/app ./app

EXPOSE 8100
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
