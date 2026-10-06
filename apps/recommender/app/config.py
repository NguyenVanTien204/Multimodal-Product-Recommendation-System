from datetime import date
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DATN_", extra="ignore")

    data_dir: Path = Path("/data")
    user_tower_artifact: str = "artifacts/user_tower_balanced_v1"
    reranker_artifact: str = "artifacts/reranker_v2"
    dataset_dir: str = "processed/balanced_u5_i2_v1"
    image_embeddings: str = "embedding/image_embeddings.npy"
    image_metadata: str = "embedding/image_embedding_metadata.parquet"
    text_embeddings: str = "embedding/text_embeddings.npy"
    text_metadata: str = "embedding/text_embedding_metadata.parquet"

    device: str = "auto"
    default_k: int = 10
    max_k: int = 50
    max_history: int = 50
    model_version: str = "user_tower_balanced_v1+reranker_v2"

    # "amazon": legacy Transformer tower + residual reranker (kept as reference/rollback).
    # "hm": H&M tower + serving rule + 7 candidate sources + LightGBM (src/datn/recommenders/hm).
    engine: str = "amazon"
    hm_dir: str = "hm/serving"                 # tower.pt, reranker_lgbm.txt, items.parquet, *_embeddings.npy, item_daily_counts.parquet
    hm_as_of: date = date(2020, 9, 16)         # the shop's clock (= test cutoff of the thesis)
    hm_cold_every: int = 0                     # 0 = best overall accuracy; 10/5 reserve slots for never-sold items
    hm_model_version: str = "hm_v1_tower_refit+lgbm_24f"
    hm_shop_ids: str = "hm/serving/shop_item_ids.txt"   # optional: article_ids that exist in the shop (one per line)

    @property
    def user_tower_dir(self) -> Path:
        return self.data_dir / self.user_tower_artifact

    @property
    def reranker_dir(self) -> Path:
        return self.data_dir / self.reranker_artifact

    @property
    def items_path(self) -> Path:
        return self.data_dir / self.dataset_dir / "items.parquet"

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.dataset_dir / "train.parquet"

    @property
    def valid_path(self) -> Path:
        return self.data_dir / self.dataset_dir / "valid.parquet"

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.dataset_dir / "test.parquet"

    @property
    def image_embeddings_path(self) -> Path:
        return self.data_dir / self.image_embeddings

    @property
    def image_metadata_path(self) -> Path:
        return self.data_dir / self.image_metadata

    @property
    def text_embeddings_path(self) -> Path:
        return self.data_dir / self.text_embeddings

    @property
    def text_metadata_path(self) -> Path:
        return self.data_dir / self.text_metadata

    @property
    def hm_path(self) -> Path:
        return self.data_dir / self.hm_dir

    @property
    def hm_shop_ids_path(self) -> Path:
        return self.data_dir / self.hm_shop_ids


settings = Settings()
