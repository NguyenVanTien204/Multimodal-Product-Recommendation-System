from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    items_path: Path = Path("data/items.parquet")
    train_path: Path = Path("data/train.parquet")
    valid_path: Path = Path("data/valid.parquet")
    test_path: Path = Path("data/test.parquet")
    artifacts_dir: Path = Path("data/artifacts/user_tower")


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.2
    max_seq_len: int = 20


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_negatives: int = 1
    objective: str = "bce"
    negative_sampling_power: float = 0.75
    negative_uniform_ratio: float = 0.0
    logq_correction: bool = False
    selection_metric: str | None = None
    patience: int = 20
    eval_every: int = 5
    grad_clip_norm: float = 5.0
    seed: int = 20260813
    device: str = "auto"


@dataclass(frozen=True)
class EvalConfig:
    ks: tuple[int, ...] = (10,)
    batch_size: int = 512
    exclude_seen: bool = True


@dataclass(frozen=True)
class ContentSourceConfig:
    name: str
    embeddings_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class ContentConfig:
    # Which of `sources` to actually load. Empty -> pure ID-based "Collaborative"
    # baseline (docs/02-overview.md's "Collaborative" row); ["image"] or
    # ["image", "text"] -> content-aware item vectors (the "CF+Image"/"Full
    # Multimodal" rows), see docs/user_tower_design.md.
    enabled: tuple[str, ...] = ()
    sources: tuple[ContentSourceConfig, ...] = ()


@dataclass(frozen=True)
class UserTowerConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    content: ContentConfig = field(default_factory=ContentConfig)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "UserTowerConfig":
        data_raw = {**DataConfig().__dict__, **raw.get("data", {})}
        data = DataConfig(**{k: Path(v) for k, v in data_raw.items()})
        model = ModelConfig(**{**ModelConfig().__dict__, **raw.get("model", {})})
        train_raw = dict(raw.get("train", {}))
        train = TrainConfig(**{**TrainConfig().__dict__, **train_raw})
        eval_raw = dict(raw.get("eval", {}))
        if "ks" in eval_raw:
            eval_raw["ks"] = tuple(eval_raw["ks"])
        eval_cfg = EvalConfig(**{**EvalConfig().__dict__, **eval_raw})

        content_raw = dict(raw.get("content", {}))
        sources = tuple(
            ContentSourceConfig(
                name=s["name"],
                embeddings_path=Path(s["embeddings_path"]),
                metadata_path=Path(s["metadata_path"]),
            )
            for s in content_raw.get("sources", [])
        )
        content = ContentConfig(enabled=tuple(content_raw.get("enabled", ())), sources=sources)

        return UserTowerConfig(data=data, model=model, train=train, eval=eval_cfg, content=content)
