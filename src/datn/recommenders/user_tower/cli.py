from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import UserTowerConfig
from .content import ContentSource, load_content_matrix
from .dataset import build_eval_examples, build_item_vocab, load_user_sequences
from .evaluate import full_ranking_evaluate, sampled_ranking_evaluate
from .model import UserTower
from .train import resolve_device, train


def _load_config(path: Path) -> UserTowerConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return UserTowerConfig.from_dict(raw)


def _apply_cli_overrides(config: UserTowerConfig, args: argparse.Namespace) -> UserTowerConfig:
    data_cfg = config.data
    content_cfg = config.content

    if getattr(args, "artifacts_dir", None) is not None:
        data_cfg = dataclasses.replace(data_cfg, artifacts_dir=args.artifacts_dir)

    if getattr(args, "content", None) is not None:
        content_cfg = dataclasses.replace(content_cfg, enabled=tuple(args.content))

    return dataclasses.replace(config, data=data_cfg, content=content_cfg)


def _run_train(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    config = _apply_cli_overrides(config, args)
    metrics = train(config)
    print(f"Best valid metrics: {metrics}")


def _run_evaluate(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    config = _apply_cli_overrides(config, args)
    device = resolve_device(config.train.device)
    vocab = build_item_vocab(config.data.items_path)
    sequences = load_user_sequences(
        config.data.train_path, config.data.valid_path, config.data.test_path, vocab
    )
    examples = build_eval_examples(sequences, config.model.max_seq_len, args.split)

    sources = [
        ContentSource(s.name, s.embeddings_path, s.metadata_path)
        for s in config.content.sources
        if s.name in config.content.enabled
    ]
    content_matrix = load_content_matrix(vocab, sources)

    checkpoint = args.checkpoint or Path(config.data.artifacts_dir) / "user_tower.pt"
    model = UserTower(
        vocab_size=vocab.vocab_size,
        max_seq_len=config.model.max_seq_len,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        content_matrix=content_matrix if content_matrix.shape[1] > 0 else None,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    results: dict[str, dict[str, float]] = {}

    if args.mode in ("full", "both"):
        logging.info("Running Full Ranking evaluation across %d catalog items...", vocab.num_items)
        results["full_ranking"] = full_ranking_evaluate(
            model, examples, vocab, config.eval.ks, config.eval.batch_size, config.eval.exclude_seen, device
        )

    if args.mode in ("sampled", "both"):
        logging.info(
            "Running Sampled Ranking evaluation (1 positive + %d %s negatives)...",
            args.num_negatives,
            args.strategy,
        )
        popularity_weights = None
        if args.strategy == "popularity":
            freq = np.ones(vocab.num_items + 1, dtype=np.float64)
            freq[0] = 0.0
            for seq in sequences.train.values():
                for item in seq:
                    freq[item] += 1.0
            popularity_weights = freq

        results["sampled_ranking"] = sampled_ranking_evaluate(
            model=model,
            examples=examples,
            vocab=vocab,
            ks=config.eval.ks,
            num_negatives=args.num_negatives,
            batch_size=config.eval.batch_size,
            device=device,
            strategy=args.strategy,
            popularity_weights=popularity_weights,
            seed=args.seed,
        )

    if args.mode == "both":
        payload = results
    elif args.mode == "full":
        payload = results["full_ranking"]
    else:
        payload = results["sampled_ranking"]
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logging.info("Saved evaluation metrics to %s", args.output)
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="SASRec-style Transformer User Tower")
    parser.add_argument("--config", type=Path, default=Path("configs/user_tower.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the User Tower")
    train_parser.add_argument(
        "--content",
        nargs="*",
        default=None,
        help="Override content modalities (e.g. --content image text or --content for CF only)",
    )
    train_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Override directory to store checkpoint and metrics",
    )
    train_parser.set_defaults(func=_run_train)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained checkpoint")
    eval_parser.add_argument("--split", choices=["valid", "test"], default="test")
    eval_parser.add_argument("--checkpoint", type=Path, default=None)
    eval_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the exact evaluation result",
    )
    eval_parser.add_argument(
        "--content",
        nargs="*",
        default=None,
        help="Override content modalities (e.g. --content image text or --content for CF only)",
    )
    eval_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Override directory to look for default checkpoint",
    )
    eval_parser.add_argument(
        "--mode",
        choices=["full", "sampled", "both"],
        default="full",
        help="Evaluation protocol: 'full' (all 152k items), 'sampled' (1 pos + N negs), or 'both'",
    )
    eval_parser.add_argument(
        "--num-negatives",
        type=int,
        default=99,
        help="Number of negative items per user for sampled ranking (default: 99)",
    )
    eval_parser.add_argument(
        "--strategy",
        choices=["random", "popularity"],
        default="random",
        help="Negative sampling strategy for sampled ranking (default: random)",
    )
    eval_parser.add_argument(
        "--seed",
        type=int,
        default=20260813,
        help="Random seed for reproducible negative sampling",
    )
    eval_parser.set_defaults(func=_run_evaluate)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.func(args)



if __name__ == "__main__":
    main()
