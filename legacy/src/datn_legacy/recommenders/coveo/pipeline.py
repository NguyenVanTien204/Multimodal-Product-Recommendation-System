from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .model import ActionAwareTwoTower, ResidualListwiseReranker


ACTIONS = {"detail": 1, "search_click": 2, "add": 3, "purchase": 4, "remove": 5}
FEATURES = ("retrieval_score", "reciprocal_rank", "log_popularity", "category_affinity",
            "price_similarity", "text_last_similarity", "image_last_similarity", "source_popular")


@dataclass(frozen=True)
class RetrievalTrainConfig:
    max_seq_len: int = 30
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 256
    epochs: int = 20
    lr: float = 8e-4
    weight_decay: float = 1e-4
    purchase_alpha: float = 2.0
    temperature: float = 0.07
    patience: int = 4
    seed: int = 20260922
    device: str = "auto"


@dataclass(frozen=True)
class RerankerTrainConfig:
    candidate_k: int = 1000
    retrieval_k: int = 900
    popularity_k: int = 100
    max_sessions_per_split: int | None = 50_000
    batch_size: int = 16
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 20260922
    device: str = "auto"


def _device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)


def _seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


def load_vocab(items_path: Path) -> tuple[list[str], dict[str, int]]:
    ids = pl.read_parquet(items_path, columns=["item_id"])["item_id"].to_list()
    return ["__PAD__", *ids], {item: i + 1 for i, item in enumerate(ids)}


def load_sessions(path: Path, vocab: dict[str, int]) -> list[dict[str, object]]:
    frame = (pl.read_parquet(path, columns=["user_id", "item_id", "action", "timestamp"])
             .with_columns(pl.col("item_id").replace_strict(vocab, default=0).cast(pl.Int64).alias("item_idx"))
             .filter(pl.col("item_idx") > 0).sort(["user_id", "timestamp"]))
    return frame.group_by("user_id", maintain_order=True).agg(
        pl.col("item_idx"), pl.col("action"), pl.col("timestamp")
    ).filter(pl.col("item_idx").list.len() >= 2).to_dicts()


def _pad(values: list[int], length: int) -> np.ndarray:
    result = np.zeros(length, dtype=np.int64); values = values[-length:]
    result[:len(values)] = values
    return result


class TransitionDataset(Dataset):
    def __init__(self, sessions: list[dict[str, object]], max_len: int) -> None:
        self.rows = sessions; self.max_len = max_len
        self.examples = [(session_index, target_position)
                         for session_index, row in enumerate(sessions)
                         for target_position in range(1, len(row["item_idx"]))]
    def __len__(self) -> int: return len(self.examples)
    def __getitem__(self, index: int):
        session_index, target_position = self.examples[index]
        row = self.rows[session_index]; items = row["item_idx"]; actions = [ACTIONS.get(x, 1) for x in row["action"]]
        target_action = row["action"][target_position]
        return (torch.from_numpy(_pad(items[:target_position], self.max_len)),
                torch.from_numpy(_pad(actions[:target_position], self.max_len)),
                torch.tensor(items[target_position]), torch.tensor(1.0 if target_action == "purchase" else 0.0),
                torch.tensor(float({"purchase": 4, "add": 2}.get(target_action, 1))))


class LastEventDataset(Dataset):
    def __init__(self, sessions: list[dict[str, object]], max_len: int) -> None:
        self.rows = sessions; self.max_len = max_len
    def __len__(self) -> int: return len(self.rows)
    def __getitem__(self, index: int):
        row = self.rows[index]; items = row["item_idx"]; actions = [ACTIONS.get(x, 1) for x in row["action"]]
        return (torch.from_numpy(_pad(items[:-1], self.max_len)),
                torch.from_numpy(_pad(actions[:-1], self.max_len)), torch.tensor(items[-1]))


@torch.no_grad()
def evaluate_retrieval(model, sessions, max_len, device, ks=(10, 100, 1000), batch_size=256):
    model.eval(); table = model.item_table(); ranks = []
    loader = DataLoader(LastEventDataset(sessions, max_len), batch_size=batch_size)
    for items, actions, targets in loader:
        items, actions, targets = items.to(device), actions.to(device), targets.to(device)
        query = model.encode(items, actions, table); scores = query @ table.T
        scores[:, 0] = -torch.inf
        scores.scatter_(1, items, -torch.inf)
        target_scores = scores.gather(1, targets[:, None])
        ranks.extend((scores.gt(target_scores).sum(1) + 1).cpu().tolist())
    ranks_np = np.asarray(ranks)
    result = {f"HitRate@{k}": float((ranks_np <= k).mean()) for k in ks}
    result.update({f"NDCG@{k}": float(np.where(ranks_np <= k, 1 / np.log2(ranks_np + 1), 0).mean()) for k in ks})
    result["MRR"] = float((1 / ranks_np).mean()) if len(ranks_np) else 0.0
    result["n_sessions"] = len(ranks)
    return result


def train_retrieval(processed_dir: Path, embeddings_dir: Path, output_dir: Path, config: RetrievalTrainConfig):
    _seed(config.seed); device = _device(config.device); output_dir.mkdir(parents=True, exist_ok=True)
    item_ids, vocab = load_vocab(processed_dir / "items.parquet")
    train = load_sessions(processed_dir / "train.parquet", vocab)
    valid = load_sessions(processed_dir / "valid.parquet", vocab)
    test = load_sessions(processed_dir / "test.parquet", vocab)
    text = np.load(embeddings_dir / "text_embeddings.npy"); image = np.load(embeddings_dir / "image_embeddings.npy")
    model = ActionAwareTwoTower(len(item_ids), config.max_seq_len, d_model=config.d_model,
        n_heads=config.n_heads, n_layers=config.n_layers, dropout=config.dropout,
        text_matrix=text, image_matrix=image).to(device)
    loader = DataLoader(TransitionDataset(train, config.max_seq_len), batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history = []; best = -1.0; stale = 0
    for epoch in range(1, config.epochs + 1):
        model.train(); losses = []
        for items, actions, targets, purchase, weights in loader:
            items, actions, targets = items.to(device), actions.to(device), targets.to(device)
            purchase, weights = purchase.to(device), weights.to(device)
            table = model.item_table(); query = model.encode(items, actions, table)
            target_vectors = table[targets]
            logits = query @ target_vectors.T / config.temperature
            # Sessions in one batch often share a popular target. Treat every equal
            # target as positive instead of turning duplicates into false negatives.
            positive_mask = targets[:, None].eq(targets[None, :])
            positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
            per_row = torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
            ranking_loss = (per_row * weights).mean()
            purchase_loss = torch.nn.functional.binary_cross_entropy_with_logits(model.purchase_head(query).squeeze(-1), purchase)
            loss = ranking_loss + config.purchase_alpha * purchase_loss
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(loss.item())
        metrics = evaluate_retrieval(model, valid, config.max_seq_len, device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **metrics})
        score = metrics["HitRate@1000"]
        if score > best:
            best, stale = score, 0
            torch.save({"model_state": model.state_dict(), "config": asdict(config), "item_ids": item_ids,
                        "valid_metrics": metrics, "epoch": epoch}, output_dir / "best_retrieval.pt")
        else:
            stale += 1
            if stale >= config.patience: break
    checkpoint = torch.load(output_dir / "best_retrieval.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate_retrieval(model, test, config.max_seq_len, device)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    result = {"selection_split": "valid", "selection_metric": "HitRate@1000", "best_valid": checkpoint["valid_metrics"], "test": test_metrics}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_retrieval(checkpoint_path: Path, embeddings_dir: Path, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False); cfg = RetrievalTrainConfig(**state["config"])
    text = np.load(embeddings_dir / "text_embeddings.npy"); image = np.load(embeddings_dir / "image_embeddings.npy")
    model = ActionAwareTwoTower(len(state["item_ids"]), cfg.max_seq_len, d_model=cfg.d_model,
        n_heads=cfg.n_heads, n_layers=cfg.n_layers, dropout=cfg.dropout, text_matrix=text, image_matrix=image).to(device)
    model.load_state_dict(state["model_state"]); model.eval()
    return model, state["item_ids"], cfg


def _catalog_features(processed_dir: Path, item_ids: list[str], train_path: Path):
    items = pl.read_parquet(processed_dir / "items.parquet").with_row_index("item_idx", offset=1)
    popularity = (pl.read_parquet(train_path).group_by("item_id").len(name="popularity"))
    return items.join(popularity, on="item_id", how="left").with_columns(pl.col("popularity").fill_null(0)).sort("item_idx")


@torch.no_grad()
def generate_candidates(split: str, processed_dir: Path, embeddings_dir: Path, checkpoint_path: Path,
                        destination: Path, config: RerankerTrainConfig):
    device = _device(config.device); model, item_ids, retrieval_cfg = load_retrieval(checkpoint_path, embeddings_dir, device)
    vocab = {x: i for i, x in enumerate(item_ids) if i}; sessions = load_sessions(processed_dir / f"{split}.parquet", vocab)
    if config.max_sessions_per_split is not None:
        sessions = sessions[:config.max_sessions_per_split]
    catalog = _catalog_features(processed_dir, item_ids, processed_dir / "train.parquet")
    category = [None, *catalog["category_path"].to_list()]; price = np.r_[np.nan, catalog["price_bucket"].to_numpy()]
    popularity = np.r_[0, catalog["popularity"].to_numpy()]; popular_ids = np.argsort(-popularity)[1:config.popularity_k + 1]
    text = np.load(embeddings_dir / "text_embeddings.npy"); image = np.load(embeddings_dir / "image_embeddings.npy")
    table = model.item_table(); rows = []
    for row in sessions:
        seq = row["item_idx"]; acts = [ACTIONS.get(x, 1) for x in row["action"]]; target = seq[-1]
        context = seq[:-1]; context_actions = acts[:-1]
        it = torch.from_numpy(_pad(context, retrieval_cfg.max_seq_len))[None].to(device)
        ac = torch.from_numpy(_pad(context_actions, retrieval_cfg.max_seq_len))[None].to(device)
        scores = (model.encode(it, ac, table) @ table.T).squeeze(0); scores[0] = -torch.inf
        scores[torch.tensor(list(set(context)), device=device)] = -torch.inf
        k = min(config.retrieval_k, len(item_ids) - 1)
        values, indices = torch.topk(scores, k); candidates = indices.cpu().tolist(); score_map = dict(zip(candidates, values.cpu().tolist()))
        for idx in popular_ids:
            if int(idx) not in score_map and int(idx) not in context: candidates.append(int(idx))
        candidates = candidates[:config.candidate_k]
        cat_counts = {}; price_values = []
        for idx in context:
            root = category[idx].split("/")[0] if category[idx] else None
            if root: cat_counts[root] = cat_counts.get(root, 0) + 1
            if not np.isnan(price[idx]): price_values.append(price[idx])
        avg_price = float(np.mean(price_values)) if price_values else np.nan; last = context[-1]
        for rank, idx in enumerate(candidates, 1):
            root = category[idx].split("/")[0] if category[idx] else None
            retr = score_map.get(idx, -1.0)
            rows.append({"session_id": row["user_id"], "item_idx": idx, "item_id": item_ids[idx],
                "label": int(idx == target), "retrieval_score": retr, "reciprocal_rank": 1.0 / rank,
                "log_popularity": math.log1p(popularity[idx]), "category_affinity": cat_counts.get(root, 0) / max(1, len(context)),
                "price_similarity": 0.0 if np.isnan(avg_price) or np.isnan(price[idx]) else 1.0 / (1.0 + abs(price[idx] - avg_price)),
                "text_last_similarity": float(text[last] @ text[idx]), "image_last_similarity": float(image[last] @ image[idx]),
                "source_popular": float(idx not in score_map)})
    frame = pl.DataFrame(rows); destination.parent.mkdir(parents=True, exist_ok=True); frame.write_parquet(destination)
    recall = float(frame.group_by("session_id").agg(pl.col("label").max())["label"].mean()) if frame.height else 0.0
    return {"split": split, "candidate_recall": recall, "sessions": len(sessions), "rows": frame.height}


class CandidateDataset(Dataset):
    def __init__(self, path: Path):
        groups = pl.read_parquet(path).sort(["session_id", "reciprocal_rank"], descending=[False, True]).partition_by("session_id", maintain_order=True)
        self.groups = [g for g in groups if g["label"].sum() > 0]
    def __len__(self): return len(self.groups)
    def __getitem__(self, i):
        g = self.groups[i]; return (torch.tensor(g.select(FEATURES).to_numpy(), dtype=torch.float32),
            torch.tensor(g["retrieval_score"].to_numpy(), dtype=torch.float32), torch.tensor(g["label"].arg_max()))


def _collate(batch): return batch


@torch.no_grad()
def evaluate_reranker(model, path: Path, device, ks=(10, 50, 100)):
    frame = pl.read_parquet(path); candidate_recall = float(frame.group_by("session_id").agg(pl.col("label").max())["label"].mean())
    dataset = CandidateDataset(path); ranks = []
    model.eval()
    for features, retrieval, target in dataset:
        scores = model(features.to(device), retrieval.to(device)); order = torch.argsort(scores, descending=True)
        ranks.append(int(torch.where(order == target.to(device))[0].item()) + 1)
    ranks = np.asarray(ranks); result = {"candidate_recall": candidate_recall, "conditional_sessions": len(ranks)}
    for k in ks:
        conditional = float((ranks <= k).mean()) if len(ranks) else 0.0
        result[f"ConditionalHR@{k}"] = conditional; result[f"HitRate@{k}"] = candidate_recall * conditional
    return result


def train_reranker(train_candidates: Path, valid_candidates: Path, test_candidates: Path,
                   output_dir: Path, config: RerankerTrainConfig):
    _seed(config.seed); device = _device(config.device); output_dir.mkdir(parents=True, exist_ok=True)
    model = ResidualListwiseReranker(len(FEATURES)).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loader = DataLoader(CandidateDataset(train_candidates), batch_size=config.batch_size, shuffle=True, collate_fn=_collate)
    best = -1.0; stale = 0; history = []
    for epoch in range(1, config.epochs + 1):
        model.train(); losses = []
        for groups in loader:
            loss = torch.stack([torch.nn.functional.cross_entropy(model(f.to(device), r.to(device))[None], y.to(device)[None]) for f, r, y in groups]).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
        metrics = evaluate_reranker(model, valid_candidates, device); history.append({"epoch": epoch, "loss": float(np.mean(losses)), **metrics})
        score = metrics["HitRate@10"]
        if score > best:
            best, stale = score, 0; torch.save({"model_state": model.state_dict(), "features": FEATURES,
                "config": asdict(config), "valid_metrics": metrics, "epoch": epoch}, output_dir / "best_reranker.pt")
        else:
            stale += 1
            if stale >= config.patience: break
    state = torch.load(output_dir / "best_reranker.pt", map_location=device, weights_only=False); model.load_state_dict(state["model_state"])
    result = {"selection_split": "valid", "best_valid": state["valid_metrics"], "test": evaluate_reranker(model, test_candidates, device)}
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
