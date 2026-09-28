from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..user_tower.content import ContentSource, load_content_matrix
from ..user_tower.dataset import ItemVocab, build_item_vocab, load_user_sequences, pad_right
from ..user_tower.model import UserTower

from .candidates import CandidateBudget, CandidateExample, PopularityStats, candidate_features
from .model import FEATURE_NAMES, ResidualListwiseRanker


def _select_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


@dataclass(frozen=True)
class Recommendation:
    item_id: str
    score: float


@dataclass(frozen=True)
class RecommendResult:
    recommendations: list[Recommendation]
    source: str  # "reranker" | "popularity_fallback"


class RerankerPipeline:
    """End-to-end retrieval + reranking: item_id (ASIN/SKU) history in, ranked
    item_ids out. Wraps a frozen User Tower (retrieval) and a ResidualListwiseRanker
    (reranking) so a serving process only needs to call `recommend`; every other
    module in `recommenders/reranker` stays a pure, checkpoint-agnostic building block.
    """

    def __init__(
        self,
        tower: UserTower,
        vocab: ItemVocab,
        popularity: PopularityStats,
        ranker: ResidualListwiseRanker,
        blend: float,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        budget: CandidateBudget,
        max_seq_len: int,
        device: torch.device,
    ) -> None:
        self.tower = tower
        self.vocab = vocab
        self.popularity = popularity
        self.ranker = ranker
        self.blend = blend
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.budget = budget
        self.max_seq_len = max_seq_len
        self.device = device
        self._idx2item = {idx: item_id for item_id, idx in vocab.item2idx.items()}

    @classmethod
    def from_artifacts(
        cls,
        user_tower_dir: Path,
        reranker_dir: Path,
        items_path: Path,
        train_path: Path,
        valid_path: Path,
        test_path: Path,
        content_sources: list[ContentSource],
        device: str = "auto",
    ) -> "RerankerPipeline":
        """Load every artifact a live pipeline needs.

        `content_sources` is passed in explicitly (not parsed from
        `training_config.json`) because that file stores paths as they existed on
        the training machine (often Windows-style `data\\embedding\\...`), which
        do not resolve inside a Linux serving container.
        """
        resolved_device = _select_device(device)

        vocab = build_item_vocab(items_path)

        training_config = json.loads((user_tower_dir / "training_config.json").read_text(encoding="utf-8"))
        model_cfg = training_config["model"]

        content_matrix = load_content_matrix(vocab, content_sources)
        if content_matrix.shape[1] == 0:
            raise ValueError("RerankerPipeline requires at least one content source (image and/or text)")

        tower = UserTower(
            vocab_size=vocab.vocab_size,
            max_seq_len=model_cfg["max_seq_len"],
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            d_ff=model_cfg["d_ff"],
            dropout=model_cfg["dropout"],
            content_matrix=content_matrix,
        ).to(resolved_device)
        state_dict = torch.load(user_tower_dir / "user_tower.pt", map_location=resolved_device)
        tower.load_state_dict(state_dict)
        tower.eval()
        for p in tower.parameters():
            p.requires_grad_(False)

        train_sequences = load_user_sequences(train_path, valid_path, test_path, vocab).train
        popularity = PopularityStats.from_sequences(vocab, train_sequences)

        run_config = json.loads((reranker_dir / "run_config.json").read_text(encoding="utf-8"))
        candidate_budget = run_config["candidate_budget"]
        budget = CandidateBudget(
            tower=candidate_budget["tower"],
            popularity=candidate_budget["popularity"],
            content_centroid=candidate_budget["content_centroid"],
            last_item=candidate_budget["last_item"],
        )

        checkpoint = torch.load(reranker_dir / "reranker.pt", map_location=resolved_device)
        if tuple(checkpoint["feature_names"]) != FEATURE_NAMES:
            raise ValueError("reranker checkpoint feature order does not match FEATURE_NAMES")
        ranker = ResidualListwiseRanker(len(FEATURE_NAMES)).to(resolved_device)
        ranker.load_state_dict(checkpoint["model_state_dict"])
        ranker.eval()
        for p in ranker.parameters():
            p.requires_grad_(False)
        blend = float(checkpoint["blend"])

        scaler = np.load(reranker_dir / "feature_scaler.npz")

        return cls(
            tower=tower,
            vocab=vocab,
            popularity=popularity,
            ranker=ranker,
            blend=blend,
            feature_mean=scaler["mean"],
            feature_std=scaler["std"],
            budget=budget,
            max_seq_len=model_cfg["max_seq_len"],
            device=resolved_device,
        )

    def _popularity_fallback(self, k: int, exclude: set[int]) -> list[Recommendation]:
        out: list[Recommendation] = []
        for idx in self.popularity.order:
            idx = int(idx)
            if idx in exclude:
                continue
            item_id = self._idx2item.get(idx)
            if item_id is None:
                continue
            out.append(Recommendation(item_id=item_id, score=float(self.popularity.score[idx])))
            if len(out) == k:
                break
        return out

    @torch.no_grad()
    def recommend(self, history_item_ids: list[str], k: int = 10) -> RecommendResult:
        """`history_item_ids` are catalog SKUs (ASINs), most-recent last."""
        known = [self.vocab.item2idx[i] for i in history_item_ids if i in self.vocab.item2idx]
        seen = set(known)

        if not known:
            return RecommendResult(recommendations=self._popularity_fallback(k, seen), source="popularity_fallback")

        context = pad_right(known, self.max_seq_len)
        example = CandidateExample(context=context, seen=frozenset(seen))
        rows = candidate_features(self.tower, [example], self.vocab, self.popularity, self.budget, self.device)
        candidate_ids, features = rows[0]
        if len(candidate_ids) == 0:
            return RecommendResult(recommendations=self._popularity_fallback(k, seen), source="popularity_fallback")

        standardized = (features - self.feature_mean) / self.feature_std
        x = torch.from_numpy(standardized.astype(np.float32)).to(self.device)
        scores = self.ranker(x, blend=self.blend).cpu().numpy()

        order = np.argsort(-scores)[:k]
        recommendations = [
            Recommendation(item_id=self._idx2item[int(candidate_ids[i])], score=float(scores[i]))
            for i in order
            if int(candidate_ids[i]) in self._idx2item
        ]
        return RecommendResult(recommendations=recommendations, source="reranker")
