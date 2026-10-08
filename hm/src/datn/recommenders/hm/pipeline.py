"""H&M serving pipeline: tower + time-aware serving rule -> 7 candidate sources -> 24 features -> LightGBM.

Ported from hm/notebooks/02_hm_retrieval.ipynb (tower, serving rule) and
03_hm_reranking_evaluation.ipynb (item stats, candidates, `featurize`). The notebook is the
reference: hm/scripts/parity_check.py runs both on the same users and compares candidates and the
24-column feature matrix, so keep the arithmetic here identical to it.

Everything is item-level and computed once at construction for a fixed "shop clock" (`as_of`):
only transactions strictly before `as_of` are used, exactly like the test cutoff in the thesis.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

_EPOCH = dt.date(1970, 1, 1)

SOURCES = ["tower", "recent", "repeat", "new", "content", "popular", "cold"]
FEATURES_V2 = [
    "tower_z", "tower_logrank", "log_pop_all", "log_pop_recent", "recent_share", "content_anchor", "content_mean",
    "from_tower", "from_recent", "from_repeat", "from_new", "from_content", "from_popular",
    "is_cold", "repeat_count", "age_days", "seen_before", "type_aff", "dept_aff", "colour_aff", "same_type_last",
    "log_price_ratio", "from_cold", "launch_prior",
]


@dataclass(frozen=True)
class HMConfig:
    maxlen: int = 30
    tower_k: int = 500
    recent_k: int = 200
    repeat_k: int = 100
    new_k: int = 100
    content_k: int = 200
    popular_k: int = 100
    cold_k: int = 100
    prior_days: int = 56
    prior_topk: int = 10
    new_days: int = 28


@dataclass(frozen=True)
class Recommendation:
    item_id: str
    score: float


@dataclass(frozen=True)
class RecommendResult:
    recommendations: list[Recommendation]
    source: str


def date_to_day(value: dt.date) -> int:
    return (value - _EPOCH).days


def _unique_recent(history: Sequence[int], limit: int) -> list[int]:
    return list(dict.fromkeys(reversed(history)))[:limit]


def interleave_cold(ranked: Sequence[int], cold_list: Sequence[int], every: int) -> list[int]:
    """Keep the reranker order but give positions every, 2*every, ... to the cold channel (in channel order)."""
    cold_set = set(cold_list)
    main = [x for x in ranked if x not in cold_set]
    out: list[int] = []
    mi = ci = 0
    while mi < len(main) or ci < len(cold_list):
        if ((len(out) + 1) % every == 0 and ci < len(cold_list)) or mi >= len(main):
            out.append(cold_list[ci])
            ci += 1
        else:
            out.append(main[mi])
            mi += 1
    return out


class HMRecommender:
    def __init__(
        self,
        *,
        item_ids: list[str],
        table: torch.Tensor,
        degree: np.ndarray,
        serving: dict,
        content: np.ndarray,
        attr_codes: dict[str, np.ndarray],
        counts: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        booster,
        as_of_day: int,
        device: torch.device,
        config: HMConfig = HMConfig(),
    ) -> None:
        self.cfg = config
        self.device = device
        self.item_ids = item_ids
        self.n_items = len(item_ids)
        self.item2idx = {item_id: i + 1 for i, item_id in enumerate(item_ids)}
        self.table = table.to(device)
        self.degree = degree
        self.cold_t = torch.tensor(degree == 0, device=device)
        self.serving = {"active_days": 0, "cold_bonus": 0.0, "pop_weight": 0.0, **(serving or {})}
        self.content_t = torch.from_numpy(content).to(device)
        self.attr = attr_codes
        self.booster = booster
        self.as_of_day = as_of_day
        self.stats = self._item_stats(*counts, cutoff=as_of_day)

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_artifacts(
        cls,
        *,
        tower_path: Path,
        ranker_path: Path,
        items_path: Path,
        text_path: Path,
        image_path: Path,
        daily_counts_path: Path,
        as_of: dt.date = dt.date(2020, 9, 16),
        device: str = "cpu",
        config: HMConfig = HMConfig(),
    ) -> "HMRecommender":
        import lightgbm as lgb

        dev = torch.device(device)
        items = pl.read_parquet(items_path)
        item_ids = items["item_id"].to_list()
        n = len(item_ids)

        text_raw = np.load(text_path).astype("float32")
        image_raw = np.load(image_path).astype("float32")
        if text_raw.shape[0] != n or image_raw.shape[0] != n:
            raise ValueError(f"embeddings rows {text_raw.shape[0]}/{image_raw.shape[0]} != {n} items")

        ckpt = torch.load(tower_path, map_location="cpu", weights_only=False, mmap=True)
        if list(map(str, ckpt["item_ids"])) != item_ids:
            raise ValueError("tower checkpoint item order differs from items.parquet")
        table = cls._item_table(ckpt, text_raw, image_raw)
        degree = np.asarray(ckpt["degree"], dtype="float64")

        content = cls._content_matrix(text_raw, image_raw)   # after the table: it needs the raw (un-renormalised) vectors
        del text_raw, image_raw

        def codes(column: str) -> np.ndarray:
            c = items[column].fill_null("").cast(pl.Categorical).to_physical().to_numpy().astype("int64") + 1
            return np.concatenate([[0], c])

        attr = {"type": codes("product_type_name"), "dept": codes("department_name"), "colour": codes("colour_group_name")}

        vocab = {item_id: i + 1 for i, item_id in enumerate(item_ids)}
        counts = (
            pl.read_parquet(daily_counts_path)
            .with_columns(pl.col("article_id").replace_strict(vocab, default=0).alias("idx"))
            .filter(pl.col("idx") > 0)
        )
        arrays = (
            counts["idx"].to_numpy(),
            counts["t_dat"].cast(pl.Int32).to_numpy(),
            counts["n"].to_numpy().astype("float64"),
            counts["price_sum"].to_numpy().astype("float64"),
        )
        return cls(
            item_ids=item_ids, table=table, degree=degree, serving=ckpt.get("serving") or {}, content=content,
            attr_codes=attr, counts=arrays, booster=lgb.Booster(model_file=str(ranker_path)),
            as_of_day=date_to_day(as_of), device=dev, config=config,
        )

    @staticmethod
    def _content_matrix(text_raw: np.ndarray, image_raw: np.ndarray) -> np.ndarray:
        """normalize(normalize(text) + normalize(image)) with a zero padding row first; in place to keep peak RAM low."""
        content = np.zeros((text_raw.shape[0] + 1, text_raw.shape[1]), dtype="float32")
        for part in (text_raw, image_raw):
            part /= np.linalg.norm(part, axis=1, keepdims=True).clip(1e-8)
            content[1:] += part
        content[1:] /= np.linalg.norm(content[1:], axis=1, keepdims=True).clip(1e-8)
        return content

    @staticmethod
    def _item_table(ckpt: dict, text_raw: np.ndarray, image_raw: np.ndarray) -> torch.Tensor:
        """Same maths as Tower.item_table() in notebook 02/03, from the raw state dict (no nn.Module, no 2x216 MB buffers)."""
        s = ckpt["state"]
        modality = ckpt.get("modality", "multimodal")
        pad = lambda m: torch.cat([torch.zeros(1, m.shape[1]), m])  # noqa: E731  (projections have no bias)
        with torch.no_grad():
            id_vectors = torch.sigmoid(s["id_gate"]) * s["id.weight"] * s["id_keep"]
            text_part = pad(torch.from_numpy(text_raw) @ s["text_proj.weight"].T)
            image_part = pad(torch.from_numpy(image_raw) @ s["image_proj.weight"].T)
            if modality == "id":
                vectors = id_vectors
            elif modality == "text":
                vectors = text_part
            elif modality == "image":
                vectors = image_part
            else:
                vectors = id_vectors + text_part + torch.sigmoid(s["image_gate"]) * image_part
            return F.normalize(vectors, dim=1)

    # ------------------------------------------------------------------ stats
    def _launch_prior(self, first_seen: np.ndarray, cutoff: int) -> np.ndarray:
        """Mean top-k cosine of each item to items that first sold in the `prior_days` before cutoff."""
        new_ids = np.where((first_seen < cutoff) & (first_seen >= cutoff - self.cfg.prior_days))[0]
        prior = np.zeros(self.n_items + 1, dtype="float32")
        if len(new_ids) == 0:
            return prior
        new_vecs = self.content_t[torch.from_numpy(new_ids).to(self.device)]
        for start in range(1, self.n_items + 1, 8192):
            block = self.content_t[start:start + 8192] @ new_vecs.T
            prior[start:start + block.shape[0]] = block.topk(min(self.cfg.prior_topk, len(new_ids)), dim=1).values.mean(1).cpu().numpy()
        return prior

    def _item_stats(self, st_idx, st_day, st_n, st_psum, *, cutoff: int) -> dict:
        n = self.n_items
        before = st_day < cutoff
        # first sale day, over transactions before the shop clock only (equivalent to the notebook, which
        # compares first_seen < cutoff over all rows, but never reads the future)
        first_seen = np.full(n + 1, 10 ** 9, dtype="int64")
        np.minimum.at(first_seen, st_idx[before], st_day[before])

        pop_all = np.bincount(st_idx[before], weights=st_n[before], minlength=n + 1)
        recent = before & (st_day >= cutoff - 7)
        pop_recent = np.bincount(st_idx[recent], weights=st_n[recent], minlength=n + 1)
        price_sum = np.bincount(st_idx[before], weights=st_psum[before], minlength=n + 1)
        price = np.where(pop_all > 0, price_sum / np.maximum(pop_all, 1), np.nan)
        price = np.where(np.isnan(price), np.nanmedian(price), price).clip(1e-4)
        age = np.where(first_seen < cutoff, cutoff - first_seen, -1)

        inactive = None
        if self.serving["active_days"]:
            window = (st_day < cutoff) & (st_day >= cutoff - self.serving["active_days"])
            active = np.bincount(st_idx[window], minlength=n + 1) > 0
            inactive = torch.tensor(~active, device=self.device)

        cold_pool = np.where(first_seen[1:] >= cutoff)[0] + 1   # no sale before the clock: truly cold
        prior = self._launch_prior(first_seen, cutoff)
        if len(cold_pool):
            mean, std = prior[cold_pool].mean(), prior[cold_pool].std() + 1e-6
        else:
            mean, std = 0.0, 1.0
        prior_z = (prior - mean) / std
        cold_pool_t = torch.from_numpy(cold_pool).to(self.device)
        keep = lambda order: [int(i) for i in order if i != 0]  # noqa: E731
        return {
            "cold_pool_t": cold_pool_t, "cold_pool_vecs": self.content_t[cold_pool_t],
            "cold_prior_z_t": torch.tensor(prior_z[cold_pool], device=self.device), "prior_z": prior_z, "inactive_t": inactive,
            "log_recent_t": torch.tensor(np.log1p(pop_recent), dtype=torch.float32, device=self.device),
            "pop_all": pop_all, "pop_recent": pop_recent, "price": price, "age": age.astype("float64"),
            "seen": (first_seen < cutoff).astype("float64"),
            "recent_ranked": keep(np.argsort(-pop_recent, kind="stable"))[: self.cfg.recent_k],
            "popular_ranked": keep(np.argsort(-pop_all, kind="stable"))[: self.cfg.popular_k],
            "new_pool": np.where((first_seen < cutoff) & (first_seen >= cutoff - self.cfg.new_days))[0],
            "first_seen": first_seen,
        }

    # -------------------------------------------------------- per-batch state
    @torch.no_grad()
    def _batch_state(self, histories: list[list[int]]) -> dict:
        cfg, st = self.cfg, self.stats
        hs = [h[-cfg.maxlen:] for h in histories]
        X = torch.tensor([[0] * (cfg.maxlen - len(h)) + h for h in hs], device=self.device)

        mask = X.ne(0)
        weights = torch.arange(1, X.shape[1] + 1, device=self.device)[None] * mask
        query = (self.table[X] * weights[:, :, None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
        query = F.normalize(query, dim=1)
        scores = query @ self.table.T
        scores[:, 0] = -torch.inf

        content = self.content_t
        anchor3 = F.normalize(content[X[:, -3:]].sum(1), dim=1)
        anchor_all = F.normalize(content[X].sum(1), dim=1)
        anchor10 = F.normalize(content[X[:, -10:]].sum(1), dim=1)

        cold_top = np.zeros((len(histories), 0), dtype="int64")
        if cfg.cold_k and len(st["cold_pool_t"]):
            usim = anchor10 @ st["cold_pool_vecs"].T
            usim_z = (usim - usim.mean(1, keepdim=True)) / usim.std(1, keepdim=True).clamp_min(1e-6)
            pick = (0.5 * usim_z + st["cold_prior_z_t"][None]).topk(min(cfg.cold_k, usim.shape[1]), dim=1).indices
            cold_top = st["cold_pool_t"][pick].cpu().numpy()

        content_scores = anchor3 @ content.T
        content_scores[:, 0] = -torch.inf

        serve = scores.clone()
        if self.serving["pop_weight"]:
            serve += self.serving["pop_weight"] * st["log_recent_t"][None]
        if self.serving["cold_bonus"]:
            serve[:, self.cold_t] += self.serving["cold_bonus"]
        if st["inactive_t"] is not None:
            serve[:, st["inactive_t"]] = -torch.inf
        serve[:, 0] = -torch.inf
        return {
            "scores": scores, "sorted": scores.sort(dim=1).values, "mu": scores[:, 1:].mean(1),
            "sd": scores[:, 1:].std(1).clamp_min(1e-6),
            "tower_top": serve.topk(cfg.tower_k, dim=1).indices.cpu().numpy(),
            "content_top": content_scores.topk(cfg.content_k, dim=1).indices.cpu().numpy(),
            "anchor3": anchor3, "anchor_all": anchor_all, "cold_top": cold_top,
        }

    def _build_sources(self, b: dict, i: int, hist_full: list[int]) -> dict[str, list[int]]:
        st, cfg = self.stats, self.cfg
        new_pool = st["new_pool"]
        new_top: list[int] = []
        if len(new_pool):
            sim = self.content_t[torch.from_numpy(new_pool).to(self.device)] @ b["anchor3"][i]
            new_top = new_pool[sim.topk(min(cfg.new_k, len(new_pool))).indices.cpu().numpy()].tolist()
        return {
            "tower": b["tower_top"][i].tolist(), "recent": st["recent_ranked"], "repeat": _unique_recent(hist_full, cfg.repeat_k),
            "new": new_top, "content": b["content_top"][i].tolist(), "popular": st["popular_ranked"], "cold": b["cold_top"][i].tolist(),
        }

    def _featurize(self, b: dict, i: int, items_arr, hist_full: list[int], sources: dict[str, list[int]]):
        """(len(items), 24) feature matrix for one user + raw tower scores. Column order == FEATURES_V2."""
        st = self.stats
        items_arr = np.asarray(items_arr, dtype="int64")
        t = torch.from_numpy(items_arr).to(self.device)
        raw = b["scores"][i, t]
        z = ((raw - b["mu"][i]) / b["sd"][i]).cpu().numpy()
        higher = (self.n_items + 1 - torch.searchsorted(b["sorted"][i], raw, right=True)).cpu().numpy()
        content_anchor = (self.content_t[t] @ b["anchor3"][i]).cpu().numpy()
        content_mean = (self.content_t[t] @ b["anchor_all"][i]).cpu().numpy()
        flags = [np.isin(items_arr, np.asarray(sources[name], dtype="int64")).astype("float32") for name in SOURCES]

        hist = np.asarray(hist_full, dtype="int64")
        hist_counts = np.bincount(hist, minlength=self.n_items + 1) if len(hist) else np.zeros(self.n_items + 1, dtype="int64")
        recent_hist = hist[-self.cfg.maxlen:]
        affinity = {}
        for name, codes in self.attr.items():
            share = np.bincount(codes[recent_hist], minlength=codes.max() + 1) / max(len(recent_hist), 1) if len(recent_hist) else np.zeros(codes.max() + 1)
            affinity[name] = share[codes[items_arr]]
        same_type_last = (self.attr["type"][items_arr] == self.attr["type"][hist[-1]]).astype("float32") if len(hist) else np.zeros(len(items_arr), dtype="float32")
        user_price = st["price"][recent_hist].mean() if len(recent_hist) else np.median(st["price"])

        pop_all, pop_recent = st["pop_all"][items_arr], st["pop_recent"][items_arr]
        columns = {
            "tower_z": z, "tower_logrank": np.log1p(higher), "log_pop_all": np.log1p(pop_all), "log_pop_recent": np.log1p(pop_recent),
            "recent_share": pop_recent / (pop_all + 1.0), "content_anchor": content_anchor, "content_mean": content_mean,
            "from_tower": flags[0], "from_recent": flags[1], "from_repeat": flags[2], "from_new": flags[3], "from_content": flags[4],
            "from_popular": flags[5], "from_cold": flags[6], "launch_prior": st["prior_z"][items_arr],
            "is_cold": (self.degree[items_arr] == 0).astype("float32"), "repeat_count": np.log1p(hist_counts[items_arr]),
            "age_days": np.clip(st["age"][items_arr], -1, 800) / 100.0, "seen_before": st["seen"][items_arr],
            "type_aff": affinity["type"], "dept_aff": affinity["dept"], "colour_aff": affinity["colour"], "same_type_last": same_type_last,
            "log_price_ratio": np.log(st["price"][items_arr] / user_price),
        }
        matrix = np.column_stack([columns[name] for name in FEATURES_V2]).astype("float32")
        return matrix, raw.cpu().numpy()

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def candidates_and_features(self, histories: list[list[int]]) -> list[dict]:
        """For each history (catalog indices, oldest first): union of candidates, feature matrix, raw tower scores, cold list."""
        b = self._batch_state(histories)
        out = []
        for i, hist in enumerate(histories):
            sources = self._build_sources(b, i, hist)
            union = list(dict.fromkeys(item for name in SOURCES for item in sources[name] if item != 0))
            x, raw = self._featurize(b, i, union, hist, sources)
            out.append({"items": np.asarray(union), "x": x, "raw": raw, "cold_list": sources["cold"], "sources": sources})
        return out

    def rank_many(
        self,
        histories: list[list[int]],
        *,
        cold_every: int = 0,
        exclude: Sequence[Sequence[int]] | None = None,
        allowed: np.ndarray | None = None,
        batch_size: int = 32,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Ranked (item indices, reranker scores) per history.

        `exclude[i]`: indices to drop for user i (e.g. already in cart); `allowed`: bool array over 0..n_items,
        items with False are never returned (e.g. no image in the shop). With `cold_every > 0` the cold channel
        owns positions every, 2*every, ... (scored 0.0: they come from the CLIP channel, not from the ranker).
        """
        results: list[tuple[np.ndarray, np.ndarray]] = []
        for start in range(0, len(histories), batch_size):
            chunk = histories[start:start + batch_size]
            for j, g in enumerate(self.candidates_and_features(chunk)):
                scores = self.booster.predict(g["x"])
                order = np.argsort(-scores, kind="stable")
                ranked, ranked_scores = g["items"][order].tolist(), dict(zip(g["items"][order].tolist(), scores[order].tolist()))
                cold_list = g["cold_list"]
                banned = set(exclude[start + j]) if exclude is not None else set()
                if allowed is not None:
                    banned |= set(np.flatnonzero(~allowed).tolist())
                if cold_every:
                    cold_list = [c for c in cold_list if c not in banned]
                    ranked = interleave_cold(ranked, cold_list, cold_every)
                ranked = [r for r in ranked if r not in banned]
                results.append((np.asarray(ranked, dtype="int64"), np.asarray([ranked_scores.get(r, 0.0) for r in ranked])))
        return results

    def recommend(
        self,
        history_item_ids: list[str],
        k: int = 10,
        *,
        cold_every: int = 0,
        exclude_history: bool = True,
        allowed: np.ndarray | None = None,
    ) -> RecommendResult:
        """`history_item_ids`: article_id strings (e.g. '0108775015'), most recent last. Unknown ids are ignored."""
        hist = [self.item2idx[i] for i in history_item_ids if i in self.item2idx]
        ranked, scores = self.rank_many(
            [hist], cold_every=cold_every, exclude=[hist] if exclude_history else None, allowed=allowed
        )[0]
        recs = [Recommendation(item_id=self.item_ids[int(i) - 1], score=float(s)) for i, s in zip(ranked[:k], scores[:k])]
        return RecommendResult(recommendations=recs, source="hm_reranker" if hist else "hm_reranker_no_history")
