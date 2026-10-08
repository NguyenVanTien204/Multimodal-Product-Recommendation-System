from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

PAD_IDX = 0


@dataclass(frozen=True)
class ItemVocab:
    """Item id <-> index mapping built from the full catalog (`items.parquet`).

    Index 0 is reserved for padding. Index `num_items + 1` is a shared "unknown item"
    bucket for any interaction item_id absent from the catalog (should not normally
    happen since the pipeline already drops interactions without item metadata).
    """

    item2idx: dict[str, int]
    num_items: int  # count of real catalog items, indices [1, num_items]

    @property
    def oov_idx(self) -> int:
        return self.num_items + 1

    @property
    def vocab_size(self) -> int:
        # +1 for PAD (index 0), +1 for the OOV bucket.
        return self.num_items + 2

    def encode(self, item_id: str) -> int:
        return self.item2idx.get(item_id, self.oov_idx)


def build_item_vocab(items_path: Path) -> ItemVocab:
    items = pl.read_parquet(items_path, columns=["item_id"])
    ids = items["item_id"].to_list()
    item2idx = {item_id: idx + 1 for idx, item_id in enumerate(ids)}
    return ItemVocab(item2idx=item2idx, num_items=len(item2idx))


@dataclass(frozen=True)
class UserSequences:
    """Per-user chronological positive-interaction sequence, split-aligned.

    `train` holds every positive train interaction for the user, sorted by timestamp.
    `valid`/`test` hold the held-out leave-last-out target item index, or None when the
    held-out interaction was not positive (rating < 4) and therefore is not a valid
    "did we recommend the right thing" target under the implicit top-K protocol.
    """

    train: dict[str, list[int]]
    valid: dict[str, int]
    test: dict[str, int]


def _positive_sorted(df: pl.DataFrame, vocab_lookup: pl.DataFrame, oov_idx: int) -> pl.DataFrame:
    return (
        df.filter(pl.col("is_positive") == 1)
        .sort("timestamp")
        .join(vocab_lookup, on="item_id", how="left")
        .with_columns(pl.col("item_idx").fill_null(oov_idx))
        .select("user_id", "item_idx")
    )


def load_user_sequences(
    train_path: Path, valid_path: Path, test_path: Path, vocab: ItemVocab
) -> UserSequences:
    vocab_lookup = pl.DataFrame(
        {"item_id": list(vocab.item2idx.keys()), "item_idx": list(vocab.item2idx.values())}
    )
    train_df = _positive_sorted(pl.read_parquet(train_path), vocab_lookup, vocab.oov_idx)
    valid_df = _positive_sorted(pl.read_parquet(valid_path), vocab_lookup, vocab.oov_idx)
    test_df = _positive_sorted(pl.read_parquet(test_path), vocab_lookup, vocab.oov_idx)

    train: dict[str, list[int]] = {
        row["user_id"]: row["item_idx"]
        for row in train_df.group_by("user_id", maintain_order=True)
        .agg(pl.col("item_idx"))
        .to_dicts()
    }
    valid = {row["user_id"]: row["item_idx"] for row in valid_df.to_dicts()}
    test = {row["user_id"]: row["item_idx"] for row in test_df.to_dicts()}
    return UserSequences(train=train, valid=valid, test=test)


def pad_right(seq: list[int], max_len: int) -> np.ndarray:
    """Keep the most recent `max_len` ids (chronological order), right-padding with PAD_IDX.

    Right-padding (rather than left-padding) is deliberate: combined with causal
    self-attention, it guarantees every position -- including padded ones -- always
    has at least one unmasked key to attend to (position 0 is never padding since
    every sequence has length >= 1). Left-padding does not have this guarantee: the
    leading pad positions would have every allowed (causal) key also masked by the
    padding mask, producing an all -inf softmax row (NaN) whose zero-weighted-but-NaN
    value can still contaminate later positions through subsequent attention layers.
    """
    trimmed = seq[-max_len:]
    out = np.full(max_len, PAD_IDX, dtype=np.int64)
    if trimmed:
        out[: len(trimmed)] = trimmed
    return out


class SASRecTrainDataset(Dataset):
    """Next-item prediction pairs, one per user with >= 2 positive train interactions.

    Follows SASRec's "predict at every position" scheme: input = seq[:-1],
    target = seq[1:], both right-padded to `max_seq_len`, so a single user
    sequence yields a supervision signal at every valid time step instead of
    only at the very last one.
    """

    def __init__(self, sequences: dict[str, list[int]], max_seq_len: int) -> None:
        self.inputs: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        for seq in sequences.values():
            if len(seq) < 2:
                continue
            self.inputs.append(pad_right(seq[:-1], max_seq_len))
            self.targets.append(pad_right(seq[1:], max_seq_len))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.targets[idx])


@dataclass(frozen=True)
class EvalExample:
    user_id: str
    context: np.ndarray  # right-padded item indices, shape (max_seq_len,)
    target: int
    seen: frozenset[int]  # items to mask out of the candidate ranking (excludes target)


def build_eval_examples(
    sequences: UserSequences, max_seq_len: int, split: str
) -> list[EvalExample]:
    """Build leave-last-out eval examples for `split` in {"valid", "test"}.

    valid: context = train history, target = held-out valid item (if positive).
    test:  context = train + valid history, target = held-out test item (if positive).
    Users whose held-out interaction was not positive are skipped for that split,
    since a low rating is not a "liked" ground truth under the implicit protocol.
    """
    examples: list[EvalExample] = []
    for user_id, train_seq in sequences.train.items():
        if split == "valid":
            target = sequences.valid.get(user_id)
            context_seq = train_seq
            seen = set(train_seq)
        elif split == "test":
            target = sequences.test.get(user_id)
            valid_item = sequences.valid.get(user_id)
            context_seq = train_seq + ([valid_item] if valid_item is not None else [])
            seen = set(context_seq)
        else:
            raise ValueError(f"Unknown split: {split}")

        if target is None:
            continue
        seen.discard(target)
        examples.append(
            EvalExample(
                user_id=user_id,
                context=pad_right(context_seq, max_seq_len),
                target=target,
                seen=frozenset(seen),
            )
        )
    return examples
