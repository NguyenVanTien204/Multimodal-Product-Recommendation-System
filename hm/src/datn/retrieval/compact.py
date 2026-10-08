"""Shrink Jina CLIP v2's resident weights on CPU without changing what it computes.

Jina ships its weights as bfloat16 but `from_pretrained(torch_dtype=float32)` (needed because CPUs
without AVX512-BF16 run bf16 matmuls slowly) upcasts all 865 M parameters to fp32: ~3.5 GB, with a ~4 GB
peak while loading.  Loading the checkpoint as bf16 and upcasting *per matmul* keeps the arithmetic in
fp32, so the embeddings match the fp32 model (cosine >= 0.9999) while the weights take half the memory
and the load never exceeds ~1.8 GB.  Dynamic int8 was tried
and rejected: it collapsed the text tower's embeddings (cosine ~0.67 against fp32).
"""

from __future__ import annotations

import ctypes
import gc
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

log = logging.getLogger(__name__)

_EMBEDDING_MIN_ROWS = 100_000  # only the 250k-row token table is worth the extra cast


class Bf16Linear(nn.Module):
    """``nn.Linear`` with bf16 storage. Accepts Jina's extra ``task_id`` / ``residual`` arguments."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        self.in_features, self.out_features = weight.shape[1], weight.shape[0]
        # Jina sizes activations from ``next(module.parameters()).dtype``; keep an fp32 parameter present.
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.register_buffer("_w", weight.detach().to(torch.bfloat16))
        self.bias = None if bias is None else nn.Parameter(bias.detach().float(), requires_grad=False)

    @property
    def weight(self) -> torch.Tensor:  # the EVA vision tower calls F.linear(x, module.weight) directly
        return self._w.float()

    def forward(self, input: torch.Tensor, task_id=None, residual: bool = False):  # noqa: A002
        out = F.linear(input, self._w.float(), self.bias)
        return (out, input) if residual else out


class Bf16Embedding(nn.Module):
    def __init__(self, ref: nn.Embedding, weight: torch.Tensor) -> None:
        super().__init__()
        self.padding_idx = ref.padding_idx
        self.num_embeddings = ref.num_embeddings
        self.embedding_dim = ref.embedding_dim
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.register_buffer("table", weight.detach().to(torch.bfloat16))

    def forward(self, input: torch.Tensor, task_id=None) -> torch.Tensor:  # noqa: A002
        return F.embedding(input, self.table, self.padding_idx).float()


def _replace(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent_name) if parent_name else root, child, new)


def compact_weights(model: nn.Module) -> int:
    """Swap every Linear / token-embedding for a bf16-stored twin. Returns how many were swapped.

    Call after the LoRA adapter has been merged: parametrized layers are rebuilt from their
    (merged) ``original`` tensor, which also frees the adapter matrices.
    """
    swapped = 0
    # Keep only names: holding the old modules in a list would keep every fp32 weight alive next to its bf16 copy.
    for name in [n for n, _ in model.named_modules()]:
        if ".parametrizations" in name or name.endswith("parametrizations"):
            continue
        mod = model.get_submodule(name)
        original = None
        if parametrize.is_parametrized(mod, "weight"):
            original = mod.parametrizations.weight.original
            if isinstance(mod, nn.Embedding):
                new: nn.Module = Bf16Embedding(mod, original)
            elif isinstance(mod, nn.Linear):
                new = Bf16Linear(original, mod.bias)
            else:
                continue
        elif type(mod) is nn.Linear:
            new = Bf16Linear(mod.weight, mod.bias)
        elif type(mod) is nn.Embedding and mod.num_embeddings >= _EMBEDDING_MIN_ROWS:
            new = Bf16Embedding(mod, mod.weight)
        else:
            continue
        _replace(model, name, new)
        swapped += 1
        mod = new = original = None  # drop the last references so the fp32 tensors are freed now
    _rest_to_fp32(model)
    release_memory()
    return swapped


def _rest_to_fp32(model: nn.Module) -> None:
    """A model loaded as bf16 still holds small bf16 tensors (norms, biases, position tables): make them fp32."""
    for mod in model.modules():
        if isinstance(mod, (Bf16Linear, Bf16Embedding)):
            continue  # their bf16 buffers are the compact weights
        for key, buf in mod._buffers.items():
            if buf is not None and buf.dtype == torch.bfloat16:
                mod._buffers[key] = buf.float()
        for param in mod.parameters(recurse=False):
            if param.dtype == torch.bfloat16:
                param.data = param.data.float()


def release_memory() -> None:
    """Hand freed heap back to the OS (glibc keeps it otherwise, inflating RSS after a big batch)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # not glibc (Windows/macOS/musl)
        pass
