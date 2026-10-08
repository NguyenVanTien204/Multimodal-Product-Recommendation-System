from __future__ import annotations

import logging
import threading
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

MODEL_ID = "jinaai/jina-clip-v2"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.clip(norm, 1e-12, None)).astype(np.float32)


_CHUNKED_BAKE_ROWS = 100_000  # the 250k x 1024 token table: a one-shot merge allocates several 0.5-1 GB temporaries


def _merge_rows(original, lora, loraid: int, chunk: int = 16384) -> None:
    """``lora.lora_forward`` for a tall embedding table, one row block at a time (same maths, in fp32)."""
    a = lora.dropout_fn(lora.lora_A[loraid])  # (rows, rank)
    b = lora.lora_B[loraid]  # (rank, cols)
    for start in range(0, original.shape[0], chunk):
        stop = start + chunk
        delta = (a[start:stop].float() @ b.float()) * lora.scaling
        original[start:stop] = (original[start:stop].float() + delta).to(original.dtype)


class JinaClipEncoder:
    """Query-time encoder for the same Jina CLIP v2 space the catalog was embedded in.

    The catalog vectors (`data/embedding/*.npy`) were produced offline with
    `encode_text` (document side) and `encode_image`. Serving reuses the identical
    model so a user query lands in the same 1024-d space:

    - text queries use ``task="retrieval.query"`` (asymmetric retrieval), and are
      compared against both the item *text* vectors and the item *image* vectors
      (CLIP's text tower is aligned with its vision tower);
    - review passages are indexed with the default document task, exactly like the
      catalog text;
    - uploaded photos go through ``encode_image`` and are compared against both
      image vectors (visual similarity) and text vectors (image -> description).

    The model is loaded lazily and thread-safely: importing this module (and
    everything that only *type-references* an encoder) never pulls in torch.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "auto",
        cache_dir: str | None = None,
        bake_lora: bool = True,
        compact: bool | None = None,
    ) -> None:
        self.model_id = model_id
        self.device_pref = device
        self.cache_dir = cache_dir
        self.bake_lora = bake_lora
        # bf16-stored weights, fp32 math (see compact.py). None = on for CPU, off for GPU (already fp16).
        self.compact = compact
        self.compacted_layers = 0
        self.baked_layers = 0
        self._model = None
        self._ready = False  # set only once weights are merged/compacted, so no thread sees a half-built model
        self._lock = threading.Lock()
        self.device = "cpu"

    # ---- lifecycle ---------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._ready

    def load(self) -> "JinaClipEncoder":
        if self._ready:
            return self
        with self._lock:
            if self._ready:
                return self
            import torch

            if self.device_pref == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.device_pref
            want_compact = self.compact if self.compact is not None else self.device == "cpu"
            if self.device.startswith("cuda"):
                dtype = torch.float16
            else:
                # Jina's checkpoint is bf16. Loading it as fp32 first peaks at ~4 GB (it spills into swap under a
                # container limit); loading bf16 and keeping the matrices that way peaks at ~1.8 GB.
                dtype = torch.bfloat16 if want_compact else torch.float32
            log.info("Loading %s on %s (%s)", self.model_id, self.device, dtype)
            self._build(dtype)
            if want_compact and not self._compact_with_check():
                from .compact import release_memory

                self._model = None
                self.compact = False
                release_memory()
                self._build(torch.float32)  # full-precision fallback
            self._ready = True
        return self

    def _build(self, dtype) -> None:
        import torch
        from transformers import AutoModel

        # ``torch_dtype`` alone leaves most Jina parameters fp32 (its remote code builds them at the default
        # dtype); switching the default too is what actually keeps the load under ~3.5 GB.
        previous = torch.get_default_dtype()
        if dtype in (torch.bfloat16, torch.float16):
            torch.set_default_dtype(dtype)
        try:
            model = AutoModel.from_pretrained(
                self.model_id, trust_remote_code=True, torch_dtype=dtype, cache_dir=self.cache_dir
            )
        finally:
            torch.set_default_dtype(previous)
        model.to(self.device)
        model.eval()
        self._model = model
        self.compacted_layers = 0
        self.baked_layers = 0
        if self.bake_lora:
            self.baked_layers = self._bake_default_lora()
            log.info("merged the default LoRA adapter into %d text-tower layers", self.baked_layers)

    def _compact_with_check(self) -> bool:
        """Store weights as bf16 (fp32 math). False -> caller reloads in fp32 if it errors or looks broken."""
        from .compact import compact_weights

        if not self.baked_layers:  # compaction rebuilds the LoRA layers from their merged weights
            return False
        try:
            swapped = compact_weights(self._model)
            # Sanity probe: same product in two languages must sit closer than an unrelated product.
            vi, en, other = self._probe_vectors(
                ["áo thun nam cổ tròn", "men's crew neck t-shirt", "wireless noise cancelling headphones"]
            )
            ok = bool(np.isfinite(vi).all()) and float(vi @ en) > float(vi @ other) + 0.05
        except Exception as exc:  # noqa: BLE001
            log.warning("weight compaction failed (%s); reloading in full precision", exc)
            return False
        if not ok:
            log.warning("compacted encoder failed its sanity probe; reloading in full precision")
            return False
        self.compacted_layers = swapped
        log.info("compacted %d layers to bf16 storage", swapped)
        return True

    def _probe_vectors(self, texts: list[str]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            out = self._model.encode_text(texts, batch_size=len(texts), convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(np.asarray(out, dtype=np.float32))

    def _bake_default_lora(self) -> int:
        """Merge the text tower's default LoRA adapter into its weights, once.

        Jina CLIP v2's text tower rebuilds ``W + B@A * scale`` for every LoRA layer
        (including the 250,002 x 1024 embedding table) on *every* forward call. That is
        negligible on a GPU but takes ~20 s per query on a CPU. `encode_text` always runs
        with the same default adapter (``default_lora_task`` = retrieval.query; the ``task``
        argument only changes the instruction prefix), so merging it once gives identical
        embeddings and makes later calls cost only the transformer itself.
        """
        import torch

        text = getattr(self._model, "text_model", None)
        loraid = getattr(text, "default_loraid", None)
        if text is None or loraid is None:
            return 0
        baked = 0
        with torch.no_grad():
            for module in text.modules():
                params = getattr(module, "parametrizations", None)
                chain = getattr(params, "weight", None) if params is not None else None
                lora = chain[0] if chain is not None and len(chain) else None
                if lora is None or not hasattr(lora, "lora_forward"):
                    continue
                original = chain.original
                if original.shape[0] >= _CHUNKED_BAKE_ROWS and original.dim() == 2:
                    _merge_rows(original, lora, loraid)
                else:
                    original.copy_(lora.lora_forward(original, loraid))
                lora.lora_forward = lambda X, current_task=None: X  # already merged
                baked += 1
        return baked

    def _trim(self) -> None:
        if self.compacted_layers:  # a 512px image batch's activations would otherwise stay in RSS
            from .compact import release_memory

            release_memory()

    # ---- encoding ----------------------------------------------------------------
    @staticmethod
    def _fit_catalog(vectors: np.ndarray) -> np.ndarray:
        """Match the catalog's vector size. The H&M catalog was embedded with Matryoshka truncation to 512-d
        (DATN_VECTOR_SIZE=512): cut the unit vector and re-normalise, which is what `truncate_dim` does."""
        from . import schema as S

        if S.VECTOR_SIZE >= vectors.shape[1]:
            return vectors
        return _l2_normalize(vectors[:, : S.VECTOR_SIZE])

    def _text(self, texts: Sequence[str], task: str | None, batch_size: int) -> np.ndarray:
        self.load()
        import torch

        with torch.inference_mode(), self._lock:
            kwargs = {"task": task} if task else {}
            out = self._model.encode_text(
                list(texts), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False, **kwargs
            )
        return self._fit_catalog(_l2_normalize(np.asarray(out, dtype=np.float32)))

    def encode_query(self, texts: Sequence[str]) -> np.ndarray:
        """User queries -> (n, VECTOR_SIZE) unit vectors (1024 for Amazon, 512 for H&M)."""
        return self._text(texts, "retrieval.query", batch_size=16)

    def encode_passages(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Documents (product text, reviews) -> (n, 1024) unit vectors."""
        return self._text(texts, None, batch_size=batch_size)

    def encode_images(self, images: Sequence, batch_size: int = 8) -> np.ndarray:
        """PIL images -> (n, 1024) unit vectors."""
        self.load()
        import torch

        prepared = [img.convert("RGB") for img in images]
        with torch.inference_mode(), self._lock:
            out = self._model.encode_image(
                prepared, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
            )
        self._trim()
        return self._fit_catalog(_l2_normalize(np.asarray(out, dtype=np.float32)))
