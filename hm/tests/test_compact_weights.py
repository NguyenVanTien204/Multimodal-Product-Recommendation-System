import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from datn.retrieval.compact import Bf16Embedding, Bf16Linear, compact_weights


class _Identity(nn.Module):
    def forward(self, x):
        return x


def _bf16_exact(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).float()


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100_000, 8, padding_idx=0)
        self.plain = nn.Linear(8, 8)
        self.merged = nn.Linear(8, 4)
        parametrize.register_parametrization(self.merged, "weight", _Identity())
        for p in self.parameters():
            p.data = _bf16_exact(p.data)

    def forward(self, ids):
        return self.merged(torch.relu(self.plain(self.emb(ids))))


def test_compact_weights_matches_fp32_and_shrinks():
    torch.manual_seed(0)
    model = _Toy().eval()
    ids = torch.randint(0, 100_000, (3, 5))
    with torch.no_grad():
        expected = model(ids)
        fp32_bytes = sum(p.numel() * 4 for p in model.parameters())
        swapped = compact_weights(model)
        actual = model(ids)

    assert swapped == 3
    assert isinstance(model.emb, Bf16Embedding) and isinstance(model.plain, Bf16Linear)
    assert isinstance(model.merged, Bf16Linear)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    stored = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))
    assert stored < fp32_bytes * 0.6


def test_bf16_linear_keeps_jina_call_signature_and_weight_access():
    lin = Bf16Linear(_bf16_exact(torch.randn(4, 6)), torch.zeros(4))
    x = torch.randn(2, 6)
    out, residual = lin(x, task_id=None, residual=True)
    assert residual is x and out.shape == (2, 4)
    assert lin.weight.dtype == torch.float32 and lin.weight.shape == (4, 6)
    assert next(lin.parameters()).dtype == torch.float32  # Jina reads this to size activations
