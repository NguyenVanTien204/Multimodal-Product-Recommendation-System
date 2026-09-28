from __future__ import annotations

from torch import Tensor, nn

# Order matters: candidates.py builds each candidate's feature vector in exactly
# this order, and a saved checkpoint's `feature_names` field must match it.
FEATURE_NAMES = (
    "tower_score", "popularity_z", "rr_tower", "rr_popularity",
    "content_centroid_score", "last_item_content_score", "rr_content", "rr_last_item",
    "tower_sim_last", "tower_sim_mean", "tower_sim_max", "history_length",
    "from_tower", "from_content", "from_last_item",
)


class ResidualListwiseRanker(nn.Module):
    """Learns a residual correction on top of the standardized User Tower score.

    `forward` returns `x[..., 0] + blend * mlp(x)`, where feature 0 is always the raw
    tower score (see FEATURE_NAMES). The output layer is zero-initialized so an
    untrained ranker (or blend=0) exactly reproduces retrieval-only ranking -- the
    residual can only ever refine, never override, the tower's ordering from scratch.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor, blend: float = 1.0) -> Tensor:
        return x[..., 0] + blend * self.net(x).squeeze(-1)
