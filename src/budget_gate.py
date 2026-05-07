import math

import torch
import torch.nn as nn


def zscore_1d(values: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    values = values.float()
    return (values - values.mean()) / (values.std(unbiased=False) + eps)


def rank_percentile_1d(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    if values.numel() <= 1:
        return torch.zeros_like(values)
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks / float(values.numel() - 1)


def normalized_top_gap(values: torch.Tensor, k: int) -> torch.Tensor:
    values = values.float()
    if values.numel() < 2:
        return values.new_tensor(0.0)
    k = min(max(int(k), 2), values.numel())
    top = torch.topk(values, k=k).values
    gap = top[0] - top[-1]
    return gap / (values.std(unbiased=False) + 1.0e-6)


class FrozenBudgetGate(nn.Module):
    """
    Lightweight gate over two frozen scorer outputs.

    The gate consumes per-token features and returns g_i in [0, 1]. Final scores
    are computed outside the module as:

        s_i = g_i * z(query_score_i) + (1 - g_i) * z(coverage_score_i)
    """

    def __init__(
        self,
        input_dim: int,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features.float()).squeeze(-1))


def build_gate_features(
    query_scores: torch.Tensor,
    coverage_scores: torch.Tensor,
    query_aux: torch.Tensor,
    coverage_aux: torch.Tensor,
    query_hidden: torch.Tensor,
    budget: int,
    max_budget: int,
    num_visual_tokens: int,
) -> torch.Tensor:
    query_z = zscore_1d(query_scores)
    coverage_z = zscore_1d(coverage_scores)
    query_rank = rank_percentile_1d(query_scores)
    coverage_rank = rank_percentile_1d(coverage_scores)

    rank_corr = zscore_1d(query_rank).mul(zscore_1d(coverage_rank)).mean()
    score_corr = query_z.mul(coverage_z).mean()
    query_gap = normalized_top_gap(query_scores, budget)
    coverage_gap = normalized_top_gap(coverage_scores, budget)

    m = query_scores.numel()
    budget_value = float(budget)
    max_budget_value = max(float(max_budget), 1.0)
    num_visual_value = max(float(num_visual_tokens), 1.0)
    query_norm = query_hidden.float().norm() / math.sqrt(float(query_hidden.numel()))

    local = torch.stack(
        [
            query_z,
            coverage_z,
            query_z - coverage_z,
            (query_z - coverage_z).abs(),
            query_rank,
            coverage_rank,
            query_rank - coverage_rank,
        ],
        dim=-1,
    )
    budget_features = query_scores.new_tensor(
        [
            budget_value / max(float(m), 1.0),
            budget_value / max_budget_value,
            float(m) / num_visual_value,
            math.log(float(m) + 1.0) / math.log(num_visual_value + 1.0),
        ],
        dtype=torch.float32,
    )
    global_features = torch.stack(
        [
            query_scores.float().std(unbiased=False),
            coverage_scores.float().std(unbiased=False),
            query_gap,
            coverage_gap,
            score_corr,
            rank_corr,
            query_norm,
        ]
    )
    repeated = torch.cat([budget_features, global_features], dim=0).expand(m, -1)
    return torch.cat(
        [
            local,
            query_aux.float(),
            coverage_aux.float(),
            repeated,
        ],
        dim=-1,
    )


def gated_scores(
    gate: FrozenBudgetGate,
    query_scores: torch.Tensor,
    coverage_scores: torch.Tensor,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_values = gate(features)
    query_z = zscore_1d(query_scores)
    coverage_z = zscore_1d(coverage_scores)
    scores = gate_values * query_z + (1.0 - gate_values) * coverage_z
    return scores, gate_values
