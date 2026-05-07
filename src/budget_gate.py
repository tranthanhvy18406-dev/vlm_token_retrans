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


class FrozenBudgetSoftmaxGate(nn.Module):
    """
    Softmax gate over N frozen scorer outputs.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, self.num_experts),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(features.float()), dim=-1)


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


def build_three_expert_gate_features(
    legacy_scores: torch.Tensor,
    query_scores: torch.Tensor,
    coverage_scores: torch.Tensor,
    legacy_aux: torch.Tensor,
    query_aux: torch.Tensor,
    coverage_aux: torch.Tensor,
    query_hidden: torch.Tensor,
    budget: int,
    max_budget: int,
    num_visual_tokens: int,
) -> torch.Tensor:
    scores = [legacy_scores.float(), query_scores.float(), coverage_scores.float()]
    score_z = [zscore_1d(item) for item in scores]
    ranks = [rank_percentile_1d(item) for item in scores]

    diff_lq = score_z[1] - score_z[0]
    diff_cl = score_z[2] - score_z[0]
    diff_qc = score_z[1] - score_z[2]
    local = torch.stack(
        [
            score_z[0],
            score_z[1],
            score_z[2],
            ranks[0],
            ranks[1],
            ranks[2],
            diff_lq,
            diff_cl,
            diff_qc,
            diff_lq.abs(),
            diff_cl.abs(),
            diff_qc.abs(),
        ],
        dim=-1,
    )

    m = legacy_scores.numel()
    budget_value = float(budget)
    max_budget_value = max(float(max_budget), 1.0)
    num_visual_value = max(float(num_visual_tokens), 1.0)
    budget_features = legacy_scores.new_tensor(
        [
            budget_value / max(float(m), 1.0),
            budget_value / max_budget_value,
            float(m) / num_visual_value,
            math.log(float(m) + 1.0) / math.log(num_visual_value + 1.0),
        ],
        dtype=torch.float32,
    )

    stds = [item.float().std(unbiased=False) for item in scores]
    gaps = [normalized_top_gap(item, budget) for item in scores]
    score_corrs = [
        score_z[0].mul(score_z[1]).mean(),
        score_z[0].mul(score_z[2]).mean(),
        score_z[1].mul(score_z[2]).mean(),
    ]
    rank_corrs = [
        zscore_1d(ranks[0]).mul(zscore_1d(ranks[1])).mean(),
        zscore_1d(ranks[0]).mul(zscore_1d(ranks[2])).mean(),
        zscore_1d(ranks[1]).mul(zscore_1d(ranks[2])).mean(),
    ]
    query_norm = query_hidden.float().norm() / math.sqrt(float(query_hidden.numel()))
    global_features = torch.stack(
        [
            stds[0],
            stds[1],
            stds[2],
            gaps[0],
            gaps[1],
            gaps[2],
            score_corrs[0],
            score_corrs[1],
            score_corrs[2],
            rank_corrs[0],
            rank_corrs[1],
            rank_corrs[2],
            query_norm,
        ]
    )
    repeated = torch.cat([budget_features, global_features], dim=0).expand(m, -1)
    return torch.cat(
        [
            local,
            legacy_aux.float(),
            query_aux.float(),
            coverage_aux.float(),
            repeated,
        ],
        dim=-1,
    )


def softmax_mixture_scores(
    gate: FrozenBudgetSoftmaxGate,
    expert_scores: list[torch.Tensor],
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = gate(features)
    stacked = torch.stack([zscore_1d(item) for item in expert_scores], dim=-1)
    scores = (weights * stacked).sum(dim=-1)
    return scores, weights
