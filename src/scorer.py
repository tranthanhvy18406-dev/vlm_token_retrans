import torch
import torch.nn as nn


class MLPRetransScorer(nn.Module):
    """
    s_i = MLP(LN(h_i^l)) + w_p^T [x_i, y_i, r_i, m_i]
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        aux_dim: int = 4,
        hidden1: int = 1024,
        hidden2: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ln = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

        self.aux_head = nn.Linear(aux_dim, 1, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        aux: torch.Tensor,
        q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        h: [M, D]
        aux: [M, 4], order: [x, y, r, m]
        return score: [M]
        """
        h_score = self.mlp(self.ln(h)).squeeze(-1)
        aux_score = self.aux_head(aux.float()).squeeze(-1)
        return h_score + aux_score


class QueryConditionedScorer(nn.Module):
    """
    Score each visual token with explicit conditioning on a prompt/query hidden state.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        proj_dim: int = 512,
        aux_dim: int = 5,
        hidden1: int = 512,
        hidden2: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.v_ln = nn.LayerNorm(hidden_dim)
        self.q_ln = nn.LayerNorm(hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, proj_dim)
        self.q_proj = nn.Linear(hidden_dim, proj_dim)

        in_dim = proj_dim * 4 + aux_dim
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, h: torch.Tensor, aux: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        h: [M, D]
        q: [D] or [1, D], prompt/query hidden that can attend to image and question.
        aux: [M, aux_dim]
        return score: [M]
        """
        if q is None:
            raise ValueError("QueryConditionedScorer requires q.")
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = q.to(device=h.device, dtype=h.dtype)

        v = self.v_proj(self.v_ln(h))
        q = self.q_proj(self.q_ln(q)).expand_as(v)
        x = torch.cat([v, q, v * q, (v - q).abs(), aux.float()], dim=-1)
        return self.head(x).squeeze(-1)


def scorer_requires_query(scorer_cfg: dict) -> bool:
    return scorer_cfg.get("type", "mlp") == "query_conditioned"


def build_scorer_from_config(cfg: dict, dropout: float | None = None) -> nn.Module:
    scorer_cfg = cfg["scorer"]
    scorer_type = scorer_cfg.get("type", "mlp")
    dropout_value = scorer_cfg["dropout"] if dropout is None else dropout

    if scorer_type == "mlp":
        return MLPRetransScorer(
            hidden_dim=scorer_cfg["hidden_dim"],
            aux_dim=scorer_cfg["aux_dim"],
            hidden1=scorer_cfg["hidden1"],
            hidden2=scorer_cfg["hidden2"],
            dropout=dropout_value,
        )

    if scorer_type == "query_conditioned":
        return QueryConditionedScorer(
            hidden_dim=scorer_cfg["hidden_dim"],
            proj_dim=scorer_cfg.get("proj_dim", 512),
            aux_dim=scorer_cfg["aux_dim"],
            hidden1=scorer_cfg.get("hidden1", 512),
            hidden2=scorer_cfg.get("hidden2", 128),
            dropout=dropout_value,
        )

    raise ValueError(f"Unsupported scorer type: {scorer_type}")
