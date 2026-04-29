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

    def forward(self, h: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """
        h: [M, D]
        aux: [M, 4], order: [x, y, r, m]
        return score: [M]
        """
        h_score = self.mlp(self.ln(h)).squeeze(-1)
        aux_score = self.aux_head(aux.float()).squeeze(-1)
        return h_score + aux_score
