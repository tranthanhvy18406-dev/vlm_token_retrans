import torch
import torch.nn as nn


class ResidualSetScorer(nn.Module):
    """
    Set-aware residual scorer on top of frozen base expert scores.

    The model consumes damaged candidates as an unordered set and predicts a
    small residual delta. Final scores are:

        final_i = base_i + delta_scale * delta_i
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        aux_dim: int = 20,
        num_experts: int = 3,
        d_model: int = 256,
        nhead: int = 4,
        layers: int = 1,
        dropout: float = 0.1,
        num_budgets: int = 3,
        delta_scale: float = 0.1,
    ):
        super().__init__()
        self.delta_scale = float(delta_scale)

        self.h_ln = nn.LayerNorm(hidden_dim)
        self.h_proj = nn.Linear(hidden_dim, d_model)
        self.aux_proj = nn.Linear(aux_dim, d_model)
        self.score_proj = nn.Linear(num_experts + 1, d_model)
        self.q_ln = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, d_model)
        self.budget_emb = nn.Embedding(num_budgets, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        aux: torch.Tensor,
        expert_scores: torch.Tensor,
        base_scores: torch.Tensor,
        q: torch.Tensor,
        budget_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = q.to(device=h.device, dtype=h.dtype)

        budget = torch.tensor(int(budget_id), device=h.device, dtype=torch.long)
        score_features = torch.cat([expert_scores.float(), base_scores.float().unsqueeze(-1)], dim=-1)
        z = (
            self.h_proj(self.h_ln(h))
            + self.aux_proj(aux.float())
            + self.score_proj(score_features)
            + self.q_proj(self.q_ln(q)).expand(h.size(0), -1)
            + self.budget_emb(budget).unsqueeze(0)
        )
        ctx = self.encoder(z.unsqueeze(0)).squeeze(0)
        delta = self.delta_head(ctx).squeeze(-1)
        return base_scores.float() + self.delta_scale * delta.float(), delta.float()
