import torch
import torch.nn.functional as F


def pairwise_ranking_loss(
    scores: torch.Tensor,
    gains: torch.Tensor,
    pairs_per_sample: int = 2048,
    min_gain_gap: float = 1.0e-6,
) -> torch.Tensor:
    """
    scores: [M]
    gains: [M]
    If gain_i > gain_j, enforce score_i > score_j.
    """
    device = scores.device
    num_items = scores.numel()

    if num_items < 2:
        return scores.sum() * 0.0

    idx_i = torch.randint(0, num_items, (pairs_per_sample,), device=device)
    idx_j = torch.randint(0, num_items, (pairs_per_sample,), device=device)

    gain_i = gains[idx_i]
    gain_j = gains[idx_j]

    valid = (gain_i - gain_j).abs() > min_gain_gap
    if valid.sum() == 0:
        return scores.sum() * 0.0

    idx_i = idx_i[valid]
    idx_j = idx_j[valid]

    sign = torch.sign(gains[idx_i] - gains[idx_j])
    score_diff = scores[idx_i] - scores[idx_j]

    return F.softplus(-sign * score_diff).mean()


def listwise_kl_loss(
    scores: torch.Tensor,
    gains: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """
    Match softmax(scores) to softmax(oracle gains).
    """
    if scores.numel() < 2:
        return scores.sum() * 0.0

    target = torch.softmax(gains.float() / tau, dim=0)
    pred_log = torch.log_softmax(scores.float(), dim=0)
    return F.kl_div(pred_log, target, reduction="batchmean")
