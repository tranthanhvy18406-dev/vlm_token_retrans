import torch


def recall_at_k(
    scores: torch.Tensor,
    gains: torch.Tensor,
    k: int,
    oracle_top_m: int | None = None,
) -> float:
    num_items = scores.numel()
    k = min(k, num_items)

    if oracle_top_m is None:
        oracle_top_m = k
    oracle_top_m = min(oracle_top_m, num_items)

    pred = torch.topk(scores, k=k).indices
    oracle = torch.topk(gains, k=oracle_top_m).indices

    pred_set = set(pred.detach().cpu().tolist())
    oracle_set = set(oracle.detach().cpu().tolist())

    return len(pred_set & oracle_set) / max(len(oracle_set), 1)


def ndcg_at_k(scores: torch.Tensor, gains: torch.Tensor, k: int) -> float:
    num_items = scores.numel()
    k = min(k, num_items)

    pred_idx = torch.topk(scores, k=k).indices
    ideal_idx = torch.topk(gains, k=k).indices
    min_gain = gains.float().min()

    def dcg(idx):
        rel = (gains[idx].float() - min_gain).clamp_min(0.0)
        discounts = torch.log2(torch.arange(2, k + 2, device=gains.device).float())
        return ((2.0 ** rel - 1.0) / discounts).sum()

    ideal = dcg(ideal_idx)
    if ideal.abs() < 1e-12:
        return 0.0

    return float((dcg(pred_idx) / ideal).item())


def recovery_ratio(loss_no: float, loss_method: float, loss_full: float) -> float:
    denom = loss_no - loss_full
    if abs(denom) < 1e-12:
        return 0.0
    return (loss_no - loss_method) / denom
