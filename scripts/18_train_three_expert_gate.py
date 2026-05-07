import argparse
import glob
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.budget_gate import (
    FrozenBudgetSoftmaxGate,
    build_three_expert_gate_features,
    softmax_mixture_scores,
)
from src.features import aux_dim_for_mode, build_aux_features
from src.losses import pairwise_ranking_loss
from src.metrics import ndcg_at_k, recall_at_k
from src.scorer import build_scorer_from_config
from src.utils import ensure_dir, load_yaml, set_seed


def load_cache_paths(cache_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    if not paths:
        raise RuntimeError(f"No cache files found in {cache_dir}")
    return paths


def load_frozen_scorer(config_path: str, checkpoint_path: str, device: str):
    cfg = load_yaml(config_path)
    scorer = build_scorer_from_config(cfg, dropout=0.0).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    scorer.load_state_dict(ckpt["model"])
    scorer.eval()
    for param in scorer.parameters():
        param.requires_grad_(False)
    return cfg, scorer


def build_aux(cache: dict, valid, scorer_cfg: dict, device: str) -> torch.Tensor:
    hidden = cache["hidden"].to(device=device, dtype=torch.float32)
    pos = cache["pos"].to(device=device, dtype=torch.float32)
    reliability = cache["reliability"].to(device=device, dtype=torch.float32)
    damaged_float = cache["damaged_float"].to(device=device, dtype=torch.float32)

    attn_q_to_vis = cache.get("attn_q_to_vis")
    attn_text_to_vis_mean = cache.get("attn_text_to_vis_mean")
    if attn_q_to_vis is not None:
        attn_q_to_vis = attn_q_to_vis.to(device=device, dtype=torch.float32)
    if attn_text_to_vis_mean is not None:
        attn_text_to_vis_mean = attn_text_to_vis_mean.to(device=device, dtype=torch.float32)

    aux_mode = scorer_cfg.get("aux_mode", "legacy")
    pos_fourier_bands = int(scorer_cfg.get("pos_fourier_bands", 0))
    attention_layers = scorer_cfg.get("attention_layers", [])
    aux = build_aux_features(
        hidden=hidden,
        pos=pos,
        reliability=reliability,
        damaged_float=damaged_float,
        selection=valid,
        mode=aux_mode,
        pos_fourier_bands=pos_fourier_bands,
        attn_q_to_vis=attn_q_to_vis,
        attn_text_to_vis_mean=attn_text_to_vis_mean,
    )
    expected = int(scorer_cfg["aux_dim"])
    inferred = aux_dim_for_mode(aux_mode, pos_fourier_bands, attention_layers)
    if aux.size(-1) != expected or inferred != expected:
        raise RuntimeError(
            "Aux dimension mismatch: "
            f"config={expected}, actual={aux.size(-1)}, inferred={inferred}, mode={aux_mode}"
        )
    return aux


def build_cache_batch(
    cache: dict,
    legacy_cfg: dict,
    query_cfg: dict,
    coverage_cfg: dict,
    device: str,
):
    hidden = cache["hidden"].to(device=device, dtype=torch.float32)
    damaged_mask = cache["damaged_mask"].to(device=device)
    oracle_gain = cache["oracle_gain"].to(device=device, dtype=torch.float32)
    valid = damaged_mask & torch.isfinite(oracle_gain)

    h = hidden[valid]
    gains = oracle_gain[valid]
    query_hidden = cache["query_hidden"].to(device=device, dtype=torch.float32)
    legacy_aux = build_aux(cache, valid, legacy_cfg["scorer"], device)
    query_aux = build_aux(cache, valid, query_cfg["scorer"], device)
    coverage_aux = build_aux(cache, valid, coverage_cfg["scorer"], device)
    return h, legacy_aux, query_aux, coverage_aux, gains, query_hidden


def topk_boundary_pairwise_loss(
    scores: torch.Tensor,
    gains: torch.Tensor,
    k: int,
    pairs: int,
) -> torch.Tensor:
    n = scores.numel()
    if n < 2:
        return scores.sum() * 0.0
    k = min(int(k), n - 1)
    if k <= 0:
        return scores.sum() * 0.0

    oracle_top = torch.topk(gains, k=k).indices
    pred_top = torch.topk(scores.detach(), k=min(2 * k, n)).indices
    oracle_mask = torch.zeros(n, dtype=torch.bool, device=scores.device)
    oracle_mask[oracle_top] = True
    hard_neg = pred_top[~oracle_mask[pred_top]]
    if hard_neg.numel() == 0:
        rest = (~oracle_mask).nonzero(as_tuple=False).squeeze(-1)
        if rest.numel() == 0:
            return scores.sum() * 0.0
        hard_neg = rest

    idx_i = oracle_top[torch.randint(0, oracle_top.numel(), (pairs,), device=scores.device)]
    idx_j = hard_neg[torch.randint(0, hard_neg.numel(), (pairs,), device=scores.device)]
    return F.softplus(-(scores[idx_i] - scores[idx_j])).mean()


def parse_weight_map(value: dict | None) -> dict[int, float]:
    if not value:
        return {}
    return {int(k): float(v) for k, v in value.items()}


def score_with_experts(
    h: torch.Tensor,
    legacy_aux: torch.Tensor,
    query_aux: torch.Tensor,
    coverage_aux: torch.Tensor,
    query_hidden: torch.Tensor,
    legacy_scorer,
    query_scorer,
    coverage_scorer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        legacy_scores = legacy_scorer(h, legacy_aux, q=query_hidden).float()
        query_scores = query_scorer(h, query_aux, q=query_hidden).float()
        coverage_scores = coverage_scorer(h, coverage_aux, q=query_hidden).float()
    return legacy_scores, query_scores, coverage_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["corruption"]["seed"])
    device = cfg["model"]["device"]

    legacy_cfg, legacy_scorer = load_frozen_scorer(
        cfg["gate"]["legacy_config"],
        cfg["gate"]["legacy_checkpoint"],
        device,
    )
    query_cfg, query_scorer = load_frozen_scorer(
        cfg["gate"]["query_config"],
        cfg["gate"]["query_checkpoint"],
        device,
    )
    coverage_cfg, coverage_scorer = load_frozen_scorer(
        cfg["gate"]["coverage_config"],
        cfg["gate"]["coverage_checkpoint"],
        device,
    )

    paths = load_cache_paths(cfg["oracle"]["save_dir"])
    gate = FrozenBudgetSoftmaxGate(
        input_dim=int(cfg["gate"]["input_dim"]),
        num_experts=3,
        hidden1=int(cfg["gate"].get("hidden1", 128)),
        hidden2=int(cfg["gate"].get("hidden2", 64)),
        dropout=float(cfg["gate"].get("dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    budgets = [int(k) for k in cfg["eval"]["budgets"]]
    max_budget = int(cfg["gate"].get("max_budget", max(budgets)))
    num_visual_tokens = int(cfg["gate"].get("num_visual_tokens", 576))
    epochs = int(cfg["train"]["epochs"])
    pairs_per_sample = int(cfg["train"]["pairs_per_sample"])
    boundary_pairs = int(cfg["train"].get("boundary_pairs_per_sample", 1024))
    pairwise_weight = float(cfg["train"].get("pairwise_weight", 1.0))
    boundary_weights = parse_weight_map(cfg["train"].get("boundary_loss_weights"))
    budget_prior = {
        int(k): torch.tensor(v, dtype=torch.float32, device=device)
        for k, v in (cfg["train"].get("budget_prior_weights") or {}).items()
    }
    prior_weight = float(cfg["train"].get("budget_prior_loss_weight", 0.0))
    ckpt_path = cfg["train"]["checkpoint_path"]
    ensure_dir(os.path.dirname(ckpt_path))

    for epoch in range(epochs):
        gate.train()
        total_loss = 0.0
        total_weights = {k: [] for k in budgets}
        used = 0
        perm = torch.randperm(len(paths)).tolist()

        for idx in tqdm(perm, desc=f"epoch {epoch}"):
            cache = torch.load(paths[idx], map_location="cpu", weights_only=True)
            h, legacy_aux, query_aux, coverage_aux, gains, query_hidden = build_cache_batch(
                cache,
                legacy_cfg,
                query_cfg,
                coverage_cfg,
                device,
            )
            if h.size(0) < 4:
                continue

            legacy_scores, query_scores, coverage_scores = score_with_experts(
                h,
                legacy_aux,
                query_aux,
                coverage_aux,
                query_hidden,
                legacy_scorer,
                query_scorer,
                coverage_scorer,
            )

            loss = gains.new_tensor(0.0)
            for budget in budgets:
                features = build_three_expert_gate_features(
                    legacy_scores=legacy_scores,
                    query_scores=query_scores,
                    coverage_scores=coverage_scores,
                    legacy_aux=legacy_aux,
                    query_aux=query_aux,
                    coverage_aux=coverage_aux,
                    query_hidden=query_hidden,
                    budget=budget,
                    max_budget=max_budget,
                    num_visual_tokens=num_visual_tokens,
                )
                scores, weights = softmax_mixture_scores(
                    gate,
                    [legacy_scores, query_scores, coverage_scores],
                    features,
                )
                loss = loss + pairwise_weight * pairwise_ranking_loss(
                    scores=scores,
                    gains=gains,
                    pairs_per_sample=pairs_per_sample,
                )
                if boundary_weights.get(budget, 0.0) > 0.0:
                    loss = loss + boundary_weights[budget] * topk_boundary_pairwise_loss(
                        scores=scores,
                        gains=gains,
                        k=budget,
                        pairs=boundary_pairs,
                    )
                if prior_weight > 0.0 and budget in budget_prior:
                    mean_weights = weights.mean(dim=0)
                    loss = loss + prior_weight * F.mse_loss(mean_weights, budget_prior[budget])
                total_weights[budget].append(weights.mean(dim=0).detach().cpu())

            loss = loss / max(len(budgets), 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["train"]["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    gate.parameters(),
                    max_norm=float(cfg["train"]["grad_clip"]),
                )
            optimizer.step()
            total_loss += float(loss.item())
            used += 1

        gate.eval()
        recalls = {k: [] for k in budgets}
        ndcgs = {k: [] for k in budgets}
        with torch.no_grad():
            for path in paths[: min(100, len(paths))]:
                cache = torch.load(path, map_location="cpu", weights_only=True)
                h, legacy_aux, query_aux, coverage_aux, gains, query_hidden = build_cache_batch(
                    cache,
                    legacy_cfg,
                    query_cfg,
                    coverage_cfg,
                    device,
                )
                if h.size(0) < 4:
                    continue
                expert_scores = score_with_experts(
                    h,
                    legacy_aux,
                    query_aux,
                    coverage_aux,
                    query_hidden,
                    legacy_scorer,
                    query_scorer,
                    coverage_scorer,
                )
                legacy_scores, query_scores, coverage_scores = expert_scores
                for budget in budgets:
                    features = build_three_expert_gate_features(
                        legacy_scores=legacy_scores,
                        query_scores=query_scores,
                        coverage_scores=coverage_scores,
                        legacy_aux=legacy_aux,
                        query_aux=query_aux,
                        coverage_aux=coverage_aux,
                        query_hidden=query_hidden,
                        budget=budget,
                        max_budget=max_budget,
                        num_visual_tokens=num_visual_tokens,
                    )
                    scores, _ = softmax_mixture_scores(gate, list(expert_scores), features)
                    k = min(budget, h.size(0))
                    recalls[budget].append(recall_at_k(scores, gains, k=k))
                    ndcgs[budget].append(ndcg_at_k(scores, gains, k=k))

        weight_parts = []
        for budget in budgets:
            if total_weights[budget]:
                weights = torch.stack(total_weights[budget]).mean(dim=0)
            else:
                weights = torch.zeros(3)
            weight_parts.append(
                f"w@{budget}=[s5:{weights[0]:.3f},h1a:{weights[1]:.3f},s7c:{weights[2]:.3f}]"
            )
        metric_parts = []
        for budget in budgets:
            mean_recall = sum(recalls[budget]) / max(len(recalls[budget]), 1)
            mean_ndcg = sum(ndcgs[budget]) / max(len(ndcgs[budget]), 1)
            metric_parts.append(f"recall@{budget}={mean_recall:.4f} ndcg@{budget}={mean_ndcg:.4f}")
        print(
            f"epoch={epoch} loss={total_loss / max(used, 1):.6f} "
            f"{' '.join(weight_parts)} {' '.join(metric_parts)}"
        )

        torch.save(
            {
                "model": gate.state_dict(),
                "config": cfg,
                "epoch": epoch,
            },
            ckpt_path,
        )

    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
