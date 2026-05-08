import argparse
import importlib.util
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.budget_gate import zscore_1d
from src.losses import pairwise_ranking_loss, topk_boundary_pairwise_loss
from src.metrics import ndcg_at_k, recall_at_k
from src.residual_set_scorer import ResidualSetScorer
from src.utils import ensure_dir, load_yaml, set_seed


def load_gate_train_helpers():
    path = os.path.join(os.path.dirname(__file__), "18_train_three_expert_gate.py")
    spec = importlib.util.spec_from_file_location("three_expert_train_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


helpers = load_gate_train_helpers()


def parse_weight_map(value: dict | None) -> dict[int, float]:
    if not value:
        return {}
    return {int(k): float(v) for k, v in value.items()}


def expert_score_matrix(
    legacy_scores: torch.Tensor,
    query_scores: torch.Tensor,
    coverage_scores: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scores = {
        "legacy": zscore_1d(legacy_scores),
        "query": zscore_1d(query_scores),
        "coverage": zscore_1d(coverage_scores),
    }
    matrix = torch.stack([scores["legacy"], scores["query"], scores["coverage"]], dim=-1)
    return matrix, scores


def base_scores_for_budget(score_map: dict[str, torch.Tensor], cfg: dict, budget: int) -> torch.Tensor:
    mapping = cfg["set_scorer"].get("base_expert_by_budget", {})
    expert = mapping.get(str(budget), mapping.get(int(budget), "query"))
    if expert not in score_map:
        raise ValueError(f"Unsupported base expert {expert!r}; expected one of {sorted(score_map)}")
    return score_map[expert]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["corruption"]["seed"])
    device = cfg["model"]["device"]

    legacy_cfg, legacy_scorer = helpers.load_frozen_scorer(
        cfg["experts"]["legacy_config"],
        cfg["experts"]["legacy_checkpoint"],
        device,
    )
    query_cfg, query_scorer = helpers.load_frozen_scorer(
        cfg["experts"]["query_config"],
        cfg["experts"]["query_checkpoint"],
        device,
    )
    coverage_cfg, coverage_scorer = helpers.load_frozen_scorer(
        cfg["experts"]["coverage_config"],
        cfg["experts"]["coverage_checkpoint"],
        device,
    )

    paths = helpers.load_cache_paths(cfg["oracle"]["save_dir"])
    model = ResidualSetScorer(
        hidden_dim=int(cfg["set_scorer"].get("hidden_dim", 4096)),
        aux_dim=int(cfg["set_scorer"]["aux_dim"]),
        num_experts=3,
        d_model=int(cfg["set_scorer"].get("d_model", 256)),
        nhead=int(cfg["set_scorer"].get("nhead", 4)),
        layers=int(cfg["set_scorer"].get("layers", 1)),
        dropout=float(cfg["set_scorer"].get("dropout", 0.1)),
        num_budgets=len(cfg["eval"]["budgets"]),
        delta_scale=float(cfg["set_scorer"].get("delta_scale", 0.1)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    budgets = [int(k) for k in cfg["eval"]["budgets"]]
    budget_to_id = {budget: idx for idx, budget in enumerate(budgets)}
    epochs = int(cfg["train"]["epochs"])
    pairs_per_sample = int(cfg["train"]["pairs_per_sample"])
    boundary_pairs = int(cfg["train"].get("boundary_pairs_per_sample", 2048))
    boundary_multiplier = int(cfg["train"].get("boundary_negative_multiplier", 2))
    boundary_weights = parse_weight_map(cfg["train"].get("boundary_loss_weights"))
    delta_l2_weight = float(cfg["train"].get("delta_l2_weight", 0.0))

    ckpt_path = cfg["train"]["checkpoint_path"]
    ensure_dir(os.path.dirname(ckpt_path))

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        used = 0
        perm = torch.randperm(len(paths)).tolist()

        for idx in tqdm(perm, desc=f"epoch {epoch}"):
            cache = torch.load(paths[idx], map_location="cpu", weights_only=True)
            h, legacy_aux, query_aux, coverage_aux, gains, query_hidden = helpers.build_cache_batch(
                cache,
                legacy_cfg,
                query_cfg,
                coverage_cfg,
                device,
            )
            if h.size(0) < 4:
                continue

            legacy_scores, query_scores, coverage_scores = helpers.score_with_experts(
                h,
                legacy_aux,
                query_aux,
                coverage_aux,
                query_hidden,
                legacy_scorer,
                query_scorer,
                coverage_scorer,
            )
            expert_matrix, score_map = expert_score_matrix(
                legacy_scores,
                query_scores,
                coverage_scores,
            )

            loss = gains.new_tensor(0.0)
            for budget in budgets:
                base_scores = base_scores_for_budget(score_map, cfg, budget)
                scores, delta = model(
                    h=h,
                    aux=coverage_aux,
                    expert_scores=expert_matrix,
                    base_scores=base_scores,
                    q=query_hidden,
                    budget_id=budget_to_id[budget],
                )
                loss = loss + pairwise_ranking_loss(
                    scores=scores,
                    gains=gains,
                    pairs_per_sample=pairs_per_sample,
                )
                if budget in boundary_weights:
                    loss = loss + boundary_weights[budget] * topk_boundary_pairwise_loss(
                        scores=scores,
                        gains=gains,
                        k=budget,
                        pairs_per_sample=boundary_pairs,
                        negative_multiplier=boundary_multiplier,
                    )
                if delta_l2_weight > 0.0:
                    loss = loss + delta_l2_weight * delta.float().pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["train"]["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(cfg["train"]["grad_clip"]),
                )
            optimizer.step()

            total_loss += float(loss.item())
            used += 1

        model.eval()
        recalls = {budget: [] for budget in budgets}
        ndcgs = {budget: [] for budget in budgets}
        with torch.no_grad():
            for path in paths[: min(100, len(paths))]:
                cache = torch.load(path, map_location="cpu", weights_only=True)
                h, legacy_aux, query_aux, coverage_aux, gains, query_hidden = helpers.build_cache_batch(
                    cache,
                    legacy_cfg,
                    query_cfg,
                    coverage_cfg,
                    device,
                )
                if h.size(0) < 4:
                    continue
                legacy_scores, query_scores, coverage_scores = helpers.score_with_experts(
                    h,
                    legacy_aux,
                    query_aux,
                    coverage_aux,
                    query_hidden,
                    legacy_scorer,
                    query_scorer,
                    coverage_scorer,
                )
                expert_matrix, score_map = expert_score_matrix(
                    legacy_scores,
                    query_scores,
                    coverage_scores,
                )
                for budget in budgets:
                    base_scores = base_scores_for_budget(score_map, cfg, budget)
                    scores, _ = model(
                        h=h,
                        aux=coverage_aux,
                        expert_scores=expert_matrix,
                        base_scores=base_scores,
                        q=query_hidden,
                        budget_id=budget_to_id[budget],
                    )
                    k = min(budget, h.size(0))
                    recalls[budget].append(recall_at_k(scores, gains, k=k))
                    ndcgs[budget].append(ndcg_at_k(scores, gains, k=k))

        metric_text = " ".join(
            f"recall@{budget}={sum(recalls[budget]) / max(len(recalls[budget]), 1):.4f} "
            f"ndcg@{budget}={sum(ndcgs[budget]) / max(len(ndcgs[budget]), 1):.4f}"
            for budget in budgets
        )
        print(
            f"epoch={epoch} loss={total_loss / max(used, 1):.6f} {metric_text}",
            flush=True,
        )
        torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, ckpt_path)

    print(f"saved checkpoint to {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
