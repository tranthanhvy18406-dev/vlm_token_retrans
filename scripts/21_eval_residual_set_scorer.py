import argparse
import importlib.util
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.budget_gate import zscore_1d
from src.corruption import corrupt_image_features
from src.dataset import load_jsonl
from src.llava_wrapper import LlavaRetransWrapper
from src.residual_set_scorer import ResidualSetScorer
from src.scorer import build_scorer_from_config
from src.utils import ensure_dir, get_torch_dtype, load_yaml, make_2d_positions, set_seed


def load_module(filename: str, module_name: str):
    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


paired = load_module("15_eval_paired_scorers.py", "paired_eval_helpers")


def load_frozen_scorer(config_path: str, checkpoint_path: str, device: str):
    cfg = load_yaml(config_path)
    scorer = build_scorer_from_config(cfg, dropout=0.0).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    scorer.load_state_dict(ckpt["model"])
    scorer.eval()
    for param in scorer.parameters():
        param.requires_grad_(False)
    return cfg, scorer


def load_residual_model(config_path: str, checkpoint_path: str, device: str):
    cfg = load_yaml(config_path)
    model = ResidualSetScorer(
        hidden_dim=int(cfg["set_scorer"].get("hidden_dim", 4096)),
        aux_dim=int(cfg["set_scorer"]["aux_dim"]),
        num_experts=3,
        d_model=int(cfg["set_scorer"].get("d_model", 256)),
        nhead=int(cfg["set_scorer"].get("nhead", 4)),
        layers=int(cfg["set_scorer"].get("layers", 1)),
        dropout=0.0,
        num_budgets=len(cfg["eval"]["budgets"]),
        delta_scale=float(cfg["set_scorer"].get("delta_scale", 0.1)),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return cfg, model


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
    return torch.stack([scores["legacy"], scores["query"], scores["coverage"]], dim=-1), scores


def base_scores_for_budget(score_map: dict[str, torch.Tensor], cfg: dict, budget: int) -> torch.Tensor:
    mapping = cfg["set_scorer"].get("base_expert_by_budget", {})
    expert = mapping.get(str(budget), mapping.get(int(budget), "query"))
    if expert not in score_map:
        raise ValueError(f"Unsupported base expert {expert!r}")
    return score_map[expert]


def score_residual_candidates(
    model: ResidualSetScorer,
    cfg: dict,
    legacy_scorer,
    query_scorer,
    coverage_scorer,
    legacy_cfg: dict,
    query_cfg: dict,
    coverage_cfg: dict,
    hidden: torch.Tensor,
    pos: torch.Tensor,
    reliability: torch.Tensor,
    damaged_float: torch.Tensor,
    candidates: torch.Tensor,
    query_hidden: torch.Tensor,
    attn_q_to_vis: torch.Tensor | None,
    attn_text_to_vis_mean: torch.Tensor | None,
    budget: int,
    budget_to_id: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    legacy_aux = paired.build_scorer_aux(
        hidden=hidden,
        pos=pos,
        reliability=reliability,
        damaged_float=damaged_float,
        candidates=candidates,
        scorer_cfg=legacy_cfg["scorer"],
        attn_q_to_vis=attn_q_to_vis,
        attn_text_to_vis_mean=attn_text_to_vis_mean,
    )
    query_aux = paired.build_scorer_aux(
        hidden=hidden,
        pos=pos,
        reliability=reliability,
        damaged_float=damaged_float,
        candidates=candidates,
        scorer_cfg=query_cfg["scorer"],
        attn_q_to_vis=attn_q_to_vis,
        attn_text_to_vis_mean=attn_text_to_vis_mean,
    )
    coverage_aux = paired.build_scorer_aux(
        hidden=hidden,
        pos=pos,
        reliability=reliability,
        damaged_float=damaged_float,
        candidates=candidates,
        scorer_cfg=coverage_cfg["scorer"],
        attn_q_to_vis=attn_q_to_vis,
        attn_text_to_vis_mean=attn_text_to_vis_mean,
    )
    h = hidden[candidates].float()
    with torch.no_grad():
        legacy_scores = legacy_scorer(h, legacy_aux, q=query_hidden).float()
        query_scores = query_scorer(h, query_aux, q=query_hidden).float()
        coverage_scores = coverage_scorer(h, coverage_aux, q=query_hidden).float()
        expert_matrix, score_map = expert_score_matrix(
            legacy_scores,
            query_scores,
            coverage_scores,
        )
        base_scores = base_scores_for_budget(score_map, cfg, budget)
        scores, _ = model(
            h=h,
            aux=coverage_aux,
            expert_scores=expert_matrix,
            base_scores=base_scores,
            q=query_hidden,
            budget_id=budget_to_id[budget],
        )
    return scores.float(), base_scores.float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="h6a_residual_set")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scorer", action="append", default=[])
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--metric_batch_size", type=int, default=8)
    parser.add_argument("--oracle_chunk_size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--delta_baselines", default="")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    cfg, model = load_residual_model(args.config, args.checkpoint, device="cuda")
    legacy_cfg = load_yaml(cfg["experts"]["legacy_config"])
    query_cfg = load_yaml(cfg["experts"]["query_config"])
    coverage_cfg = load_yaml(cfg["experts"]["coverage_config"])
    ref_cfgs = [load_yaml(paired.parse_scorer_arg(raw)[1]) for raw in args.scorer]
    paired.validate_configs(ref_cfgs + [legacy_cfg, query_cfg, coverage_cfg])
    wrapper_cfg = paired.choose_wrapper_config(ref_cfgs + [legacy_cfg, query_cfg, coverage_cfg])
    wrapper_cfg["model"]["attn_implementation"] = "eager"
    set_seed(wrapper_cfg["corruption"]["seed"])

    device = wrapper_cfg["model"]["device"]
    if device != "cuda":
        cfg, model = load_residual_model(args.config, args.checkpoint, device=device)
    legacy_cfg, legacy_scorer = load_frozen_scorer(
        cfg["experts"]["legacy_config"],
        cfg["experts"]["legacy_checkpoint"],
        device,
    )
    query_cfg, query_scorer = load_frozen_scorer(
        cfg["experts"]["query_config"],
        cfg["experts"]["query_checkpoint"],
        device,
    )
    coverage_cfg, coverage_scorer = load_frozen_scorer(
        cfg["experts"]["coverage_config"],
        cfg["experts"]["coverage_checkpoint"],
        device,
    )
    ref_specs = paired.load_scorer_specs(args.scorer, device=device) if args.scorer else []

    budgets = [int(k) for k in wrapper_cfg["eval"]["budgets"]]
    budget_to_id = {budget: idx for idx, budget in enumerate(budgets)}
    eval_metric = paired.normalize_metric_name(
        wrapper_cfg.get("eval", {}).get("metric", wrapper_cfg["oracle"].get("target", "gt_ce"))
    )
    delta_baselines = paired.parse_label_list(args.delta_baselines)
    method_names = ["random", "hidden_norm", "oracle_single"]
    method_names += [spec.label for spec in ref_specs] + ["fixed_base", args.label]

    print(f"eval jsonl: {args.jsonl}")
    print(f"max_samples: {args.max_samples}")
    print(f"eval metric: {eval_metric}")
    print(f"budgets: {budgets}")
    print(f"reference scorers: {[spec.label for spec in ref_specs]}")
    print(f"residual label: {args.label}")
    print(f"delta baselines: {delta_baselines}")

    wrapper = LlavaRetransWrapper(
        model_name=wrapper_cfg["model"]["name"],
        device=device,
        dtype=get_torch_dtype(wrapper_cfg["model"]["dtype"]),
        cache_dir=wrapper_cfg["model"].get("cache_dir"),
        local_files_only=bool(wrapper_cfg["model"].get("local_files_only", False)),
        device_map=wrapper_cfg["model"].get("device_map"),
        attn_implementation=wrapper_cfg["model"].get("attn_implementation"),
    )
    samples = load_jsonl(
        args.jsonl,
        image_root=wrapper_cfg["data"]["image_root"],
        max_samples=args.max_samples,
    )

    layer_idx = int(wrapper_cfg["model"]["layer_idx"])
    attention_layers = coverage_cfg["scorer"].get("attention_layers", [2, 4, 8, 12])
    stats_all = paired.make_stats(budgets, method_names)
    stats_vs = paired.make_stats(budgets, method_names)
    stats_other = paired.make_stats(budgets, method_names)

    for sample_idx, sample in enumerate(tqdm(samples, desc="residual set eval")):
        try:
            prepared = wrapper.prepare_inputs(sample.image, sample.question, sample.answer)
            full = prepared.image_features
            corrupted, damaged_mask, reliability = corrupt_image_features(
                full,
                drop_ratio=wrapper_cfg["corruption"]["drop_ratio"],
                mode=wrapper_cfg["corruption"]["mode"],
                mask_value=wrapper_cfg["corruption"]["mask_value"],
            )
            candidates = damaged_mask.nonzero(as_tuple=False).squeeze(-1)
            if candidates.numel() == 0:
                continue

            teacher_log_probs = None
            if eval_metric == "teacher_kl":
                teacher_log_probs = wrapper.compute_teacher_log_probs_from_image_features(
                    prepared,
                    full,
                )
            loss_full = paired.compute_metric_from_image_features(
                wrapper,
                prepared,
                full,
                eval_metric,
                teacher_log_probs,
            )
            loss_no = paired.compute_metric_from_image_features(
                wrapper,
                prepared,
                corrupted,
                eval_metric,
                teacher_log_probs,
            )
            vision_sensitive = bool(loss_full.item() < loss_no.item())

            hidden, query_hidden = wrapper.get_layer_visual_and_query_hidden(
                prepared=prepared,
                image_features=corrupted,
                layer_idx=layer_idx,
            )
            attn_q_to_vis, attn_text_to_vis_mean = wrapper.get_prompt_to_visual_attention(
                prepared=prepared,
                image_features=corrupted,
                layer_indices=attention_layers,
            )
            pos = make_2d_positions(full.shape[1], device=full.device)
            damaged_float = damaged_mask.float().unsqueeze(-1)
            oracle_ranked = paired.rank_oracle(
                wrapper=wrapper,
                prepared=prepared,
                corrupted=corrupted,
                full=full,
                candidates=candidates,
                metric_name=eval_metric,
                teacher_log_probs=teacher_log_probs,
                chunk_size=int(args.oracle_chunk_size),
            )

            ref_scores = {}
            for spec in ref_specs:
                ref_scores[spec.label] = paired.score_candidates(
                    spec=spec,
                    hidden=hidden,
                    pos=pos,
                    reliability=reliability,
                    damaged_float=damaged_float,
                    candidates=candidates,
                    query_hidden=query_hidden,
                    attn_q_to_vis=attn_q_to_vis,
                    attn_text_to_vis_mean=attn_text_to_vis_mean,
                )

            for budget in budgets:
                residual_scores, base_scores = score_residual_candidates(
                    model=model,
                    cfg=cfg,
                    legacy_scorer=legacy_scorer,
                    query_scorer=query_scorer,
                    coverage_scorer=coverage_scorer,
                    legacy_cfg=legacy_cfg,
                    query_cfg=query_cfg,
                    coverage_cfg=coverage_cfg,
                    hidden=hidden,
                    pos=pos,
                    reliability=reliability,
                    damaged_float=damaged_float,
                    candidates=candidates,
                    query_hidden=query_hidden,
                    attn_q_to_vis=attn_q_to_vis,
                    attn_text_to_vis_mean=attn_text_to_vis_mean,
                    budget=budget,
                    budget_to_id=budget_to_id,
                )
                selections = {
                    "random": paired.choose_random(candidates, budget),
                    "hidden_norm": paired.choose_by_scores(
                        candidates,
                        hidden[candidates].float().norm(dim=-1),
                        budget,
                    ),
                    "oracle_single": oracle_ranked[: min(budget, oracle_ranked.numel())],
                    "fixed_base": paired.choose_by_scores(candidates, base_scores, budget),
                    args.label: paired.choose_by_scores(candidates, residual_scores, budget),
                }
                for label, scores in ref_scores.items():
                    selections[label] = paired.choose_by_scores(candidates, scores, budget)

                losses = paired.compute_selection_losses(
                    wrapper=wrapper,
                    prepared=prepared,
                    corrupted=corrupted,
                    full=full,
                    selections=selections,
                    metric_name=eval_metric,
                    teacher_log_probs=teacher_log_probs,
                    batch_size=int(args.metric_batch_size),
                )
                paired.append_losses(stats_all, budget, loss_full, loss_no, losses)
                if vision_sensitive:
                    paired.append_losses(stats_vs, budget, loss_full, loss_no, losses)
                else:
                    paired.append_losses(stats_other, budget, loss_full, loss_no, losses)

            if sample_idx % 10 == 0:
                print(
                    f"sample={sample_idx} metric={eval_metric} "
                    f"full={loss_full.item():.6f} no={loss_no.item():.6f} "
                    f"vision_sensitive={vision_sensitive}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[WARN] sample {sample_idx} failed: {exc}", flush=True)

    summary = {
        "all": paired.print_stats_block(
            "All samples",
            stats_all,
            budgets,
            method_names,
            delta_baselines,
            eval_metric,
            int(args.bootstrap),
            seed=123,
        ),
        "vision_sensitive": paired.print_stats_block(
            "Vision-sensitive samples: full_metric < no_metric",
            stats_vs,
            budgets,
            method_names,
            delta_baselines,
            eval_metric,
            int(args.bootstrap),
            seed=456,
        ),
        "other": paired.print_stats_block(
            "Other samples: full_metric >= no_metric",
            stats_other,
            budgets,
            method_names,
            delta_baselines,
            eval_metric,
            int(args.bootstrap),
            seed=789,
        ),
    }
    if args.output_json:
        ensure_dir(os.path.dirname(args.output_json))
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summaries": summary}, f, indent=2)
        print(f"wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
