import argparse
import importlib.util
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.budget_gate import FrozenBudgetGate, build_gate_features, gated_scores
from src.corruption import corrupt_image_features
from src.dataset import load_jsonl
from src.llava_wrapper import LlavaRetransWrapper
from src.scorer import build_scorer_from_config
from src.utils import get_torch_dtype, load_yaml, make_2d_positions, set_seed


def load_paired_eval_module():
    path = os.path.join(os.path.dirname(__file__), "15_eval_paired_scorers.py")
    spec = importlib.util.spec_from_file_location("paired_eval_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


paired = load_paired_eval_module()


def load_frozen_scorer(config_path: str, checkpoint_path: str, device: str):
    cfg = load_yaml(config_path)
    scorer = build_scorer_from_config(cfg, dropout=0.0).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    scorer.load_state_dict(ckpt["model"])
    scorer.eval()
    for param in scorer.parameters():
        param.requires_grad_(False)
    return cfg, scorer


def load_gate(config_path: str, checkpoint_path: str, device: str):
    cfg = load_yaml(config_path)
    gate = FrozenBudgetGate(
        input_dim=int(cfg["gate"]["input_dim"]),
        hidden1=int(cfg["gate"].get("hidden1", 128)),
        hidden2=int(cfg["gate"].get("hidden2", 64)),
        dropout=0.0,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    gate.load_state_dict(ckpt["model"])
    gate.eval()

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
    return cfg, gate, query_cfg, query_scorer, coverage_cfg, coverage_scorer


def score_gate_candidates(
    gate,
    query_scorer,
    coverage_scorer,
    gate_cfg: dict,
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
) -> torch.Tensor:
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
        query_scores = query_scorer(h, query_aux, q=query_hidden).float()
        coverage_scores = coverage_scorer(h, coverage_aux, q=query_hidden).float()
        features = build_gate_features(
            query_scores=query_scores,
            coverage_scores=coverage_scores,
            query_aux=query_aux,
            coverage_aux=coverage_aux,
            query_hidden=query_hidden,
            budget=budget,
            max_budget=int(gate_cfg["gate"].get("max_budget", 64)),
            num_visual_tokens=int(gate_cfg["gate"].get("num_visual_tokens", hidden.size(0))),
        )
        scores, _ = gated_scores(gate, query_scores, coverage_scores, features)
    return scores.float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate_label", default="g1_gate")
    parser.add_argument("--gate_config", required=True)
    parser.add_argument("--gate_checkpoint", required=True)
    parser.add_argument(
        "--scorer",
        action="append",
        default=[],
        help="Optional reference scorer: label,config_path,checkpoint_path",
    )
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--metric_batch_size", type=int, default=8)
    parser.add_argument("--oracle_chunk_size", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--delta_baselines", default="")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    gate_cfg = load_yaml(args.gate_config)
    query_cfg = load_yaml(gate_cfg["gate"]["query_config"])
    coverage_cfg = load_yaml(gate_cfg["gate"]["coverage_config"])
    ref_cfgs = [load_yaml(paired.parse_scorer_arg(raw)[1]) for raw in args.scorer]
    paired.validate_configs(ref_cfgs + [query_cfg, coverage_cfg])
    wrapper_cfg = paired.choose_wrapper_config(ref_cfgs + [query_cfg, coverage_cfg])
    wrapper_cfg["model"]["attn_implementation"] = "eager"
    set_seed(wrapper_cfg["corruption"]["seed"])

    device = wrapper_cfg["model"]["device"]
    ref_specs = paired.load_scorer_specs(args.scorer, device=device) if args.scorer else []
    (
        loaded_gate_cfg,
        gate,
        query_cfg,
        query_scorer,
        coverage_cfg,
        coverage_scorer,
    ) = load_gate(args.gate_config, args.gate_checkpoint, device)

    budgets = [int(k) for k in wrapper_cfg["eval"]["budgets"]]
    eval_metric = paired.normalize_metric_name(
        wrapper_cfg.get("eval", {}).get("metric", wrapper_cfg["oracle"].get("target", "gt_ce"))
    )
    oracle_chunk_size = int(
        args.oracle_chunk_size or wrapper_cfg["oracle"].get("candidate_chunk_size", 16)
    )
    method_names = ["random", "hidden_norm", "oracle_single"]
    method_names += [spec.label for spec in ref_specs] + [args.gate_label]
    delta_baselines = paired.parse_label_list(args.delta_baselines)

    print(f"eval jsonl: {args.jsonl}")
    print(f"max_samples: {args.max_samples}")
    print(f"eval metric: {eval_metric}")
    print(f"budgets: {budgets}")
    print(f"reference scorers: {[spec.label for spec in ref_specs]}")
    print(f"gate label: {args.gate_label}")
    print(f"delta baselines: {delta_baselines}")
    print(f"attn_implementation: {wrapper_cfg['model'].get('attn_implementation')}")
    print(f"oracle chunk size: {oracle_chunk_size}")

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

    for sample_idx, sample in enumerate(tqdm(samples, desc="budget gate eval")):
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
            teacher_log_probs = None
            if eval_metric == "teacher_kl":
                teacher_log_probs = wrapper.compute_teacher_log_probs_from_image_features(
                    prepared,
                    full,
                )
            loss_full = paired.compute_metric_from_image_features(
                wrapper=wrapper,
                prepared=prepared,
                image_features=full,
                metric_name=eval_metric,
                teacher_log_probs=teacher_log_probs,
            )
            loss_no = paired.compute_metric_from_image_features(
                wrapper=wrapper,
                prepared=prepared,
                image_features=corrupted,
                metric_name=eval_metric,
                teacher_log_probs=teacher_log_probs,
            )
            oracle_ranked = paired.rank_oracle(
                wrapper=wrapper,
                prepared=prepared,
                corrupted=corrupted,
                full=full,
                candidates=candidates,
                metric_name=eval_metric,
                teacher_log_probs=teacher_log_probs,
                chunk_size=oracle_chunk_size,
            )

            pos = make_2d_positions(full.shape[1], device=device)
            damaged_float = damaged_mask.float().unsqueeze(-1)
            hidden_norm_scores = hidden[candidates].float().norm(dim=-1)
            ref_scores = {
                spec.label: paired.score_candidates(
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
                for spec in ref_specs
            }

            is_vision_sensitive = float(loss_full.item()) < float(loss_no.item())
            if sample_idx % 10 == 0:
                print(
                    f"sample={sample_idx} metric={eval_metric} "
                    f"full={loss_full.item():.6f} no={loss_no.item():.6f} "
                    f"vision_sensitive={is_vision_sensitive}"
                )

            for budget in budgets:
                selections = {
                    "random": paired.choose_random(candidates, budget),
                    "hidden_norm": paired.choose_by_scores(candidates, hidden_norm_scores, budget),
                    "oracle_single": oracle_ranked[: min(budget, oracle_ranked.numel())],
                }
                for spec in ref_specs:
                    selections[spec.label] = paired.choose_by_scores(
                        candidates,
                        ref_scores[spec.label],
                        budget,
                    )
                gate_scores = score_gate_candidates(
                    gate=gate,
                    query_scorer=query_scorer,
                    coverage_scorer=coverage_scorer,
                    gate_cfg=loaded_gate_cfg,
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
                )
                selections[args.gate_label] = paired.choose_by_scores(
                    candidates,
                    gate_scores,
                    budget,
                )
                losses = paired.compute_selection_losses(
                    wrapper=wrapper,
                    prepared=prepared,
                    corrupted=corrupted,
                    full=full,
                    selections=selections,
                    metric_name=eval_metric,
                    teacher_log_probs=teacher_log_probs,
                    batch_size=args.metric_batch_size,
                )
                paired.append_losses(stats_all, budget, loss_full, loss_no, losses)
                if is_vision_sensitive:
                    paired.append_losses(stats_vs, budget, loss_full, loss_no, losses)
                else:
                    paired.append_losses(stats_other, budget, loss_full, loss_no, losses)
        except Exception as exc:
            print(f"[WARN] budget gate eval sample failed: {exc}")
            continue

    summaries = {
        "all": paired.print_stats_block(
            title="All samples",
            stats=stats_all,
            budgets=budgets,
            method_names=method_names,
            delta_baselines=delta_baselines,
            metric_name=eval_metric,
            num_bootstrap=args.bootstrap,
            seed=wrapper_cfg["corruption"]["seed"],
        ),
        "vision_sensitive": paired.print_stats_block(
            title="Vision-sensitive samples: full_metric < no_metric",
            stats=stats_vs,
            budgets=budgets,
            method_names=method_names,
            delta_baselines=delta_baselines,
            metric_name=eval_metric,
            num_bootstrap=args.bootstrap,
            seed=wrapper_cfg["corruption"]["seed"] + 13,
        ),
        "other": paired.print_stats_block(
            title="Other samples: full_metric >= no_metric",
            stats=stats_other,
            budgets=budgets,
            method_names=method_names,
            delta_baselines=delta_baselines,
            metric_name=eval_metric,
            num_bootstrap=args.bootstrap,
            seed=wrapper_cfg["corruption"]["seed"] + 29,
        ),
    }
    if args.output_json:
        output = {
            "jsonl": args.jsonl,
            "max_samples": args.max_samples,
            "metric": eval_metric,
            "budgets": budgets,
            "gate_label": args.gate_label,
            "gate_config": args.gate_config,
            "gate_checkpoint": args.gate_checkpoint,
            "reference_scorers": [
                {
                    "label": spec.label,
                    "config": spec.config_path,
                    "checkpoint": spec.checkpoint_path,
                }
                for spec in ref_specs
            ],
            "delta_baselines": delta_baselines,
            "bootstrap": args.bootstrap,
            "summaries": summaries,
            "raw_stats": {
                "all": stats_all,
                "vision_sensitive": stats_vs,
                "other": stats_other,
            },
        }
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
