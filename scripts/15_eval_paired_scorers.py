import argparse
import json
import os
import sys
from dataclasses import dataclass

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.corruption import corrupt_image_features
from src.dataset import load_jsonl
from src.features import aux_dim_for_mode, build_aux_features
from src.llava_wrapper import LlavaRetransWrapper
from src.metrics import recovery_ratio
from src.scorer import build_scorer_from_config, scorer_requires_query
from src.utils import get_torch_dtype, load_yaml, make_2d_positions, mean_or_nan, set_seed


@dataclass
class ScorerSpec:
    label: str
    config_path: str
    checkpoint_path: str
    config: dict
    model: torch.nn.Module


def normalize_metric_name(name: str) -> str:
    name = name.lower()
    if name in {"ce", "gt_ce", "ground_truth_ce"}:
        return "gt_ce"
    if name in {"teacher_kl", "clean_teacher_kl", "consistency_kl"}:
        return "teacher_kl"
    raise ValueError(f"Unsupported eval/oracle metric: {name}")


def parse_scorer_arg(value: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            "--scorer must have format label,config_path,checkpoint_path; "
            f"got {value!r}"
        )
    return parts[0], parts[1], parts[2]


def parse_label_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def compute_metric_from_image_features(
    wrapper: LlavaRetransWrapper,
    prepared,
    image_features: torch.Tensor,
    metric_name: str,
    teacher_log_probs: torch.Tensor | None,
) -> torch.Tensor:
    if metric_name == "gt_ce":
        return wrapper.compute_loss_from_image_features(prepared, image_features)
    return wrapper.compute_teacher_kl_from_image_features(
        prepared=prepared,
        teacher_log_probs=teacher_log_probs,
        image_features=image_features,
    )


def compute_metric_batch_from_embeds(
    wrapper: LlavaRetransWrapper,
    prepared,
    embeds: torch.Tensor,
    metric_name: str,
    teacher_log_probs: torch.Tensor | None,
) -> torch.Tensor:
    count = embeds.size(0)
    attention_mask = prepared.attention_mask.repeat(count, 1)
    labels = prepared.labels.repeat(count, 1)
    if metric_name == "gt_ce":
        return wrapper.compute_loss_batch_from_embeds(
            embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
    return wrapper.compute_teacher_kl_batch_from_embeds(
        embeds=embeds,
        attention_mask=attention_mask,
        labels=labels,
        teacher_log_probs=teacher_log_probs,
    )


def build_restored_set_embeds(
    prepared,
    corrupted_features: torch.Tensor,
    full_features: torch.Tensor,
    selected_sets: list[torch.Tensor],
) -> torch.Tensor:
    count = len(selected_sets)
    base_features = corrupted_features.repeat(count, 1, 1)
    for row, indices in enumerate(selected_sets):
        if indices.numel() == 0:
            continue
        indices = indices.to(base_features.device)
        base_features[row, indices, :] = full_features[0, indices, :]

    base_embeds = prepared.inputs_embeds_full.repeat(count, 1, 1)
    base_embeds[:, prepared.image_positions, :] = base_features
    return base_embeds


def compute_selection_losses(
    wrapper: LlavaRetransWrapper,
    prepared,
    corrupted: torch.Tensor,
    full: torch.Tensor,
    selections: dict[str, torch.Tensor],
    metric_name: str,
    teacher_log_probs: torch.Tensor | None,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    names = list(selections.keys())
    losses: dict[str, torch.Tensor] = {}
    for start in range(0, len(names), batch_size):
        chunk_names = names[start : start + batch_size]
        embeds = build_restored_set_embeds(
            prepared=prepared,
            corrupted_features=corrupted,
            full_features=full,
            selected_sets=[selections[name] for name in chunk_names],
        )
        chunk_losses = compute_metric_batch_from_embeds(
            wrapper=wrapper,
            prepared=prepared,
            embeds=embeds,
            metric_name=metric_name,
            teacher_log_probs=teacher_log_probs,
        )
        for name, value in zip(chunk_names, chunk_losses):
            losses[name] = value.detach()
    return losses


def rank_oracle(
    wrapper: LlavaRetransWrapper,
    prepared,
    corrupted: torch.Tensor,
    full: torch.Tensor,
    candidates: torch.Tensor,
    metric_name: str,
    teacher_log_probs: torch.Tensor | None,
    chunk_size: int,
) -> torch.Tensor:
    base_metric = compute_metric_from_image_features(
        wrapper=wrapper,
        prepared=prepared,
        image_features=corrupted,
        metric_name=metric_name,
        teacher_log_probs=teacher_log_probs,
    )
    gains = []
    all_cands = []
    for start in range(0, candidates.numel(), chunk_size):
        cand = candidates[start : start + chunk_size]
        count = cand.numel()
        if count == 0:
            continue
        embeds = wrapper.build_candidate_restored_embeds(
            prepared=prepared,
            corrupted_features=corrupted,
            full_features=full,
            candidate_indices=cand,
        )
        candidate_metrics = compute_metric_batch_from_embeds(
            wrapper=wrapper,
            prepared=prepared,
            embeds=embeds,
            metric_name=metric_name,
            teacher_log_probs=teacher_log_probs,
        )
        gains.append(base_metric - candidate_metrics)
        all_cands.append(cand)

    gains = torch.cat(gains, dim=0)
    all_cands = torch.cat(all_cands, dim=0)
    order = torch.argsort(gains, descending=True)
    return all_cands[order]


def choose_random(candidates: torch.Tensor, k: int) -> torch.Tensor:
    if candidates.numel() <= k:
        return candidates
    perm = torch.randperm(candidates.numel(), device=candidates.device)
    return candidates[perm[:k]]


def choose_by_scores(candidates: torch.Tensor, scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), candidates.numel())
    return candidates[torch.topk(scores, k=k).indices]


def build_scorer_aux(
    hidden: torch.Tensor,
    pos: torch.Tensor,
    reliability: torch.Tensor,
    damaged_float: torch.Tensor,
    candidates: torch.Tensor,
    scorer_cfg: dict,
    attn_q_to_vis: torch.Tensor | None,
    attn_text_to_vis_mean: torch.Tensor | None,
) -> torch.Tensor:
    aux_mode = scorer_cfg.get("aux_mode", "legacy")
    pos_fourier_bands = int(scorer_cfg.get("pos_fourier_bands", 0))
    attention_layers = scorer_cfg.get("attention_layers", [])
    aux = build_aux_features(
        hidden=hidden,
        pos=pos,
        reliability=reliability,
        damaged_float=damaged_float,
        selection=candidates,
        mode=aux_mode,
        pos_fourier_bands=pos_fourier_bands,
        attn_q_to_vis=attn_q_to_vis,
        attn_text_to_vis_mean=attn_text_to_vis_mean,
    )
    expected_aux_dim = int(scorer_cfg["aux_dim"])
    actual_aux_dim = aux.size(-1)
    inferred_aux_dim = aux_dim_for_mode(aux_mode, pos_fourier_bands, attention_layers)
    if actual_aux_dim != expected_aux_dim or inferred_aux_dim != expected_aux_dim:
        raise RuntimeError(
            "Aux dimension mismatch: "
            f"config={expected_aux_dim}, actual={actual_aux_dim}, inferred={inferred_aux_dim}. "
            f"mode={aux_mode}, pos_fourier_bands={pos_fourier_bands}"
        )
    return aux


def score_candidates(
    spec: ScorerSpec,
    hidden: torch.Tensor,
    pos: torch.Tensor,
    reliability: torch.Tensor,
    damaged_float: torch.Tensor,
    candidates: torch.Tensor,
    query_hidden: torch.Tensor | None,
    attn_q_to_vis: torch.Tensor | None,
    attn_text_to_vis_mean: torch.Tensor | None,
) -> torch.Tensor:
    scorer_cfg = spec.config["scorer"]
    aux = build_scorer_aux(
        hidden=hidden,
        pos=pos,
        reliability=reliability,
        damaged_float=damaged_float,
        candidates=candidates,
        scorer_cfg=scorer_cfg,
        attn_q_to_vis=attn_q_to_vis,
        attn_text_to_vis_mean=attn_text_to_vis_mean,
    )
    with torch.no_grad():
        return spec.model(hidden[candidates].float(), aux, q=query_hidden).float()


def make_stats(budgets: list[int], method_names: list[str]) -> dict[int, dict[str, list[float]]]:
    stats = {}
    for k in budgets:
        stats[k] = {"full": [], "no": []}
        for name in method_names:
            stats[k][name] = []
    return stats


def append_losses(
    stats: dict[int, dict[str, list[float]]],
    k: int,
    loss_full: torch.Tensor,
    loss_no: torch.Tensor,
    losses: dict[str, torch.Tensor],
) -> None:
    stats[k]["full"].append(float(loss_full.item()))
    stats[k]["no"].append(float(loss_no.item()))
    for name, value in losses.items():
        stats[k][name].append(float(value.item()))


def sample_recovery_values(stats_for_k: dict[str, list[float]], method: str) -> list[float]:
    values = []
    for loss_full, loss_no, loss_method in zip(
        stats_for_k["full"],
        stats_for_k["no"],
        stats_for_k[method],
    ):
        values.append(recovery_ratio(loss_no, loss_method, loss_full))
    return values


def aggregate_recovery(
    stats_for_k: dict[str, list[float]],
    method: str,
    indices: torch.Tensor | None = None,
) -> float:
    full = torch.tensor(stats_for_k["full"], dtype=torch.float32)
    no = torch.tensor(stats_for_k["no"], dtype=torch.float32)
    method_values = torch.tensor(stats_for_k[method], dtype=torch.float32)
    if indices is not None:
        full = full[indices]
        no = no[indices]
        method_values = method_values[indices]
    if full.numel() == 0:
        return float("nan")
    return recovery_ratio(
        float(no.mean().item()),
        float(method_values.mean().item()),
        float(full.mean().item()),
    )


def bootstrap_aggregate_recovery_ci(
    stats_for_k: dict[str, list[float]],
    method: str,
    num_bootstrap: int,
    seed: int,
    scale: float = 1.0,
) -> tuple[float, float]:
    n = len(stats_for_k["full"])
    if n == 0:
        return float("nan"), float("nan")
    if n == 1 or num_bootstrap <= 0:
        value = scale * aggregate_recovery(stats_for_k, method)
        return value, value

    generator = torch.Generator().manual_seed(seed)
    means = []
    for _ in range(num_bootstrap):
        idx = torch.randint(0, n, (n,), generator=generator)
        means.append(torch.tensor(aggregate_recovery(stats_for_k, method, idx)))
    boot = torch.stack(means)
    lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975]))
    return scale * float(lo.item()), scale * float(hi.item())


def bootstrap_aggregate_delta_ci(
    stats_for_k: dict[str, list[float]],
    method: str,
    baseline: str,
    num_bootstrap: int,
    seed: int,
    scale: float = 1.0,
) -> tuple[float, float]:
    n = len(stats_for_k["full"])
    if n == 0:
        return float("nan"), float("nan")
    if n == 1 or num_bootstrap <= 0:
        value = aggregate_recovery(stats_for_k, method) - aggregate_recovery(
            stats_for_k,
            baseline,
        )
        value *= scale
        return value, value

    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(num_bootstrap):
        idx = torch.randint(0, n, (n,), generator=generator)
        delta = aggregate_recovery(stats_for_k, method, idx) - aggregate_recovery(
            stats_for_k,
            baseline,
            idx,
        )
        values.append(torch.tensor(delta))
    boot = torch.stack(values)
    lo, hi = torch.quantile(boot, torch.tensor([0.025, 0.975]))
    return scale * float(lo.item()), scale * float(hi.item())


def stats_to_summary(
    stats: dict[int, dict[str, list[float]]],
    budgets: list[int],
    method_names: list[str],
    delta_baselines: list[str],
    num_bootstrap: int,
    seed: int,
) -> dict[str, dict]:
    summary = {}
    for k in budgets:
        budget_summary = {
            "full_loss": mean_or_nan(stats[k]["full"]),
            "no_loss": mean_or_nan(stats[k]["no"]),
            "n": len(stats[k]["full"]),
            "methods": {},
            "deltas": {},
        }
        recoveries = {}
        for name in method_names:
            if not stats[k].get(name):
                continue
            rec = aggregate_recovery(stats[k], name)
            sample_rec = mean_or_nan(sample_recovery_values(stats[k], name))
            recoveries[name] = rec
            lo, hi = bootstrap_aggregate_recovery_ci(
                stats[k],
                name,
                num_bootstrap=num_bootstrap,
                seed=seed + int(k) * 1000 + len(name),
                scale=100.0,
            )
            budget_summary["methods"][name] = {
                "loss": mean_or_nan(stats[k][name]),
                "recovery": 100.0 * rec,
                "sample_mean_recovery": 100.0 * sample_rec,
                "recovery_ci95": [lo, hi],
                "n": len(stats[k][name]),
            }

        for baseline in delta_baselines:
            if baseline not in recoveries:
                continue
            budget_summary["deltas"][baseline] = {}
            base_recovery = recoveries[baseline]
            for name, rec in recoveries.items():
                if name == baseline:
                    continue
                lo, hi = bootstrap_aggregate_delta_ci(
                    stats[k],
                    method=name,
                    baseline=baseline,
                    num_bootstrap=num_bootstrap,
                    seed=seed + int(k) * 2000 + len(name) + len(baseline),
                    scale=100.0,
                )
                budget_summary["deltas"][baseline][name] = {
                    "delta_recovery": 100.0 * (rec - base_recovery),
                    "delta_ci95": [lo, hi],
                }
        summary[str(k)] = budget_summary
    return summary


def print_stats_block(
    title: str,
    stats: dict[int, dict[str, list[float]]],
    budgets: list[int],
    method_names: list[str],
    delta_baselines: list[str],
    metric_name: str,
    num_bootstrap: int,
    seed: int,
) -> dict[str, dict]:
    sample_count = len(stats[budgets[0]]["full"]) if budgets else 0
    metric_label = "teacher KL" if metric_name == "teacher_kl" else "GT CE loss"
    print(
        f"\n===== {title} (n={sample_count}), lower {metric_label} is better; "
        "recovery uses official mean-loss aggregation ====="
    )
    summary = stats_to_summary(
        stats=stats,
        budgets=budgets,
        method_names=method_names,
        delta_baselines=delta_baselines,
        num_bootstrap=num_bootstrap,
        seed=seed,
    )
    if sample_count == 0:
        print("No samples.")
        return summary

    for k in budgets:
        print(f"\nK={k}")
        budget = summary[str(k)]
        print(f"{'full':18s}: {budget['full_loss']:.6f}")
        print(f"{'no':18s}: {budget['no_loss']:.6f}")
        for name in method_names:
            method = budget["methods"].get(name)
            if method is None:
                continue
            lo, hi = method["recovery_ci95"]
            print(
                f"{name:18s}: loss={method['loss']:.6f} "
                f"recovery={method['recovery']:.2f}% "
                f"ci95=[{lo:.2f}, {hi:.2f}]"
            )

        for baseline in delta_baselines:
            if baseline not in budget["deltas"]:
                continue
            print(f"Paired delta vs {baseline}:")
            for name, delta in budget["deltas"][baseline].items():
                lo, hi = delta["delta_ci95"]
                print(
                    f"{name:18s}: {delta['delta_recovery']:+.2f} "
                    f"ci95=[{lo:+.2f}, {hi:+.2f}]"
                )
    return summary


def load_scorer_specs(raw_specs: list[str], device: str) -> list[ScorerSpec]:
    specs = []
    labels = set()
    for raw in raw_specs:
        label, config_path, checkpoint_path = parse_scorer_arg(raw)
        if label in labels:
            raise ValueError(f"Duplicate scorer label: {label}")
        labels.add(label)

        cfg = load_yaml(config_path)
        model = build_scorer_from_config(cfg, dropout=0.0).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()
        specs.append(
            ScorerSpec(
                label=label,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                config=cfg,
                model=model,
            )
        )
    return specs


def choose_wrapper_config(configs: list[dict]) -> dict:
    cfg = configs[0]
    use_eager = any(
        item["model"].get("attn_implementation") == "eager"
        or item["scorer"].get("aux_mode", "legacy") == "norm_attn"
        for item in configs
    )
    cfg = json.loads(json.dumps(cfg))
    if use_eager:
        cfg["model"]["attn_implementation"] = "eager"
    return cfg


def validate_configs(configs: list[dict]) -> None:
    first = configs[0]
    first_model = first["model"]["name"]
    first_layer = first["model"]["layer_idx"]
    first_budgets = first["eval"]["budgets"]
    first_corruption = first["corruption"]
    norm_attn_layers = None
    for cfg in configs:
        if cfg["model"]["name"] != first_model:
            raise RuntimeError("All scorers must use the same model.")
        if cfg["model"]["layer_idx"] != first_layer:
            raise RuntimeError("All scorers must use the same layer_idx.")
        if cfg["eval"]["budgets"] != first_budgets:
            raise RuntimeError("All scorers must use the same eval budgets.")
        if cfg["corruption"] != first_corruption:
            raise RuntimeError("All scorers must use the same corruption config.")
        if cfg["scorer"].get("aux_mode", "legacy") == "norm_attn":
            layers = tuple(cfg["scorer"].get("attention_layers", [2, 4, 8, 12]))
            if norm_attn_layers is None:
                norm_attn_layers = layers
            elif norm_attn_layers != layers:
                raise RuntimeError("All norm_attn scorers must use the same attention_layers.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scorer",
        action="append",
        required=True,
        help="Repeatable: label,config_path,checkpoint_path",
    )
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--metric_batch_size", type=int, default=8)
    parser.add_argument("--oracle_chunk_size", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--delta_baselines", default="")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    raw_cfgs = [load_yaml(parse_scorer_arg(raw)[1]) for raw in args.scorer]
    validate_configs(raw_cfgs)
    wrapper_cfg = choose_wrapper_config(raw_cfgs)
    set_seed(wrapper_cfg["corruption"]["seed"])

    device = wrapper_cfg["model"]["device"]
    specs = load_scorer_specs(args.scorer, device=device)
    budgets = [int(k) for k in wrapper_cfg["eval"]["budgets"]]
    eval_metric = normalize_metric_name(
        wrapper_cfg.get("eval", {}).get("metric", wrapper_cfg["oracle"].get("target", "gt_ce"))
    )
    oracle_chunk_size = int(
        args.oracle_chunk_size or wrapper_cfg["oracle"].get("candidate_chunk_size", 16)
    )
    method_names = ["random", "hidden_norm", "oracle_single"] + [spec.label for spec in specs]
    delta_baselines = parse_label_list(args.delta_baselines)

    print(f"eval jsonl: {args.jsonl}")
    print(f"max_samples: {args.max_samples}")
    print(f"eval metric: {eval_metric}")
    print(f"budgets: {budgets}")
    print(f"scorers: {[spec.label for spec in specs]}")
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
    needs_query = any(scorer_requires_query(spec.config["scorer"]) for spec in specs)
    needs_attention = any(
        spec.config["scorer"].get("aux_mode", "legacy") == "norm_attn" for spec in specs
    )
    attention_layers = None
    if needs_attention:
        for spec in specs:
            if spec.config["scorer"].get("aux_mode", "legacy") == "norm_attn":
                attention_layers = spec.config["scorer"].get("attention_layers", [2, 4, 8, 12])
                break

    stats_all = make_stats(budgets, method_names)
    stats_vs = make_stats(budgets, method_names)
    stats_other = make_stats(budgets, method_names)

    for sample_idx, sample in enumerate(tqdm(samples, desc="paired scorer eval")):
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

            if needs_query:
                hidden, query_hidden = wrapper.get_layer_visual_and_query_hidden(
                    prepared=prepared,
                    image_features=corrupted,
                    layer_idx=layer_idx,
                )
            else:
                hidden = wrapper.get_layer_visual_hidden(
                    prepared=prepared,
                    image_features=corrupted,
                    layer_idx=layer_idx,
                )
                query_hidden = None

            attn_q_to_vis = None
            attn_text_to_vis_mean = None
            if needs_attention:
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
            loss_full = compute_metric_from_image_features(
                wrapper=wrapper,
                prepared=prepared,
                image_features=full,
                metric_name=eval_metric,
                teacher_log_probs=teacher_log_probs,
            )
            loss_no = compute_metric_from_image_features(
                wrapper=wrapper,
                prepared=prepared,
                image_features=corrupted,
                metric_name=eval_metric,
                teacher_log_probs=teacher_log_probs,
            )
            oracle_ranked = rank_oracle(
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
            scorer_scores = {
                spec.label: score_candidates(
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
                for spec in specs
            }

            is_vision_sensitive = float(loss_full.item()) < float(loss_no.item())
            if sample_idx % 10 == 0:
                print(
                    f"sample={sample_idx} metric={eval_metric} "
                    f"full={loss_full.item():.6f} no={loss_no.item():.6f} "
                    f"vision_sensitive={is_vision_sensitive}"
                )

            for k in budgets:
                selections = {
                    "random": choose_random(candidates, k),
                    "hidden_norm": choose_by_scores(candidates, hidden_norm_scores, k),
                    "oracle_single": oracle_ranked[: min(k, oracle_ranked.numel())],
                }
                for spec in specs:
                    selections[spec.label] = choose_by_scores(
                        candidates,
                        scorer_scores[spec.label],
                        k,
                    )
                losses = compute_selection_losses(
                    wrapper=wrapper,
                    prepared=prepared,
                    corrupted=corrupted,
                    full=full,
                    selections=selections,
                    metric_name=eval_metric,
                    teacher_log_probs=teacher_log_probs,
                    batch_size=args.metric_batch_size,
                )
                append_losses(stats_all, k, loss_full, loss_no, losses)
                if is_vision_sensitive:
                    append_losses(stats_vs, k, loss_full, loss_no, losses)
                else:
                    append_losses(stats_other, k, loss_full, loss_no, losses)

        except Exception as exc:
            print(f"[WARN] paired eval sample failed: {exc}")
            continue

    summaries = {
        "all": print_stats_block(
            title="All samples",
            stats=stats_all,
            budgets=budgets,
            method_names=method_names,
            delta_baselines=delta_baselines,
            metric_name=eval_metric,
            num_bootstrap=args.bootstrap,
            seed=wrapper_cfg["corruption"]["seed"],
        ),
        "vision_sensitive": print_stats_block(
            title="Vision-sensitive samples: full_metric < no_metric",
            stats=stats_vs,
            budgets=budgets,
            method_names=method_names,
            delta_baselines=delta_baselines,
            metric_name=eval_metric,
            num_bootstrap=args.bootstrap,
            seed=wrapper_cfg["corruption"]["seed"] + 13,
        ),
        "other": print_stats_block(
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
            "scorers": [
                {
                    "label": spec.label,
                    "config": spec.config_path,
                    "checkpoint": spec.checkpoint_path,
                }
                for spec in specs
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
