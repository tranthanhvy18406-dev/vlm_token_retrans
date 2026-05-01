import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.corruption import corrupt_image_features
from src.dataset import load_jsonl
from src.llava_wrapper import LlavaRetransWrapper
from src.utils import ensure_dir, get_torch_dtype, load_yaml, make_2d_positions, set_seed


def normalize_oracle_target(target: str) -> str:
    target = target.lower()
    if target in {"ce", "gt_ce", "ground_truth_ce"}:
        return "gt_ce"
    if target in {"teacher_kl", "clean_teacher_kl", "consistency_kl"}:
        return "teacher_kl"
    raise ValueError(f"Unsupported oracle target: {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["corruption"]["seed"])

    save_dir = cfg["oracle"]["save_dir"]
    ensure_dir(save_dir)

    wrapper = LlavaRetransWrapper(
        model_name=cfg["model"]["name"],
        device=cfg["model"]["device"],
        dtype=get_torch_dtype(cfg["model"]["dtype"]),
        cache_dir=cfg["model"].get("cache_dir"),
        local_files_only=bool(cfg["model"].get("local_files_only", False)),
        device_map=cfg["model"].get("device_map"),
    )

    if args.split == "train":
        jsonl_path = cfg["data"]["train_jsonl"]
    else:
        jsonl_path = cfg["data"].get("test_jsonl") or cfg["data"]["val_jsonl"]
    samples = load_jsonl(
        jsonl_path,
        image_root=cfg["data"]["image_root"],
        max_samples=args.max_samples,
    )

    max_candidates = cfg["oracle"]["max_candidates_per_sample"]
    chunk_size = cfg["oracle"]["candidate_chunk_size"]
    oracle_target = normalize_oracle_target(cfg["oracle"].get("target", "gt_ce"))
    layer_idx = cfg["model"]["layer_idx"]
    print(f"oracle target: {oracle_target}")

    for sample_idx, sample in enumerate(tqdm(samples, desc="building oracle cache")):
        out_path = os.path.join(save_dir, f"{sample_idx:06d}.pt")
        if os.path.exists(out_path):
            continue

        try:
            prepared = wrapper.prepare_inputs(sample.image, sample.question, sample.answer)
            full_features = prepared.image_features

            corrupted_features, damaged_mask, reliability = corrupt_image_features(
                full_features,
                drop_ratio=cfg["corruption"]["drop_ratio"],
                mode=cfg["corruption"]["mode"],
                mask_value=cfg["corruption"]["mask_value"],
            )

            teacher_log_probs = None
            if oracle_target == "gt_ce":
                base_metric = wrapper.compute_loss_from_image_features(prepared, corrupted_features)
                base_metric_name = "gt_ce"
            else:
                teacher_log_probs = wrapper.compute_teacher_log_probs_from_image_features(
                    prepared,
                    full_features,
                )
                base_metric = wrapper.compute_teacher_kl_from_image_features(
                    prepared=prepared,
                    teacher_log_probs=teacher_log_probs,
                    image_features=corrupted_features,
                )
                base_metric_name = "teacher_kl"

            visual_hidden = wrapper.get_layer_visual_hidden(
                prepared=prepared,
                image_features=corrupted_features,
                layer_idx=layer_idx,
            )

            damaged_indices = damaged_mask.nonzero(as_tuple=False).squeeze(-1)
            if damaged_indices.numel() > max_candidates:
                perm = torch.randperm(damaged_indices.numel(), device=damaged_indices.device)
                damaged_indices = damaged_indices[perm[:max_candidates]]

            num_tokens = full_features.shape[1]
            oracle_gain = torch.full((num_tokens,), float("nan"), device=full_features.device)

            for start in range(0, damaged_indices.numel(), chunk_size):
                cand = damaged_indices[start : start + chunk_size]
                count = cand.numel()
                if count == 0:
                    continue

                embeds = wrapper.build_candidate_restored_embeds(
                    prepared=prepared,
                    corrupted_features=corrupted_features,
                    full_features=full_features,
                    candidate_indices=cand,
                )

                if oracle_target == "gt_ce":
                    candidate_metrics = wrapper.compute_loss_batch_from_embeds(
                        embeds=embeds,
                        attention_mask=prepared.attention_mask.repeat(count, 1),
                        labels=prepared.labels.repeat(count, 1),
                    )
                else:
                    candidate_metrics = wrapper.compute_teacher_kl_batch_from_embeds(
                        embeds=embeds,
                        attention_mask=prepared.attention_mask.repeat(count, 1),
                        labels=prepared.labels.repeat(count, 1),
                        teacher_log_probs=teacher_log_probs,
                    )

                gains = (base_metric - candidate_metrics).to(dtype=oracle_gain.dtype)
                oracle_gain[cand] = gains

            pos = make_2d_positions(num_tokens, device=full_features.device)
            damaged_float = damaged_mask.float().unsqueeze(-1)

            cache = {
                "sample_idx": sample_idx,
                "image": sample.image,
                "question": sample.question,
                "answer": sample.answer,
                "layer_idx": layer_idx,
                "drop_ratio": cfg["corruption"]["drop_ratio"],
                "oracle_target": oracle_target,
                "base_metric_name": base_metric_name,
                "hidden": visual_hidden.detach().cpu().to(torch.float16),
                "damaged_mask": damaged_mask.detach().cpu(),
                "reliability": reliability.detach().cpu().to(torch.float16),
                "damaged_float": damaged_float.detach().cpu().to(torch.float16),
                "pos": pos.detach().cpu().to(torch.float16),
                "oracle_gain": oracle_gain.detach().cpu().to(torch.float32),
                "base_metric": float(base_metric.item()),
            }

            torch.save(cache, out_path)

        except Exception as exc:
            print(f"[WARN] sample {sample_idx} failed: {exc}")
            continue


if __name__ == "__main__":
    main()
