import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.corruption import corrupt_image_features
from src.dataset import load_jsonl
from src.llava_wrapper import LlavaRetransWrapper
from src.metrics import recovery_ratio
from src.scorer import MLPRetransScorer
from src.utils import get_torch_dtype, load_yaml, make_2d_positions, mean_or_nan, set_seed


def restore_by_indices(corrupted, full, indices):
    restored = corrupted.clone()
    restored[:, indices, :] = full[:, indices, :]
    return restored


def choose_random(candidates, k):
    if candidates.numel() <= k:
        return candidates
    perm = torch.randperm(candidates.numel(), device=candidates.device)
    return candidates[perm[:k]]


def choose_hidden_norm(hidden, candidates, k):
    scores = hidden[candidates].float().norm(dim=-1)
    k = min(k, candidates.numel())
    return candidates[torch.topk(scores, k=k).indices]


def choose_mlp(scorer, hidden, pos, reliability, damaged_float, candidates, k):
    h = hidden[candidates].float()
    aux = torch.cat(
        [
            pos[candidates].float(),
            reliability[candidates].float(),
            damaged_float[candidates].float(),
        ],
        dim=-1,
    )
    scores = scorer(h, aux)
    k = min(k, candidates.numel())
    return candidates[torch.topk(scores, k=k).indices]


def rank_oracle(wrapper, prepared, corrupted, full, candidates, chunk_size=16):
    base_loss = wrapper.compute_loss_from_image_features(prepared, corrupted)

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

        losses = wrapper.compute_loss_batch_from_embeds(
            embeds=embeds,
            attention_mask=prepared.attention_mask.repeat(count, 1),
            labels=prepared.labels.repeat(count, 1),
        )

        gains.append(base_loss - losses)
        all_cands.append(cand)

    gains = torch.cat(gains, dim=0)
    all_cands = torch.cat(all_cands, dim=0)
    order = torch.argsort(gains, descending=True)
    return all_cands[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["corruption"]["seed"])

    device = cfg["model"]["device"]

    wrapper = LlavaRetransWrapper(
        model_name=cfg["model"]["name"],
        device=device,
        dtype=get_torch_dtype(cfg["model"]["dtype"]),
        cache_dir=cfg["model"].get("cache_dir"),
        local_files_only=bool(cfg["model"].get("local_files_only", False)),
        device_map=cfg["model"].get("device_map"),
    )

    scorer = MLPRetransScorer(
        hidden_dim=cfg["scorer"]["hidden_dim"],
        aux_dim=cfg["scorer"]["aux_dim"],
        hidden1=cfg["scorer"]["hidden1"],
        hidden2=cfg["scorer"]["hidden2"],
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    scorer.load_state_dict(ckpt["model"])
    scorer.eval()

    samples = load_jsonl(
        cfg["data"]["val_jsonl"],
        image_root=cfg["data"]["image_root"],
        max_samples=args.max_samples,
    )

    budgets = cfg["eval"]["budgets"]
    layer_idx = cfg["model"]["layer_idx"]

    stats = {
        k: {
            "full": [],
            "no": [],
            "random": [],
            "hidden_norm": [],
            "mlp": [],
            "oracle": [],
        }
        for k in budgets
    }

    for sample in tqdm(samples, desc="eval retrans loss"):
        try:
            prepared = wrapper.prepare_inputs(sample.image, sample.question, sample.answer)
            full = prepared.image_features

            corrupted, damaged_mask, reliability = corrupt_image_features(
                full,
                drop_ratio=cfg["corruption"]["drop_ratio"],
                mode=cfg["corruption"]["mode"],
                mask_value=cfg["corruption"]["mask_value"],
            )

            candidates = damaged_mask.nonzero(as_tuple=False).squeeze(-1)
            if candidates.numel() == 0:
                continue

            hidden = wrapper.get_layer_visual_hidden(
                prepared=prepared,
                image_features=corrupted,
                layer_idx=layer_idx,
            )

            num_tokens = full.shape[1]
            pos = make_2d_positions(num_tokens, device=device)
            damaged_float = damaged_mask.float().unsqueeze(-1)

            loss_full = wrapper.compute_loss_from_image_features(prepared, full)
            loss_no = wrapper.compute_loss_from_image_features(prepared, corrupted)
            oracle_ranked = rank_oracle(
                wrapper=wrapper,
                prepared=prepared,
                corrupted=corrupted,
                full=full,
                candidates=candidates,
                chunk_size=cfg["oracle"]["candidate_chunk_size"],
            )

            for k in budgets:
                idx_random = choose_random(candidates, k)
                idx_hidden = choose_hidden_norm(hidden, candidates, k)
                idx_mlp = choose_mlp(
                    scorer=scorer,
                    hidden=hidden,
                    pos=pos,
                    reliability=reliability,
                    damaged_float=damaged_float,
                    candidates=candidates,
                    k=k,
                )
                idx_oracle = oracle_ranked[: min(k, oracle_ranked.numel())]

                losses = {
                    "random": wrapper.compute_loss_from_image_features(
                        prepared,
                        restore_by_indices(corrupted, full, idx_random),
                    ),
                    "hidden_norm": wrapper.compute_loss_from_image_features(
                        prepared,
                        restore_by_indices(corrupted, full, idx_hidden),
                    ),
                    "mlp": wrapper.compute_loss_from_image_features(
                        prepared,
                        restore_by_indices(corrupted, full, idx_mlp),
                    ),
                    "oracle": wrapper.compute_loss_from_image_features(
                        prepared,
                        restore_by_indices(corrupted, full, idx_oracle),
                    ),
                }

                stats[k]["full"].append(float(loss_full.item()))
                stats[k]["no"].append(float(loss_no.item()))
                for name, value in losses.items():
                    stats[k][name].append(float(value.item()))

        except Exception as exc:
            print(f"[WARN] eval sample failed: {exc}")
            continue

    print("\n===== Eval results, lower loss is better =====")
    for k in budgets:
        print(f"\nK={k}")
        full_mean = mean_or_nan(stats[k]["full"])
        no_mean = mean_or_nan(stats[k]["no"])
        for name, values in stats[k].items():
            if values:
                mean_val = mean_or_nan(values)
                print(f"{name:12s}: {mean_val:.6f}")
        print("Recovery ratios:")
        for name in ["random", "hidden_norm", "mlp", "oracle"]:
            method_mean = mean_or_nan(stats[k][name])
            recovery = recovery_ratio(no_mean, method_mean, full_mean)
            print(f"{name:12s}: {100.0 * recovery:.2f}%")


if __name__ == "__main__":
    main()
