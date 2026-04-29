import argparse
import glob
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.losses import listwise_kl_loss, pairwise_ranking_loss
from src.metrics import ndcg_at_k, recall_at_k
from src.scorer import MLPRetransScorer
from src.utils import ensure_dir, load_yaml, set_seed


def load_cache_paths(cache_dir: str):
    paths = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    if not paths:
        raise RuntimeError(f"No cache files found in {cache_dir}")
    return paths


def build_candidate_tensors(cache, device):
    hidden = cache["hidden"].to(device=device, dtype=torch.float32)
    pos = cache["pos"].to(device=device, dtype=torch.float32)
    reliability = cache["reliability"].to(device=device, dtype=torch.float32)
    damaged_float = cache["damaged_float"].to(device=device, dtype=torch.float32)
    damaged_mask = cache["damaged_mask"].to(device=device)
    oracle_gain = cache["oracle_gain"].to(device=device, dtype=torch.float32)

    valid = damaged_mask & torch.isfinite(oracle_gain)

    h = hidden[valid]
    aux = torch.cat(
        [
            pos[valid],
            reliability[valid],
            damaged_float[valid],
        ],
        dim=-1,
    )
    gains = oracle_gain[valid]

    return h, aux, gains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["corruption"]["seed"])

    device = cfg["model"]["device"]
    cache_dir = cfg["oracle"]["save_dir"]
    paths = load_cache_paths(cache_dir)

    scorer = MLPRetransScorer(
        hidden_dim=cfg["scorer"]["hidden_dim"],
        aux_dim=cfg["scorer"]["aux_dim"],
        hidden1=cfg["scorer"]["hidden1"],
        hidden2=cfg["scorer"]["hidden2"],
        dropout=cfg["scorer"]["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        scorer.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    epochs = int(cfg["train"]["epochs"])
    pairs_per_sample = int(cfg["train"]["pairs_per_sample"])
    list_w = float(cfg["train"]["list_loss_weight"])

    ckpt_path = cfg["train"]["checkpoint_path"]
    ensure_dir(os.path.dirname(ckpt_path))

    for epoch in range(epochs):
        scorer.train()
        total_loss = 0.0
        used = 0

        perm = torch.randperm(len(paths)).tolist()

        for idx in tqdm(perm, desc=f"epoch {epoch}"):
            cache = torch.load(paths[idx], map_location="cpu")
            h, aux, gains = build_candidate_tensors(cache, device)

            if h.size(0) < 4:
                continue

            scores = scorer(h, aux)

            loss_rank = pairwise_ranking_loss(
                scores=scores,
                gains=gains,
                pairs_per_sample=pairs_per_sample,
            )

            loss_list = listwise_kl_loss(scores=scores, gains=gains)
            loss = loss_rank + list_w * loss_list

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if cfg["train"]["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    scorer.parameters(),
                    max_norm=float(cfg["train"]["grad_clip"]),
                )

            optimizer.step()

            total_loss += float(loss.item())
            used += 1

        scorer.eval()
        recalls = []
        ndcgs = []

        with torch.no_grad():
            for path in paths[: min(100, len(paths))]:
                cache = torch.load(path, map_location="cpu")
                h, aux, gains = build_candidate_tensors(cache, device)
                if h.size(0) < 4:
                    continue

                scores = scorer(h, aux)
                recalls.append(recall_at_k(scores, gains, k=min(32, h.size(0))))
                ndcgs.append(ndcg_at_k(scores, gains, k=min(32, h.size(0))))

        mean_loss = total_loss / max(used, 1)
        mean_recall = sum(recalls) / max(len(recalls), 1)
        mean_ndcg = sum(ndcgs) / max(len(ndcgs), 1)

        print(
            f"epoch={epoch} "
            f"loss={mean_loss:.6f} "
            f"recall@32={mean_recall:.4f} "
            f"ndcg@32={mean_ndcg:.4f}"
        )

        torch.save(
            {
                "model": scorer.state_dict(),
                "config": cfg,
                "epoch": epoch,
            },
            ckpt_path,
        )

    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
