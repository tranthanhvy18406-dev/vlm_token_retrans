import os
import random

import numpy as np
import torch
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_dtype(name: str):
    name = name.lower()
    if name in ["fp16", "float16"]:
        return torch.float16
    if name in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if name in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def make_2d_positions(num_tokens: int, device=None) -> torch.Tensor:
    """
    Return normalized 2D coordinates: [N, 2].
    Assumes visual tokens form a square grid, e.g. 576 = 24 x 24.
    """
    side = int(num_tokens ** 0.5)
    if side * side != num_tokens:
        raise ValueError(f"num_tokens={num_tokens} is not a square number.")

    ys, xs = torch.meshgrid(
        torch.arange(side, device=device),
        torch.arange(side, device=device),
        indexing="ij",
    )
    xs = xs.reshape(-1).float()
    ys = ys.reshape(-1).float()

    xs = 2.0 * xs / max(side - 1, 1) - 1.0
    ys = 2.0 * ys / max(side - 1, 1) - 1.0

    return torch.stack([xs, ys], dim=-1)


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))
