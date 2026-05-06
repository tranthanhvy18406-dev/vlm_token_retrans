import torch


def _zscore(values: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    values = values.float()
    return (values - values.mean()) / (values.std(unbiased=False) + eps)


def position_fourier_features(pos: torch.Tensor, bands: int = 0) -> torch.Tensor:
    """
    Build deterministic Fourier features from normalized [x, y] positions.
    pos: [M, 2], with values usually in [-1, 1].
    """
    if bands <= 0:
        return pos.new_empty((pos.size(0), 0))

    features = []
    for band in range(bands):
        scale = torch.pi * (2.0 ** band)
        x = pos[:, 0:1] * scale
        y = pos[:, 1:2] * scale
        features.extend([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)])
    return torch.cat(features, dim=-1)


def aux_dim_for_mode(mode: str, pos_fourier_bands: int = 0) -> int:
    if mode == "legacy":
        return 4
    if mode == "norm_stats":
        return 2 + 4 * int(pos_fourier_bands) + 3
    raise ValueError(f"Unsupported aux feature mode: {mode}")


def build_aux_features(
    hidden: torch.Tensor,
    pos: torch.Tensor,
    reliability: torch.Tensor,
    damaged_float: torch.Tensor,
    selection: torch.Tensor,
    mode: str = "legacy",
    pos_fourier_bands: int = 0,
) -> torch.Tensor:
    """
    Build per-candidate auxiliary features for the scorer.

    selection can be either a boolean mask over visual tokens or integer indices.
    For norm_stats mode, z-scoring is done within the current candidate set so
    the scalar magnitude features describe each damaged token relative to its peers.
    """
    h = hidden[selection].float()
    selected_pos = pos[selection].float()

    if mode == "legacy":
        return torch.cat(
            [
                selected_pos,
                reliability[selection].float(),
                damaged_float[selection].float(),
            ],
            dim=-1,
        )

    if mode == "norm_stats":
        raw_norm_z = _zscore(h.norm(dim=-1, keepdim=True))
        raw_mean_z = _zscore(h.mean(dim=-1, keepdim=True))
        raw_std_z = _zscore(h.std(dim=-1, keepdim=True, unbiased=False))
        return torch.cat(
            [
                selected_pos,
                position_fourier_features(selected_pos, bands=pos_fourier_bands),
                raw_norm_z,
                raw_mean_z,
                raw_std_z,
            ],
            dim=-1,
        )

    raise ValueError(f"Unsupported aux feature mode: {mode}")
