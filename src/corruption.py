import torch


def corrupt_image_features(
    image_features: torch.Tensor,
    drop_ratio: float,
    mode: str = "erasure",
    mask_value: str = "zero",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    image_features: [1, N, D]
    return:
        corrupted_features: [1, N, D]
        damaged_mask: [N], bool
        reliability: [N, 1], 1 means reliable, 0 means damaged
    """
    if image_features.dim() != 3 or image_features.size(0) != 1:
        raise ValueError("This stage-1 prototype assumes batch size = 1.")

    _, num_tokens, _ = image_features.shape
    device = image_features.device

    damaged_mask = torch.rand(num_tokens, device=device) < drop_ratio
    corrupted = image_features.clone()

    if mode == "erasure":
        if mask_value == "zero":
            corrupted[:, damaged_mask, :] = 0.0
        else:
            raise ValueError(f"Unsupported mask_value: {mask_value}")
    else:
        raise ValueError(f"Unsupported corruption mode: {mode}")

    reliability = (~damaged_mask).float().unsqueeze(-1)
    return corrupted, damaged_mask, reliability
