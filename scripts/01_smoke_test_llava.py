import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.corruption import corrupt_image_features
from src.dataset import load_jsonl
from src.llava_wrapper import LlavaRetransWrapper
from src.utils import get_torch_dtype, load_yaml, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["corruption"]["seed"])

    wrapper = LlavaRetransWrapper(
        model_name=cfg["model"]["name"],
        device=cfg["model"]["device"],
        dtype=get_torch_dtype(cfg["model"]["dtype"]),
        cache_dir=cfg["model"].get("cache_dir"),
        local_files_only=bool(cfg["model"].get("local_files_only", False)),
        device_map=cfg["model"].get("device_map"),
    )

    samples = load_jsonl(
        cfg["data"]["train_jsonl"],
        image_root=cfg["data"]["image_root"],
        max_samples=1,
    )
    sample = samples[0]

    prepared = wrapper.prepare_inputs(sample.image, sample.question, sample.answer)

    corrupted, damaged_mask, _ = corrupt_image_features(
        prepared.image_features,
        drop_ratio=cfg["corruption"]["drop_ratio"],
        mode=cfg["corruption"]["mode"],
        mask_value=cfg["corruption"]["mask_value"],
    )

    loss = wrapper.compute_loss_from_image_features(prepared, corrupted)

    hidden = wrapper.get_layer_visual_hidden(
        prepared=prepared,
        image_features=corrupted,
        layer_idx=cfg["model"]["layer_idx"],
    )

    print("Smoke test passed.")
    print("image_features:", tuple(prepared.image_features.shape))
    print("image_positions:", tuple(prepared.image_positions.shape))
    print("damaged:", int(damaged_mask.sum().item()))
    print("loss:", float(loss.item()))
    print("visual hidden:", tuple(hidden.shape))


if __name__ == "__main__":
    main()
