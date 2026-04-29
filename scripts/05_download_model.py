import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from huggingface_hub import snapshot_download

from src.utils import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_name = cfg["model"]["name"]
    cache_dir = cfg["model"].get("cache_dir")

    if os.path.isdir(model_name):
        print(f"Using local model directory: {model_name}")
        return

    path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        local_files_only=bool(cfg["model"].get("local_files_only", False)),
    )
    print(f"Downloaded/resolved {model_name} at {path}")


if __name__ == "__main__":
    main()
