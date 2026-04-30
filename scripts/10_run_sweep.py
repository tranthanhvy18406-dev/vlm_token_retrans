import argparse
import os
import subprocess
import sys
from copy import deepcopy

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import ensure_dir, load_yaml


def parse_values(raw: str, value_type):
    return [value_type(item.strip()) for item in raw.split(",") if item.strip()]


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def write_config(cfg: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def make_sweep_config(base_cfg: dict, sweep_name: str, value, args) -> tuple[dict, str]:
    cfg = deepcopy(base_cfg)

    if sweep_name == "layer":
        tag = f"layer{int(value):02d}"
        cfg["model"]["layer_idx"] = int(value)
    elif sweep_name == "drop":
        tag = f"drop{str(value).replace('.', 'p')}"
        cfg["corruption"]["drop_ratio"] = float(value)
    else:
        raise ValueError(f"Unsupported sweep: {sweep_name}")

    if args.train_jsonl:
        cfg["data"]["train_jsonl"] = args.train_jsonl
    if args.val_jsonl:
        cfg["data"]["val_jsonl"] = args.val_jsonl

    cfg["data"]["max_samples"] = args.train_samples
    cfg["oracle"]["save_dir"] = os.path.join(args.cache_root, sweep_name, tag)
    cfg["train"]["checkpoint_path"] = os.path.join(
        args.checkpoint_root,
        sweep_name,
        f"mlp_scorer_{tag}.pt",
    )

    config_path = os.path.join(args.config_out_dir, sweep_name, f"{tag}.yaml")
    return cfg, config_path


def run_one(cfg: dict, config_path: str, args) -> None:
    write_config(cfg, config_path)

    if args.smoke:
        run_command(
            [
                sys.executable,
                "scripts/01_smoke_test_llava.py",
                "--config",
                config_path,
                "--skip_debug_checks",
            ],
            args.dry_run,
        )

    run_command(
        [
            sys.executable,
            "scripts/02_build_oracle_cache.py",
            "--config",
            config_path,
            "--split",
            "train",
            "--max_samples",
            str(args.train_samples),
        ],
        args.dry_run,
    )
    run_command(
        [
            sys.executable,
            "scripts/03_train_mlp_scorer.py",
            "--config",
            config_path,
        ],
        args.dry_run,
    )
    run_command(
        [
            sys.executable,
            "scripts/04_eval_retrans_loss.py",
            "--config",
            config_path,
            "--checkpoint",
            cfg["train"]["checkpoint_path"],
            "--jsonl",
            cfg["data"]["val_jsonl"],
            "--max_samples",
            str(args.val_samples),
        ],
        args.dry_run,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/stage1_mlp_gqa.yaml")
    parser.add_argument("--sweep", choices=["layer", "drop"], required=True)
    parser.add_argument("--values", required=True)
    parser.add_argument("--train_samples", type=int, default=300)
    parser.add_argument("--val_samples", type=int, default=100)
    parser.add_argument("--train_jsonl", default=None)
    parser.add_argument("--val_jsonl", default=None)
    parser.add_argument("--config_out_dir", default="outputs/sweep_configs")
    parser.add_argument("--cache_root", default="outputs/oracle_cache/sweeps")
    parser.add_argument("--checkpoint_root", default="outputs/checkpoints/sweeps")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    value_type = int if args.sweep == "layer" else float
    values = parse_values(args.values, value_type)

    print(
        f"sweep={args.sweep}, values={values}, "
        f"train_samples={args.train_samples}, val_samples={args.val_samples}",
        flush=True,
    )
    for value in values:
        cfg, config_path = make_sweep_config(base_cfg, args.sweep, value, args)
        print(f"\n===== {args.sweep}={value} =====", flush=True)
        print(f"config: {config_path}", flush=True)
        print(f"cache: {cfg['oracle']['save_dir']}", flush=True)
        print(f"checkpoint: {cfg['train']['checkpoint_path']}", flush=True)
        run_one(cfg, config_path, args)


if __name__ == "__main__":
    main()
