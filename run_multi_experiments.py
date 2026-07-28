import argparse
import glob
import os
from typing import List

from run_experiment import load_config, run_experiment_from_config


def find_config_files(patterns: List[str]) -> List[str]:
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple RL experiments from a list of config files."
    )
    parser.add_argument(
        "--configs",
        "-c",
        nargs="+",
        required=True,
        help="Config file paths or glob patterns (e.g. configs/ppo_*.yaml).",
    )
    args = parser.parse_args()

    config_files = find_config_files(args.configs)
    if not config_files:
        print("No config files found for patterns:", args.configs)
        return

    print("Found config files:")
    for cf in config_files:
        print("  -", cf)

    for cf in config_files:
        print("\n=== Running experiment for config:", cf, "===")
        cfg = load_config(cf)
        run_experiment_from_config(cfg)


if __name__ == "__main__":
    main()
