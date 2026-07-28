import argparse
import json
import os
from typing import Any, Dict

try:
    import yaml  # PyYAML, recommandé pour les configs
except ImportError:
    yaml = None

from run_rl_batch_ppo import run_batch_ppo


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML or JSON config file into a Python dict.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in [".yaml", ".yml"]:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is not installed. Install it with `pip install pyyaml` "
                "or use a JSON config instead."
            )
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        raise ValueError(f"Unsupported config extension: {ext}")

    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a single dict at top level.")

    return cfg


def run_experiment_from_config(cfg: Dict[str, Any]) -> None:
    """
    Dispatch experiment based on config dict.

    For now, supports PPO batch experiments via run_batch_ppo().
    """
    run_name = cfg.get("run_name", "ppo_experiment")
    model_path = cfg.get("model_path", "ppo_portfolio")
    seed = int(cfg.get("seed", 42))

    env_cfg = cfg.get("env", {})
    rl_cfg = cfg.get("rl", {})
    out_cfg = cfg.get("output", {})

    max_turns = int(env_cfg.get("max_turns", 30))
    tickers = env_cfg.get("tickers", None)
    cash = float(env_cfg.get("cash", 10000))

    algo = rl_cfg.get("algo", "PPO")
    num_episodes = int(rl_cfg.get("num_episodes", 100))

    out_csv = out_cfg.get("csv", f"output/{run_name}_batch.csv")

    if algo.upper() != "PPO":
        raise NotImplementedError(
            f"Unsupported RL algo '{algo}'. Currently only 'PPO' via run_batch_ppo is implemented."
        )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    print(f"[run_experiment] Starting experiment '{run_name}'")
    print(f"  Model path     : {model_path}")
    print(f"  Algo           : {algo}")
    print(f"  Seed           : {seed}")
    print(f"  Num episodes   : {num_episodes}")
    print(f"  Max turns      : {max_turns}")
    print(f"  Tickers        : {tickers}")
    print(f"  Cash           : {cash}")
    print(f"  Output CSV     : {out_csv}")

    rows = run_batch_ppo(
        model_path=model_path,
        num_episodes=num_episodes,
        max_turns=max_turns,
        tickers=tickers,
        cash=cash,
        out_csv=out_csv,
        seed=seed,
        run_name=run_name,
    )

    print(f"[run_experiment] Finished experiment '{run_name}'.")
    print(f"  Total rows written: {len(rows)}")
    print(f"  CSV file          : {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RL experiments based on a YAML/JSON config file."
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML or JSON config file.",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)
    run_experiment_from_config(cfg)


if __name__ == "__main__":
    main()