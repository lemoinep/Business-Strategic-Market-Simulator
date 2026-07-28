import argparse
import os
from typing import Dict, Any

import numpy as np
import pandas as pd


def compute_episode_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-episode metrics from RL batch data:
      - final_portfolio_value
      - max_drawdown
      - cumulative_reward
      - num_steps
    """
    episodes = []
    for ep, ep_df in df.groupby("episode"):
        ep_df = ep_df.sort_values("step")

        # Portfolio value over time
        pv = ep_df["portfolio_value"].values

        # Final value
        final_pv = pv[-1] if len(pv) > 0 else np.nan

        # Max drawdown (peak to trough) [web:340]
        # Normalize by peak
        running_max = np.maximum.accumulate(pv)
        drawdowns = (pv - running_max) / running_max
        max_dd = drawdowns.min() if len(drawdowns) > 0 else np.nan

        # Cumulative reward
        cum_reward = ep_df["reward"].sum()

        num_steps = len(ep_df)

        episodes.append(
            {
                "episode": ep,
                "final_portfolio_value": final_pv,
                "max_drawdown": max_dd,
                "cumulative_reward": cum_reward,
                "num_steps": num_steps,
            }
        )

    return pd.DataFrame(episodes)


def compute_aggregate_metrics(ep_metrics: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute aggregate metrics across episodes:
      - mean / median final PV
      - mean max drawdown
      - mean cumulative reward
      - % episodes with positive cumulative reward
    """
    agg = {}

    agg["num_episodes"] = len(ep_metrics)
    agg["final_pv_mean"] = float(ep_metrics["final_portfolio_value"].mean())
    agg["final_pv_median"] = float(ep_metrics["final_portfolio_value"].median())
    agg["final_pv_min"] = float(ep_metrics["final_portfolio_value"].min())
    agg["final_pv_max"] = float(ep_metrics["final_portfolio_value"].max())

    agg["max_drawdown_mean"] = float(ep_metrics["max_drawdown"].mean())
    agg["max_drawdown_min"] = float(ep_metrics["max_drawdown"].min())
    agg["max_drawdown_max"] = float(ep_metrics["max_drawdown"].max())

    agg["cum_reward_mean"] = float(ep_metrics["cumulative_reward"].mean())
    agg["cum_reward_median"] = float(ep_metrics["cumulative_reward"].median())

    pct_positive_reward = (
        (ep_metrics["cumulative_reward"] > 0).sum() / len(ep_metrics) * 100.0
        if len(ep_metrics) > 0
        else 0.0
    )
    agg["pct_positive_reward"] = pct_positive_reward

    return agg


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RL batch CSV and compute performance metrics."
    )
    parser.add_argument(
        "--csv",
        "-f",
        required=True,
        help="Path to RL batch CSV file (e.g. output/rl_batch_megacap_tech.csv).",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Optional path to save episode metrics CSV. If not set, uses <csv>_metrics.csv.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)

    ep_metrics = compute_episode_metrics(df)
    agg = compute_aggregate_metrics(ep_metrics)

    out_metrics_csv = args.out or os.path.splitext(args.csv)[0] + "_metrics.csv"
    ep_metrics.to_csv(out_metrics_csv, index=False)

    print(f"Episode metrics saved to: {out_metrics_csv}\n")
    print("Aggregate metrics:")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()