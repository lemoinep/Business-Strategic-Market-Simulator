# Analysis of RL trajectories with Pandas


import pandas as pd
import os

import matplotlib.pyplot as plt


def load_data(path: str = "rl_batch_data.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def print_stats(df: pd.DataFrame):
    print("=== Basic stats ===")
    print(df.head())
    print("\n=== Describe ===")
    print(df.describe())

    nb_episodes = df["episode"].nunique()
    print(f"\nNumber of episodes:", nb_episodes)

    steps_per_ep = df.groupby("episode")["step"].max()
    print("Average steps per episode:", steps_per_ep.mean())
    print("Min steps per episode:", steps_per_ep.min())
    print("Max steps per episode:", steps_per_ep.max())

    print("\n=== Reward per episode (mean & sum) ===")
    reward_by_ep = df.groupby("episode")["reward"].agg(["mean", "sum"])
    print(reward_by_ep.head())


def save_random_episodes_plots(df: pd.DataFrame, n: int = 3, out_dir: str = "plots"):
    """
    Save trajectories of n random episodes as PNG (matplotlib).
    Does not show anything on screen.
    """
    os.makedirs(out_dir, exist_ok=True)

    sample_eps = df["episode"].drop_duplicates().sample(n, random_state=0).tolist()
    print(f"\nSaving trajectories for episodes {sample_eps} in {out_dir}/")

    for ep in sample_eps:
        dfe = df[df["episode"] == ep]
        plt.figure()
        plt.plot(dfe["step"], dfe["portfolio_value"])
        plt.xlabel("Step")
        plt.ylabel("Portfolio value")
        plt.title(f"Portfolio value - Episode {ep}")
        plt.grid(True)
        outfile = os.path.join(out_dir, f"episode_{ep}_portfolio_value.png")
        plt.savefig(outfile)
        plt.close()
        print(f"  -> {outfile}")


def save_reward_histogram(df: pd.DataFrame, out_dir: str = "plots"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.hist(df["reward"], bins=50)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Reward distribution per step")
    outfile = os.path.join(out_dir, "reward_histogram.png")
    plt.savefig(outfile)
    plt.close()
    print(f"\nReward histogram saved to {outfile}")


def main():
    df = load_data("rl_batch_data.csv")

    print_stats(df)
    save_random_episodes_plots(df, n=3, out_dir="plots")
    save_reward_histogram(df, out_dir="plots")

    print("\n=== Analysis complete ===")
    print("Stats are printed in the console, plots are saved as PNG in the 'plots/' folder.")


if __name__ == "__main__":
    main()