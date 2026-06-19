# compare_random_vs_ppo
# Statistical comparison: random policy vs PPO

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def summarize(df: pd.DataFrame, label: str):
    print(f"\n=== {label} ===")
    nb_episodes = df["episode"].nunique()
    print("Number of episodes:", nb_episodes)

    # Total reward per episode
    rewards = df.groupby("episode")["reward"].sum()
    print("Mean total reward:", rewards.mean())
    print("Min total reward:", rewards.min())
    print("Max total reward:", rewards.max())

    # Final portfolio value per episode
    last_vals = df.sort_values(["episode", "step"]).groupby("episode")["portfolio_value"].last()
    print("Mean final portfolio value:", last_vals.mean())
    print("Min final portfolio value:", last_vals.min())
    print("Max final portfolio value:", last_vals.max())


def main():
    df_rand = load_csv("rl_batch_data.csv")
    df_ppo = load_csv("rl_batch_data_ppo.csv")

    summarize(df_rand, "Random policy")
    summarize(df_ppo, "PPO policy")


if __name__ == "__main__":
    main()