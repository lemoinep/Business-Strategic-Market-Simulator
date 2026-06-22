import csv
from typing import List

from tqdm import tqdm

from stable_baselines3 import PPO
from business_sim.core_portfolio import PortfolioEnv


def ppo_policy(model: PPO, state: List[float]) -> List[float]:
    action, _ = model.predict(state, deterministic=True)
    return action.tolist()


def run_batch_ppo(
    model_path: str = "ppo_portfolio",
    num_episodes: int = 100,
    max_turns: int = 30,
    tickers=None,
    cash: float = 10000,
    out_csv: str = "rl_batch_data_ppo.csv",
):
    model = PPO.load(model_path)

    rows = []

    for ep in tqdm(range(num_episodes), desc="Generating PPO episodes"):
        env = PortfolioEnv(max_turns=max_turns, tickers=tickers, cash=cash)
        state = env.reset()
        done = False
        step_idx = 0

        while not done:
            action = ppo_policy(model, state)
            next_state, reward, done, info = env.step(action)
            step_idx += 1

            row = {
                "episode": ep,
                "step": step_idx,
                "reward": reward,
                "done": int(done),
            }

            for i, s_val in enumerate(next_state):
                row[f"state_{i}"] = s_val

            row["cash"] = info.get("cash", 0.0)
            row["portfolio_value"] = info.get("portfolio_value", 0.0)
            row["market_fear"] = info.get("market_fear", 0.0)
            row["market_liquidity"] = info.get("market_liquidity", 0.0)
            row["market_volatility"] = info.get("market_volatility", 0.0)
            row["center_control"] = info.get("center_control", 0.0)
            row["risk_tension"] = info.get("risk_tension", 0.0)
            row["ai_personality"] = info.get("ai_personality", "")
            row["ai_phase"] = info.get("ai_phase", "")
            row["turn"] = info.get("turn", step_idx)

            for i, a in enumerate(action):
                row[f"action_{i}"] = a

            rows.append(row)
            state = next_state

    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return rows


if __name__ == "__main__":
    run_batch_ppo(
        model_path="ppo_portfolio",
        num_episodes=100,
        max_turns=30,
        tickers=["NVDA", "AMD", "MSFT", "SPY"],
        cash=20000,
        out_csv="rl_batch_data_ppo.csv",
    )