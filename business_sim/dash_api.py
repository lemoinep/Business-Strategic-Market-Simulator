import os
import sys

from typing import Dict, Any, List

import pandas as pd


from business_sim.core_portfolio import PortfolioState, simulate_single_turn


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    

def init_portfolio_state_from_config(config: Dict[str, Any]) -> PortfolioState:
    """
    Create a PortfolioState from a simple configuration:
    {
      "cash": 10000.0,
      "positions": {"NVDA": {"quantity": 20}, ...}
    }
    If no tickers are provided, fall back to PortfolioState.default().
    """
    positions = config.get("positions", {})
    tickers = [t.upper() for t in positions.keys()]

    if not tickers:
        state = PortfolioState.default()
    else:
        state = PortfolioState.from_tickers(tickers)
    return state


def explain_snapshot_result(result: Dict[str, Any]) -> str:
    """
    Generate a human-readable explanation for a single Sun Tzu / Chess AI snapshot.

    `result` is the dict returned by simulate_single_turn:
      - turn, portfolio_value, cash
      - ai_personality, ai_phase, risk_tension, center_control
      - tactics (list of strings)
      - beat_market (bool)
      - socio (optional)
    """
    turn = result.get("turn", 1)
    value = result.get("portfolio_value", 0.0)
    cash = result.get("cash", 0.0)
    phase = result.get("ai_phase", "unknown")
    personality = result.get("ai_personality", "unknown")
    tension = float(result.get("risk_tension", 0.0))
    center_control = float(result.get("center_control", 0.0))
    beat_market = bool(result.get("beat_market", False))
    tactics = result.get("tactics", [])

    lines = []

    # Overview
    lines.append(
        f"Turn {turn}: the portfolio is valued at {value:.2f} with {cash:.2f} in cash."
    )

    # Phase & personality
    lines.append(
        f"The AI identifies the current strategic phase as '{phase}' "
        f"with a '{personality}' personality."
    )

    # Risk tension & center control
    lines.append(
        f"Risk tension is {tension:.3f}, indicating a "
        + ("low-stress environment" if tension < 0.3 else "high-stress environment" if tension > 0.7 else "moderate tension")
        + f", and center control is {center_control:.3f}, "
        + ("strong control of core tickers." if center_control > 0.6 else "limited control of core tickers.")
    )

    # Market performance
    lines.append(
        "The portfolio "
        + ("beats" if beat_market else "does not beat")
        + " the simulated market benchmark over this turn."
    )

    # Tactics summary
    if tactics:
        lines.append("Key tactical recommendations:")
        for t in tactics:
            lines.append(f"  - {t}")
    else:
        lines.append("No explicit tactical recommendations were generated for this snapshot.")

    return "\n".join(lines)


def explain_simulation_results(sim_df: pd.DataFrame) -> str:
    """
    Generate a human-readable summary of a Sun Tzu simulation over multiple turns.

    Expects sim_df with columns:
      - turn
      - portfolio_value
      - beat_market
      - risk_tension
      - ai_phase (optional)
    """
    if sim_df.empty:
        return "Simulation returned no data."

    n_turns = len(sim_df)
    final_value = float(sim_df["portfolio_value"].iloc[-1])
    initial_value = float(sim_df["portfolio_value"].iloc[0])
    total_return = (final_value / initial_value - 1.0) if initial_value > 0 else 0.0

    beat_series = sim_df.get("beat_market")
    n_beat = int(beat_series.astype(bool).sum()) if beat_series is not None else 0

    # Max drawdown (reuse your compute_max_drawdown or local logic)
    values = sim_df["portfolio_value"].astype(float)
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    max_dd = float(drawdown.min())

    # Average tension and phase distribution
    avg_tension = float(sim_df.get("risk_tension", pd.Series([0.0] * n_turns)).mean())
    phase_counts = sim_df.get("ai_phase", pd.Series(["unknown"] * n_turns)).value_counts()
    dominant_phase = phase_counts.index[0] if len(phase_counts) > 0 else "unknown"

    lines = []
    lines.append(
        f"Simulation over {n_turns} turns: initial value {initial_value:.2f}, final value {final_value:.2f}, "
        f"total return {total_return * 100:.2f}%."
    )
    lines.append(
        f"The portfolio beat the market benchmark on {n_beat} turns out of {n_turns}."
    )
    lines.append(
        f"Maximum drawdown during the simulation was {max_dd * 100:.2f}%."
    )
    lines.append(
        f"Average strategic tension was {avg_tension:.3f}, with '{dominant_phase}' as the dominant phase."
    )

    return "\n".join(lines)




def update_state_with_market(
    state: PortfolioState,
    prices_row: pd.Series,
    positions_cfg: Dict[str, Any],
    cash: float,
) -> PortfolioState:
    """
    Update an existing PortfolioState with:
    - current quantities (positions_cfg),
    - current cash.

    We intentionally do NOT set asset.price here, because AssetType.price
    is a read-only property in your core model. Price updates should be
    handled by the core engine (e.g. via market data inside simulate_single_turn).
    """
    # Update cash if the PortfolioState exposes it
    try:
        state.cash = float(cash)
    except Exception:
        # If 'cash' is not an attribute or is read-only, just ignore
        pass

    # Update quantities only; do not touch price
    for asset in getattr(state, "assets", []):
        ticker = getattr(asset, "ticker", None)
        if ticker and ticker in positions_cfg:
            qty = positions_cfg[ticker].get("quantity", None)
            if qty is not None:
                try:
                    asset.quantity = float(qty)
                except Exception:
                    # If quantity is not settable, ignore
                    pass

    return state


def compute_allocation_vector(config: Dict[str, Any], state: PortfolioState) -> List[float]:
    values = []
    for asset in state.assets:
        values.append(asset.quantity * asset.price)
    total = sum(values) if values else 0.0
    if total <= 0:
        n = len(state.assets)
        return [1.0 / n] * n if n > 0 else []

    return [v / total for v in values]


def run_single_ai_turn(state: PortfolioState, alloc_vector: List[float]) -> Dict[str, Any]:
    """
    Call simulate_single_turn for a single turn and return the result dict.
    For the MVP, we simply use turn_index = 1.
    """
    turn_index = 1
    result = simulate_single_turn(state, alloc_vector, turn_index)
    return result


def get_ai_snapshot(
    prices_df: pd.DataFrame,
    positions_cfg: Dict[str, Any],
    cash: float,
) -> tuple[PortfolioState, Dict[str, Any]]:
    """
    Build a PortfolioState from the latest market data and current positions,
    compute an allocation vector, and run a single AI turn.
    Returns:
      (updated_state, ai_result_dict)
    """
    latest_row = prices_df.iloc[-1]

    # Start from a state built from the config (positions + cash)
    state = init_portfolio_state_from_config({"positions": positions_cfg, "cash": cash})

    # Update the state with current quantities and cash
    state = update_state_with_market(state, latest_row, positions_cfg, cash)

    # Compute allocation and run a single AI turn
    alloc_vector = compute_allocation_vector({}, state)
    result = simulate_single_turn(state, alloc_vector, 1)

    return state, result



def run_ai_simulation(
    state: PortfolioState,
    alloc_vector: List[float],
    n_turns: int,
) -> List[Dict[str, Any]]:
    """
    Run a multi-turn simulation using the Sun Tzu / Chess AI engine.

    It calls simulate_single_turn iteratively, updating the state in place
    and collecting the turn_result dicts returned by the core engine.
    """
    results: List[Dict[str, Any]] = []

    for turn_index in range(1, n_turns + 1):
        turn_result = simulate_single_turn(state, alloc_vector, turn_index)
        results.append(turn_result)

        try:
            if hasattr(state, "total_value") and state.total_value() <= 0:
                break
        except Exception:
            pass

    return results



def run_rl_simulation(
    model_path: str,
    env_config: Dict[str, Any],
    n_steps: int,
) -> List[Dict[str, Any]]:
    """
    Run a RL-based portfolio simulation using a PPO model.

    Returns a list of dicts with at least:
      - step
      - portfolio_value
      - reward
      - beat_market (optional, bool)
    """
    from stable_baselines3 import PPO
    from gym_portfolio_env import PortfolioEnv  

    env = PortfolioEnv(**env_config)
    model = PPO.load(model_path)
    obs, _ = env.reset()
    results: List[Dict[str, Any]] = []

    for t in range(1, n_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        state_info = info.get("state_info", {})

        results.append({
            "step": t,
            "portfolio_value": state_info.get("portfolio_value"),
            "reward": float(reward),
            "beat_market": bool(state_info.get("beat_market", False)),
        })

        if terminated or truncated:
            break

    return results




