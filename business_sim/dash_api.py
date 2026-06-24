import os
import sys

from typing import Dict, Any, List

import pandas as pd


from business_sim.core_portfolio import PortfolioState, simulate_single_turn


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
   
    
from typing import Dict, Any, List
from business_sim.core_portfolio import PortfolioState


def build_decision_recommendation(
    state: PortfolioState,
    result: Dict[str, Any],
    max_trade_fraction: float = 0.1,
    max_position_weight: float = 0.25,
    profit_threshold: float = 0.20,
    drawdown_threshold: float = -0.15,
) -> Dict[str, Any]:
    """
    Build a human-readable decision recommendation based on the current state
    and Sun Tzu AI snapshot result.

    Adds richer sell logic:
      - Reduce positions that are too large (max_position_weight).
      - Take profits above profit_threshold (+20% by default).
      - Cut losses / reduce exposure below drawdown_threshold (-15% by default).

    Returns:
      - 'text': explanation string ("I recommend ...")
      - 'actions': list of structured trade actions
    """
    tactics: List[str] = result.get("tactics", [])
    tension = float(result.get("risk_tension", 0.0))
    center_control = float(result.get("center_control", 0.0))

    actions: List[Dict[str, Any]] = []
    reasons: List[str] = []

    tickers_in_state = [a.ticker for a in state.assets]

    # 1) Tactiques Sun Tzu: renforcement / ajustement
    for t in tactics:
        t_lower = t.lower()
        if "strengthen" in t_lower:
            # Try to extract ticker from the text
            parts = t.split()
            ticker = None
            for p in parts:
                p_clean = "".join(c for c in p if c.isalpha()).upper()
                if p_clean in tickers_in_state:
                    ticker = p_clean
                    break
            if ticker is None:
                continue

            asset = next(a for a in state.assets if a.ticker == ticker)
            max_qty = int(asset.quantity * max_trade_fraction) or 1

            # Base rule from tension: low tension = buy, high tension = sell
            if tension < 0.3:
                side = "buy"
                qty = max_qty
            elif tension > 0.7:
                side = "sell"
                qty = max_qty
            else:
                # Moderate tension: small adjust depending on center_control
                side = "buy" if center_control > 0.5 else "hold"
                qty = max(1, int(max_qty / 2))

            if side != "hold" and qty > 0:
                actions.append({
                    "ticker": ticker,
                    "side": side,
                    "quantity": qty,
                    "reason": f"Tactic: {t}",
                })

    # 2) Risk management: reduce oversized positions
    total_value = sum(a.quantity * a.price for a in state.assets) or 1.0
    for asset in state.assets:
        position_value = asset.quantity * asset.price
        weight = position_value / total_value
        if weight > max_position_weight and asset.quantity > 0:
            # Reduce the position by a fraction of its size
            sell_qty = int(asset.quantity * max_trade_fraction) or 1
            actions.append({
                "ticker": asset.ticker,
                "side": "sell",
                "quantity": sell_qty,
                "reason": (
                    f"Risk management: reduce oversized position (weight={weight:.2f} > "
                    f"{max_position_weight:.2f})."
                ),
            })

    # 3) Profit-taking and drawdown logic (if avg_price is available)
    for asset in state.assets:
        if getattr(asset, "avg_price", None) is None or asset.avg_price <= 0:
            continue  # skip if we don't have cost basis

        current_price = asset.price
        pnl_percent = (current_price - asset.avg_price) / asset.avg_price

        if pnl_percent >= profit_threshold and asset.quantity > 0:
            # Take partial profits
            sell_qty = int(asset.quantity * max_trade_fraction) or 1
            actions.append({
                "ticker": asset.ticker,
                "side": "sell",
                "quantity": sell_qty,
                "reason": (
                    f"Profit taking: position up {pnl_percent:.0%} from cost basis "
                    f"(threshold={profit_threshold:.0%})."
                ),
            })
        elif pnl_percent <= drawdown_threshold and asset.quantity > 0:
            # Cut losses / reduce exposure
            sell_qty = int(asset.quantity * max_trade_fraction) or 1
            actions.append({
                "ticker": asset.ticker,
                "side": "sell",
                "quantity": sell_qty,
                "reason": (
                    f"Drawdown control: position down {pnl_percent:.0%} from cost basis "
                    f"(threshold={drawdown_threshold:.0%})."
                ),
            })

    # 4) 
    if not actions:
        text = (
            "I recommend not making any major transactions at the moment.\n"
            "The current tactics and risk signals do not suggest a clear adjustment for your portfolio."
        )
    else:
        lines = ["I recommend adjusting your portfolio as follows:"]
        for a in actions:
            verb = "buy" if a["side"] == "buy" else "sell"
            lines.append(
                f"- {verb} {a['quantity']} shares of {a['ticker']} "
                f"({a['reason']})"
            )
        lines.append(
            "These recommendations are based on the current phase, market tension, "
            "central market control and basic risk-management rules (position sizing, "
            "profit-taking, and drawdown control)."
        )
        text = "\n".join(lines)

    return {
        "text": text,
        "actions": actions,
    }




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


def update_position_avg_price(
    positions: Dict[str, Dict[str, float]],
    ticker: str,
    side: str,
    qty: int,
    trade_price: float,
) -> None:
    """
    Update quantity and avg_price in positions[ticker] using average cost method.

    positions[ticker] is expected to have:
      - 'quantity': current shares
      - 'avg_price': current avg cost per share (if absent, assumed 0)
    """
    if ticker not in positions or qty <= 0:
        return

    current_qty = float(positions[ticker].get("quantity", 0.0))
    current_avg_price = float(positions[ticker].get("avg_price", 0.0))

    if side == "buy":
        # Average cost update: new_avg = (old_cost + new_cost) / new_shares
        total_cost_old = current_avg_price * current_qty
        total_cost_new = total_cost_old + qty * trade_price
        total_shares_new = current_qty + qty

        if total_shares_new > 0:
            new_avg_price = total_cost_new / total_shares_new
        else:
            new_avg_price = 0.0

        positions[ticker]["quantity"] = total_shares_new
        positions[ticker]["avg_price"] = new_avg_price

    elif side == "sell":
        total_shares_new = max(0.0, current_qty - qty)

        positions[ticker]["quantity"] = total_shares_new
        if total_shares_new <= 0:
            positions[ticker]["avg_price"] = 0.0
            
            


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




