import random
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

from .core_market import AssetType, SocioEconomicState, get_stock_price
from .core_ai import SunTzuChessMarketAI


@dataclass
class PortfolioState:
    assets: List[AssetType]
    cash: float
    market_ai: SunTzuChessMarketAI
    market_cond: Dict[str, float]
    socio_state: Optional[SocioEconomicState] = None
    risk_aversion: float = 0.7

    @classmethod
    def default(cls, cash: float = 10000):
        stocks = [
            {"name": "Apple", "ticker": "AAPL", "quantity": 20},
            {"name": "Google", "ticker": "GOOG", "quantity": 3},
            {"name": "Tesla", "ticker": "TSLA", "quantity": 8},
            {"name": "Microsoft", "ticker": "MSFT", "quantity": 10},
            {"name": "S&P 500 ETF", "ticker": "SPY", "quantity": 10},
        ]  
        assets = [
            AssetType(s["name"], s["ticker"], s["quantity"], 0.2, 0.9) for s in stocks
        ]
        market_ai = SunTzuChessMarketAI(
            personality=random.choice(["bullish", "bearish", "sideways"])
        )  
        market_cond = {
            "fear_index": random.uniform(0.1, 0.9),
            "liquidity": random.uniform(0.5, 1.0),
            "volatility": random.uniform(0.1, 0.4),
        }  
        return cls(
            assets=assets,
            cash=cash,
            market_ai=market_ai,
            market_cond=market_cond,
        )

    @classmethod
    def from_tickers(
        cls,
        tickers: List[str],
        quantities: Optional[List[int]] = None,
        cash: float = 10000,
    ):
        from .core_market import build_assets_from_tickers

        assets = build_assets_from_tickers(tickers, quantities)
        market_ai = SunTzuChessMarketAI(
            personality=random.choice(["bullish", "bearish", "sideways"])
        )
        market_cond = {
            "fear_index": random.uniform(0.1, 0.9),
            "liquidity": random.uniform(0.5, 1.0),
            "volatility": random.uniform(0.1, 0.4),
        }
        return cls(
            assets=assets,
            cash=cash,
            market_ai=market_ai,
            market_cond=market_cond,
        )

    def total_value(self) -> float:
        return self.cash + sum(a.value for a in self.assets)

    def asset_allocation(self) -> List[float]:
        tv = self.total_value()
        return [a.value / tv for a in self.assets] if tv > 0 else [0.0] * len(
            self.assets
        )

    def update_market_conditions(self):
        self.market_cond["fear_index"] = min(
            1.0,
            max(
                0.0,
                self.market_cond["fear_index"] + random.uniform(-0.1, 0.1),
            ),
        )
        self.market_cond["liquidity"] = min(
            1.0,
            max(
                0.0,
                self.market_cond["liquidity"] + random.uniform(-0.05, 0.05),
            ),
        )
        self.market_cond["volatility"] = min(
            1.0,
            max(
                0.0,
                self.market_cond["volatility"] + random.uniform(-0.07, 0.07),
            ),
        )

    def market_index_perf(self) -> float:
        return get_stock_price("SPY")

    def load_portfolio_from_file(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        cash = data.get("cash", 10000)
        stocks = data.get("stocks", [])
        assets: List[AssetType] = []
        for s in stocks:
            at = AssetType(
                name=s["name"],
                ticker=s["ticker"],
                quantity=s["quantity"],
                volatility=0.2,
                liquidity=0.9,
            )
            at.buy_date = s.get("buy_date")
            at.buy_value = s.get("buy_value")
            assets.append(at)
        self.cash = cash
        self.assets = assets

    def save_portfolio_to_file(self, filepath: str):
        stocks_data = []
        for asset in self.assets:
            stocks_data.append(
                {
                    "name": asset.name,
                    "ticker": asset.ticker,
                    "quantity": asset.quantity,
                    "buy_date": asset.buy_date,
                    "buy_value": asset.buy_value,
                }
            )
        data = {"cash": self.cash, "stocks": stocks_data}
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


def simulate_single_turn(
    state: PortfolioState,
    alloc_percent: List[float],  
    turn_index: int,
) -> Dict:
    state.update_market_conditions()
    if state.socio_state is not None:
        state.socio_state.update()

    portfolio_value_before = state.total_value()
    market_index_start = state.market_index_perf()

    all_recs: List[str] = []

    fear = state.market_cond["fear_index"]
    vol = state.market_cond["volatility"]
    liq = state.market_cond["liquidity"]

    for asset in state.assets:
        control_central = state.market_ai.evaluate_center_control(state.assets)
        risk_tension = state.market_ai.compute_risk_tension(
            fear_index=fear, volatility=vol, liquidity=liq, socio=state.socio_state
        )
        recs = state.market_ai.strategy_recommendations(
            asset=asset,
            risk_tension=risk_tension,
            cash_available=state.cash,
            control_central=control_central,
            turn_index=turn_index,
            fear_index=fear,
            volatility=vol,
            liquidity=liq,
        )
        all_recs.extend(recs)


    tickers = [a.ticker for a in state.assets]
    alloc_used = ai_adjust_allocation(alloc_percent, all_recs, tickers)

    total_value = state.total_value()
    for i, asset in enumerate(state.assets):
        target_value = total_value * alloc_percent[i]
        price = asset.price
        qty_target = target_value / price if price > 0 else 0
        diff_qty = int(qty_target - asset.quantity)
        if diff_qty > 0:
            cost = diff_qty * price
            if cost <= state.cash:
                state.cash -= cost
                asset.quantity += diff_qty
        elif diff_qty < 0:
            revenue = -diff_qty * price
            state.cash += revenue
            asset.quantity += diff_qty

    ai_react = state.market_ai.adjust_market_conditions(
        state.market_cond, portfolio_value_before
    )
    if ai_react["strong_drop"]:
        for asset in state.assets:
            drop = random.uniform(0.05, 0.15)
            if random.random() < 0.8:
                asset.quantity = max(0, int(asset.quantity * (1 - drop)))
    if ai_react["surprise_rally"]:
        for asset in state.assets:
            gain = random.uniform(0.04, 0.09)
            if random.random() < 0.7:
                asset.quantity = int(asset.quantity * (1 + gain))

    new_value = state.total_value()
    market_index_end = state.market_index_perf()
    beat_market = (new_value - portfolio_value_before) > (
        market_index_end - market_index_start
    )

    state.market_ai.observe_outcome(
        beat_market=beat_market,
        risk_tension=risk_tension,
        portfolio_dist=state.asset_allocation(),
        control_central=control_central,
    )
    state.market_ai.decide_personality()

    turn_result = {
        "turn": turn_index,
        "portfolio_value": new_value,
        "cash": state.cash,
        "market_fear": fear,
        "market_liquidity": liq,
        "market_volatility": vol,
        "ai_personality": state.market_ai.personality,
        "ai_phase": state.market_ai.phase,
        "center_control": control_central,
        "risk_tension": risk_tension,
        "tactics": all_recs,
        "beat_market": beat_market,
    }
    if state.socio_state is not None:
        turn_result["socio"] = {
            "unemployment_rate": state.socio_state.unemployment_rate,
            "inflation_rate": state.socio_state.inflation_rate,
            "consumption_index": state.socio_state.consumption_index,
        }

    return turn_result
    
    
def ai_adjust_allocation(
    alloc_percent: List[float],
    recs: List[str],
    tickers: List[str],
) -> List[float]:
    
    alloc = alloc_percent[:]
    for i, tk in enumerate(tickers):
        has_attack = any(tk in r and "[ATTACK]" in r for r in recs)
        has_def = any(
            tk in r and ("[DEFENSE]" in r or "[SIMPLIFICATION]" in r) for r in recs
        )
        if has_attack and not has_def:
            alloc[i] *= 1.2
        elif has_def and not has_attack:
            alloc[i] *= 0.8

    s = sum(alloc)
    if s > 0:
        alloc = [a / s for a in alloc]
    return alloc
    
    
class MultiAgentInvestor:
    def __init__(self, name: str, risk_aversion: float, cash: float):
        self.name = name
        self.risk_aversion = risk_aversion
        self.cash = cash
        self.assets: Dict[str, AssetType] = {}
        self.personality = random.choice(["bullish", "bearish", "sideways"])
        self.market_ai = SunTzuChessMarketAI(self.personality)

    def evaluate_center_control(self) -> float:
        center_tickers = ["AAPL", "GOOG", "MSFT"]
        central_value = sum(
            asset.value for tk, asset in self.assets.items() if tk in center_tickers
        )
        total = sum(asset.value for asset in self.assets.values())
        return central_value / total if total > 0 else 0.0

    def take_turn(
        self,
        socio: SocioEconomicState,
        market_cond: Dict[str, float],
        turn_index: int,
    ):
        logs: List[str] = []
        actions: List[Dict] = []

        for ticker, asset in self.assets.items():
            control_central = self.evaluate_center_control()
            risk_tension = self.market_ai.compute_risk_tension(
                socio=socio,
                volatility=market_cond["volatility"],
                liquidity=market_cond["liquidity"],
            )
            recs = self.market_ai.strategy_recommendations(
                asset=asset,
                risk_tension=risk_tension,
                cash_available=self.cash,
                control_central=control_central,
                turn_index=turn_index,
                volatility=market_cond["volatility"],
                liquidity=market_cond["liquidity"],
            )

            for r in recs:
                logs.append(f"{self.name} AI: {r}")

            action = {
                "agent": self.name,
                "asset": ticker,
                "personality": self.personality,
                "phase": self.market_ai.phase,
                "decision": None,
                "recommendations": recs,
                "quantity_before": asset.quantity,
                "cash_before": self.cash,
            }

            if (
                self.personality == "bullish"
                and risk_tension < 0.5
                and self.cash > asset.price
            ):
                buy_qty = random.randint(1, 2)
                asset.quantity += buy_qty
                self.cash -= asset.price * buy_qty
                logs.append(f"{self.name}: Bought {buy_qty} {asset.ticker}")
                action["decision"] = f"buy {buy_qty}"
            elif (
                self.personality == "bearish"
                and risk_tension > 0.6
                and asset.quantity > 1
            ):
                sell_qty = random.randint(1, 2)
                asset.quantity -= sell_qty
                self.cash += asset.price * sell_qty
                logs.append(f"{self.name}: Sold {sell_qty} {asset.ticker}")
                action["decision"] = f"sell {sell_qty}"
            else:
                action["decision"] = "hold"

            action["quantity_after"] = asset.quantity
            action["cash_after"] = self.cash
            actions.append(action)

        beat_market = random.random() > 0.5
        self.market_ai.observe_outcome(
            beat_market=beat_market,
            risk_tension=risk_tension,
        )
        self.market_ai.decide_personality()

        return logs, actions


class MultiAgentPortfolio:
    def __init__(self, n_agents: int = 4, base_risk_aversion: float = 0.5):
        self.socio_state = SocioEconomicState()
        self.market_cond = {
            "fear_index": random.uniform(0.1, 0.9),
            "liquidity": random.uniform(0.5, 1.0),
            "volatility": random.uniform(0.14, 0.4),
        }
        self.base_risk_aversion = base_risk_aversion
        self.agents: List[MultiAgentInvestor] = []

        for i in range(n_agents):
            risk_av = random.uniform(0.1, 0.9)
            agent = MultiAgentInvestor(f"Investor_{i+1}", risk_av, 10000)
            agent.assets = {
                "AAPL": AssetType("Apple", "AAPL", random.randint(10, 30), 0.2, 0.9),
                "GOOG": AssetType("Google", "GOOG", random.randint(1, 6), 0.25, 0.85),
                "TSLA": AssetType("Tesla", "TSLA", random.randint(4, 12), 0.3, 0.8),
                "MSFT": AssetType(
                    "Microsoft", "MSFT", random.randint(5, 15), 0.2, 0.85
                ),
                "SPY": AssetType(
                    "S&P 500 ETF", "SPY", random.randint(4, 20), 0.18, 0.95
                ),
            }
            self.agents.append(agent)

        self.history: List[Dict] = []

    def update_market_and_socio(self):
        self.market_cond["fear_index"] = min(
            1.0,
            max(
                0.0,
                self.market_cond["fear_index"] + random.uniform(-0.1, 0.1),
            ),
        )
        self.market_cond["liquidity"] = min(
            1.0,
            max(
                0.0,
                self.market_cond["liquidity"] + random.uniform(-0.05, 0.05),
            ),
        )
        self.market_cond["volatility"] = min(
            1.0,
            max(
                0.0,
                self.market_cond["volatility"] + random.uniform(-0.07, 0.07),
            ),
        )
        self.socio_state.update()

    def one_turn(self, turn_index: int) -> Dict:
        self.update_market_and_socio()

        logs: List[str] = []
        actions: List[Dict] = []

        for agent in self.agents:
            agent_logs, agent_actions = agent.take_turn(
                self.socio_state, self.market_cond, turn_index
            )
            logs += agent_logs
            actions += agent_actions

        turn_record = {
            "turn": turn_index,
            "logs": logs,
            "actions": actions,
            "socio": {
                "unemployment_rate": self.socio_state.unemployment_rate,
                "inflation_rate": self.socio_state.inflation_rate,
                "consumption_index": self.socio_state.consumption_index,
            },
            "market": dict(self.market_cond),
            "total_market_value": self.total_market_value(),
        }
        self.history.append(turn_record)
        return turn_record

    def total_market_value(self) -> float:
        value = 0.0
        for agent in self.agents:
            value += agent.cash
            for asset in agent.assets.values():
                value += asset.value
        return value
    
    
class PortfolioEnv:
    """
    Simple RL-style environment for the single-portfolio simulator.
    API similar to Gym: reset() -> state, step(action) -> (state, reward, done, info).
    """

    def __init__(self, max_turns: int = 50, tickers: Optional[List[str]] = None, cash: float = 10000):
        self.max_turns = max_turns
        self._tickers = tickers
        self._cash = cash
        self.turn_index = 0
        self.state_obj: Optional[PortfolioState] = None

    def reset(self) -> List[float]:
        """
        Reset environment and return initial state vector.
        """
        if self._tickers:
            self.state_obj = PortfolioState.from_tickers(self._tickers, cash=self._cash)
        else:
            self.state_obj = PortfolioState.default(cash=self._cash)
        self.turn_index = 0
        return self._build_state_vector(initial=True)

    def step(self, action_alloc: List[float]):
        """
        Perform one simulation turn given an allocation action.

        action_alloc: list of floats of length N == len(self.state_obj.assets),
                      representing target weights, expected to sum ~1.
        Returns: (state, reward, done, info_dict)
        """
        assert self.state_obj is not None, "Call reset() before step()."

        # Normalize action if needed
        n = len(self.state_obj.assets)
        if len(action_alloc) != n:
            raise ValueError(f"Action length {len(action_alloc)} != number of assets {n}")
        s = sum(action_alloc)
        if s <= 0:
            action_alloc = [1.0 / n] * n
        else:
            action_alloc = [a / s for a in action_alloc]

        self.turn_index += 1
        prev_value = self.state_obj.total_value()
        turn_result = simulate_single_turn(self.state_obj, action_alloc, self.turn_index)
        new_value = turn_result["portfolio_value"]

        reward = new_value - prev_value

        state_vec = self._build_state_vector_from_turn(turn_result)
        done = self.turn_index >= self.max_turns or new_value <= 0

        info = turn_result 

        return state_vec, reward, done, info

    def _build_state_vector(self, initial: bool = False) -> List[float]:
        """
        Build a state vector from the current PortfolioState (no turn_result yet).
        """
        tv = self.state_obj.total_value()
        alloc = self.state_obj.asset_allocation()
        fear = self.state_obj.market_cond["fear_index"]
        liq = self.state_obj.market_cond["liquidity"]
        vol = self.state_obj.market_cond["volatility"]

        return [
            self.state_obj.cash,
            tv,
            fear,
            liq,
            vol,
            *alloc,
        ]

    def _build_state_vector_from_turn(self, tr: Dict) -> List[float]:
        """
        Build a state vector from a turn_result dict.
        """
        return [
            tr["cash"],
            tr["portfolio_value"],
            tr["market_fear"],
            tr["market_liquidity"],
            tr["market_volatility"],
            tr["center_control"],
            tr["risk_tension"],
        ] 
