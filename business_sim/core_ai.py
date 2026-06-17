import random
from typing import List, Dict, Tuple, Optional

from .core_market import AssetType, SocioEconomicState


class SunTzuChessMarketAI:

    def __init__(self, personality: str, memory_len: int = 7):
        self.personality = personality
        self.memory_len = memory_len
        self.memory: List[Dict] = []
        self.last_allocation: Optional[List[float]] = None
        self.phase = "opening"
        self.tension_profile: List[float] = []
        self.turn_count = 0

    def observe_outcome(
        self,
        beat_market: bool,
        risk_tension: float,
        portfolio_dist: Optional[List[float]] = None,
        control_central: Optional[float] = None,
    ):
        
        self.memory.append(
            {
                "beat_market": beat_market,
                "risk_tension": risk_tension,
                "portfolio_dist": portfolio_dist,
                "control_central": control_central,
            }
        )
        if len(self.memory) > self.memory_len:
            self.memory.pop(0)

        if portfolio_dist is not None:
            self.last_allocation = portfolio_dist

        self.tension_profile.append(risk_tension)
        self.turn_count += 1
        self.update_phase()

    def decide_personality(self):
        n = len(self.memory)
        if n < self.memory_len:
            return
        wins = [m["beat_market"] for m in self.memory]
        win_rate = sum(wins) / n
        if win_rate > 0.7:
            self.personality = "bullish"
        elif win_rate < 0.3:
            self.personality = "bearish"
        else:
            self.personality = "sideways"

    def update_phase(self):
        if self.turn_count < 3:
            self.phase = "opening"
        elif max(self.tension_profile[-3:]) > 0.65:
            self.phase = "middlegame"
        elif min(self.tension_profile[-3:]) < 0.35:
            self.phase = "endgame"
        else:
            self.phase = "stability"

    def compute_risk_tension(
        self,
        fear_index: Optional[float] = None,
        volatility: Optional[float] = None,
        liquidity: Optional[float] = None,
        socio: Optional[SocioEconomicState] = None,
    ) -> float:
        
        if socio is not None and volatility is not None and liquidity is not None:
            return (
                0.25 * socio.unemployment_rate
                + 0.25 * socio.inflation_rate
                + 0.3 * volatility
                + 0.2 * (1 - liquidity)
            ) 
        else:
            fear = fear_index or 0.5
            vol = volatility or 0.2
            liq = liquidity or 0.8
            return 0.4 * fear + 0.3 * vol + 0.3 * (1 - liq)  


    def evaluate_center_control(self, assets: List[AssetType]) -> float:
        center_tickers = ["AAPL", "GOOG", "MSFT"]  # [page:1][page:2]
        central_value = sum(a.value for a in assets if a.ticker in center_tickers)
        total = sum(a.value for a in assets)
        return central_value / total if total > 0 else 0.0


    def adjust_market_conditions(self, market_sentiment, portfolio_perf):
        self.decide_personality()
        return {
            "volatile": self.personality == "sideways",
            "strong_drop": self.personality == "bearish" and self.phase == "middlegame",
            "surprise_rally": self.personality == "bullish" and random.random() < 0.15,
        }


    def strategy_recommendations(
        self,
        asset: AssetType,
        risk_tension: float,
        cash_available: float,
        control_central: float,
        turn_index: int,
        fear_index: Optional[float] = None,
        volatility: Optional[float] = None,
        liquidity: Optional[float] = None,
    ) -> List[str]:
        recs: List[str] = []

        if risk_tension > 0.7 and (liquidity is not None and liquidity < 0.5):
            recs.append(
                f"[DEFENSE] Reduce {asset.ticker}, high tension and low liquidity."
            )

        elif control_central > 0.55 and cash_available > asset.price:
            recs.append(
                f"[ATTACK] Strengthen {asset.ticker}, central market control (>55%)."
            )

        elif risk_tension < 0.4:
            recs.append(
                f"[ENDGAME] Stabilize {asset.ticker}: low tension, aim for regular returns."
            )

        elif volatility is not None and volatility > 0.6 and asset.quantity > 0:
            recs.append(
                f"[FLEXIBILITY] Sell part of {asset.ticker} due to excessive volatility."
            )
        elif liquidity is not None and liquidity < 0.4:
            recs.append(
                "[CAUTION] Preserve liquidity, avoid overly aggressive moves."
            )

        if self.phase == "opening":
            recs.append(f"[PREPARATION] Position {asset.ticker} for next cycle.")
        elif self.phase == "middlegame":
            recs.append(
                f"[TENSION] Exploit imbalances, play on mobility of {asset.ticker}."
            )
        elif self.phase == "endgame":
            recs.append(
                f"[SIMPLIFICATION] Reduce risks, liquidate {asset.ticker} if necessary."
            )
        elif self.phase == "stability":
            recs.append("[STABLE] Maximize returns without high risk.")

        if turn_index % 4 == 0:
            recs.append(
                f"[PROPHYLAXIS] Re-evaluate {asset.ticker} "
                f"(anticipate market change, turn {turn_index})."
            )

        return recs