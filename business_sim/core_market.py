import random
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import yfinance as yf  


def get_stock_price(ticker: str) -> float:
    try:
        data = yf.Ticker(ticker).history(period="1d")  # [page:1]
        price = data['Close'].iloc[-1]
        return float(price)
    except Exception:
        simulated_prices = {
            'AAPL': random.uniform(140, 170),
            'GOOG': random.uniform(2500, 3000),
            'TSLA': random.uniform(700, 900),
            'MSFT': random.uniform(280, 350),
            'SPY': random.uniform(400, 450)
        }  # [page:1]
        return simulated_prices.get(ticker, random.uniform(100, 1000))

class SocioEconomicState:
    def __init__(self):
        self.unemployment_rate = random.uniform(0.03, 0.12)
        self.inflation_rate = random.uniform(0.01, 0.08)
        self.consumption_index = random.uniform(0.7, 1.2) 

    def update(self):
        self.unemployment_rate += random.uniform(-0.005, 0.007)
        self.unemployment_rate = min(max(self.unemployment_rate, 0.02), 0.20)

        self.inflation_rate += random.uniform(-0.002, 0.003)
        self.inflation_rate = min(max(self.inflation_rate, 0.005), 0.15)

        self.consumption_index += random.uniform(-0.03, 0.03)
        self.consumption_index = min(max(self.consumption_index, 0.4), 1.6) 


@dataclass
class AssetType:
    name: str
    ticker: str
    quantity: int
    volatility: float
    liquidity: float
    special: Optional[Dict] = None
    buy_date: Optional[str] = None
    buy_value: Optional[float] = None

    @property
    def price(self) -> float:
        return get_stock_price(self.ticker)  

    @property
    def value(self) -> float:
        return self.quantity * self.price  


def build_assets_from_tickers(
    tickers: List[str],
    quantities: Optional[List[int]] = None,
    default_quantity: int = 10,
) -> List[AssetType]:
    if quantities is None:
        quantities = [default_quantity] * len(tickers)

    assets = []
    for tk, qty in zip(tickers, quantities):
        price = get_stock_price(tk)
        asset = AssetType(
            name=tk,
            ticker=tk,
            quantity=qty,
            volatility=0.2,
            liquidity=0.9,
            special={"initial_price": price},
        )
        assets.append(asset)
    return assets


def load_portfolio_config(path: str):
    with open(path, "r") as f:
        cfg = json.load(f)

    cash = cfg.get("cash", 10000)
    tickers = cfg.get("tickers", [])
    quantities = cfg.get("quantities", [10] * len(tickers))

    assets = build_assets_from_tickers(tickers, quantities)
    return cash, assets