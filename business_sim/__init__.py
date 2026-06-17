"""
business_sim

Core engine for BusinessStrategicMarketSimulator.

This package provides:
- Market and socio-economic models (core_market)
- Sun Tzu + chess-inspired strategic AI (core_ai)
- Portfolio and multi-agent market simulators (core_portfolio)
- IO utilities for JSON/CSV export and RL datasets (core_io)
"""

from .core_market import AssetType, SocioEconomicState, get_stock_price, build_assets_from_tickers, load_portfolio_config
from .core_ai import SunTzuChessMarketAI
from .core_portfolio import (
    PortfolioState,
    simulate_single_turn,
    MultiAgentInvestor,
    MultiAgentPortfolio,
)
from .core_io import export_json, export_state_vectors_csv