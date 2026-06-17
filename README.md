# Business Strategic Market Simulator

[![Version](https://img.shields.io/badge/version-1.0-green.svg)](https://github.com/lemoinep/Business-Strategic-Market-Simulator)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

---

BusinessStrategicMarketSimulator is a research‑oriented playground for **business‑driven portfolio and market simulation**, combining:

- a **single‑portfolio “business strategy” simulator** with Sun Tzu and chess‑inspired AI,  
- a **multi‑agent market simulator** with heterogeneous investors and socio‑economic dynamics,  
- a **clean core / GUI separation** and **RL‑ready** export suitable for HPC experiments. 

The project uses live or simulated market data to explore how strategic reasoning can influence investment decisions in a business context. This is a continuation of the work from my previous repo. [github](https://github.com/lemoinep/BusinessStrategySimulator)

***

## Features

- **Single‑portfolio simulator (GUI)**  
  - Dynamic stock universe: user‑defined tickers (e.g. `NVDA, AMD, INTC, MSFT, SPY`). 
  - Sun Tzu + chess‑inspired AI (phases, tension, center control, defense/attack/stability). 
  - Real‑time allocation and rebalancing, configurable number of turns and initial cash. 
  - JSON and CSV export of simulation runs for further analysis or RL. 

- **Multi‑agent market simulator (GUI)**  
  - Multiple investors with different risk profiles and personalities (bullish / bearish / sideways).
  - Socio‑economic model (unemployment, inflation, consumption index) influencing market tension. 
  - Sun Tzu + chess‑based decision logic per agent (buy/sell/hold, center control). 
  - JSON export of multi‑agent trajectories (logs, actions, socio‑economic states, market conditions). 

- **Core engine ready for RL / HPC**  
  - Core package `business_sim` exposes pure‑Python models: assets, socio‑economics, AI, portfolio dynamics.
  - Single‑step simulation function `simulate_single_turn` for environment‑style usage.
  - State‑vector and report export (`core_io`) for offline RL dataset generation. 
  - Clear separation between **core logic** and **Tkinter GUIs**, making it easy to run large batches on clusters without any GUI. 

***

## Repository structure

The repository is organized to separate the **business/market logic** from the **user interfaces**:


- `business_sim` is the **core library** (no GUI, no Tkinter). 
- `BusinessStrategicPortfolioGUI.py` is the **single‑portfolio GUI front‑end**.
- `BusinessMultiAgentMarketGUI.py` (wrapper around `business_sim/gui_multi.py`) is the **multi‑agent GUI front‑end**. 

***

## Single‑portfolio simulator

The single‑portfolio simulator focuses on **one business‑oriented portfolio** managed by an AI that blends:

- Sun Tzu’s *Art of War* principles (avoid unfavorable battles, defense vs attack, flexibility),  
- chess strategy concepts (opening/middlegame/endgame, center control, prophylaxis).

### Key elements

- **PortfolioState** (in `core_portfolio.py`)  
  - Holds the list of `AssetType` objects, current cash, and market conditions (fear, liquidity, volatility). 
  - Can be initialized with default assets (AAPL, GOOG, TSLA, MSFT, SPY) or a custom list of tickers. 

- **SunTzuChessMarketAI** (in `core_ai.py`)  
  - Tracks tension over time and decides on phases: opening, middlegame, endgame, stability. 
  - Evaluates center control (e.g. dominance on AAPL/GOOG/MSFT) and generates strategic recommendations.
  - Influences market conditions and portfolio allocation via attack/defense/stabilization patterns.

- **Simulation loop**  
  - GUI collects: number of turns, initial cash, target allocation, custom tickers. 
  - `simulate_single_turn` updates market conditions, applies AI, rebalances towards the (possibly AI‑adjusted) target allocation, and produces a `turn_result` dictionary with metrics and logs. 
  - Results are appended to `sim_data` and can be exported.

***

## Multi‑agent market simulator

The multi‑agent simulator models a **small market of heterogeneous investors** interacting in a common environment. 
### Key elements

- **SocioEconomicState** (in `core_market.py`)  
  - Tracks unemployment, inflation, and consumption index, updated stochastically each turn. 
  - Feeds into the AI’s risk tension computations, affecting agent behavior. 

- **MultiAgentInvestor** (in `core_portfolio.py`)  
  - Each investor has a name, risk aversion, cash, and their own `AssetType` positions. 
  - Uses `SunTzuChessMarketAI` to evaluate tension and center control, then decide whether to buy, sell, or hold.

- **MultiAgentPortfolio** (in `core_portfolio.py`)  
  - Manages the socio‑economic state and shared market conditions. 
  - Holds a list of investors and coordinates their actions at each turn. 
  - `one_turn` updates the macro environment, lets each agent take actions, and returns a structured record (logs, actions, socio‑economic metrics, total market value). 

- **MultiAgent GUI** (`business_sim/gui_multi.py` + wrapper)  
  - Lets you choose the number of turns, number of agents, and base risk aversion. 
  - Displays socio‑economic evolution and all AI/agent logs directly in the Tkinter interface. 
  - Exports the full multi‑agent history to JSON.

***

