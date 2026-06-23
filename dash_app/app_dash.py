# Author(s): Dr. Patrick Lemoine

import os
import sys
import webbrowser
import threading


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import yaml
import pandas as pd
import yfinance as yf

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from business_sim.dash_api import (
    init_portfolio_state_from_config,
    update_state_with_market,
    compute_allocation_vector,
    run_single_ai_turn,
    get_ai_snapshot, 
    run_ai_simulation,
    run_rl_simulation,
    explain_snapshot_result,
    explain_simulation_results,
)


from business_sim.dash_api import update_state_with_market

def compute_max_drawdown(series: pd.Series) -> float:
    """
    Compute max drawdown from a series of portfolio values.
    Returns a float (e.g. -0.25 for -25%).
    """
    if series.empty:
        return 0.0
    # Convert to float
    s = series.astype(float)
    running_max = s.cummax()
    drawdown = s / running_max - 1.0
    return float(drawdown.min())


# ---------- Load portfolio configuration ----------

with open("dash_app/portfolio_config.yaml", "r") as f:
    PORTFOLIO_CONFIG = yaml.safe_load(f)

POSITIONS = PORTFOLIO_CONFIG.get("positions", {})
TICKERS = sorted(list(POSITIONS.keys()))
BENCHMARK = PORTFOLIO_CONFIG.get("benchmark", "SPY")

ALL_TICKERS = TICKERS + [BENCHMARK]




# ---------- Fetch market prices via yfinance ----------

def fetch_prices(tickers, period="1y", interval="1d"):
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    # Handle the case where nothing is returned
    if data is None or data.empty:
        raise RuntimeError(f"No price data returned by yfinance for tickers={tickers}")

    # If columns are MultiIndex (typical case for multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        # Prefer 'Adj Close' if available
        level0 = {c[0] for c in data.columns}
        if "Adj Close" in level0:
            close = data["Adj Close"]
        elif "Close" in level0:
            close = data["Close"]
        else:
            # Fallback: use the first level as close
            first_level = list(level0)[0]
            close = data[first_level]
    else:
        # Single ticker: DataFrame with simple columns
        cols = list(data.columns)
        if "Adj Close" in cols:
            close = data["Adj Close"].to_frame()
        elif "Close" in cols:
            close = data["Close"].to_frame()
        else:
            # Fallback: use the first column as close
            close = data.iloc[:, 0].to_frame()

        # Ensure column name matches the ticker(s)
        close.columns = [tickers] if isinstance(tickers, str) else tickers

    close = close.dropna(how="all")
    if close.empty:
        raise RuntimeError(f"No valid close prices after cleaning for tickers={tickers}")

    return close


prices_df = fetch_prices(ALL_TICKERS)
prices_df.index.name = "date"


# ---------- Initialize portfolio state ----------

portfolio_state = init_portfolio_state_from_config(PORTFOLIO_CONFIG)
alloc_vector = compute_allocation_vector(PORTFOLIO_CONFIG, portfolio_state)


# ---------- Dash app ----------

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Sun Tzu Strategic Portfolio Dashboard"

app.layout = html.Div([
    html.H1("Sun Tzu: Strategic Portfolio Dashboard"),

    dcc.Store(id="prices-store", data=prices_df.reset_index().to_dict("records")),

    dcc.Tabs(
        id="tabs",
        value="tab-live",
        children=[
            dcc.Tab(label="Live & AI Snapshot", value="tab-live"),
            dcc.Tab(label="AI Simulation (N turns)", value="tab-sim"),
        ],
    ),

    html.Div(id="tabs-content"),
])


def layout_live_tab():
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Ticker"),
                dcc.Dropdown(
                    id="ticker-dropdown",
                    options=[{"label": t, "value": t} for t in TICKERS],
                    value=TICKERS[0] if TICKERS else None,
                    clearable=False,
                ),
                dcc.Graph(id="price-chart"),
                dcc.Graph(id="portfolio-chart"),
            ], style={"width": "65%", "display": "inline-block"}),

            html.Div([
                html.H3("Sun Tzu / Chess AI"),

                html.Button("Analyze (1 turn)", id="ai-run-button", n_clicks=0),
                html.Div(id="ai-last-run"),

                html.Br(),
                html.Div(id="ai-phase"),
                html.Div(id="ai-tension"),
                html.Div(id="ai-center-control"),
                html.Div(id="ai-personality"),
                html.Div(id="ai-posture"),

                html.H4("Tactics"),
                html.Ul(id="ai-tactics"),
                
                html.H4("AI Explanation"),
                html.Pre(id="ai-explanation", style={"whiteSpace": "pre-wrap"}),
            ], style={
                "width": "30%",
                "display": "inline-block",
                "verticalAlign": "top",
                "marginLeft": "3%",
            }),
        ]),
    ])


def layout_sim_tab():
    return html.Div([
        html.H3("AI Simulation (N turns)"),

        html.Div([
            html.Label("Number of turns"),
            dcc.Slider(
                id="sim-horizon-slider",
                min=5,
                max=60,
                step=5,
                value=20,
                marks={i: str(i) for i in range(5, 65, 10)},
            ),

            html.Br(),
            html.Label("Simulation mode"),
            dcc.RadioItems(
                id="sim-mode-radio",
                options=[
                    {"label": "Sun Tzu / Chess AI", "value": "suntzu"},
                    {"label": "RL (PPO)", "value": "rl"},
                ],
                value="suntzu",
                labelStyle={"display": "inline-block", "marginRight": "10px"},
            ),

            html.Br(),
            html.Button("Run Simulation", id="sim-run-button", n_clicks=0),
            html.Div(id="sim-last-run"),
            
            html.H4("Simulation summary"),
            html.Pre(id="sim-explanation", style={"whiteSpace": "pre-wrap"}),

        ], style={"marginBottom": "20px"}),

        dcc.Graph(id="sim-portfolio-chart"),
        dcc.Graph(id="sim-tension-chart"),
    ])


@app.callback(
    Output("tabs-content", "children"),
    Input("tabs", "value"),
)
def render_tabs(tab_value):
    if tab_value == "tab-sim":
        return layout_sim_tab()
    # default: live view
    return layout_live_tab()


@app.callback(
    [
        Output("sim-portfolio-chart", "figure"),
        Output("sim-tension-chart", "figure"),
        Output("sim-last-run", "children"),
        Output("sim-explanation", "children"), 
    ],
    Input("sim-run-button", "n_clicks"),
    State("sim-horizon-slider", "value"),
    State("sim-mode-radio", "value"),
    State("prices-store", "data"),
    prevent_initial_call=True,
)
def run_simulation(n_clicks, n_turns, mode, prices_data):
    if not n_clicks:
        empty_fig = go.Figure()
        return empty_fig, empty_fig, ""

    if not prices_data:
        msg = "Error: no price data available for simulation."
        empty_fig = go.Figure()
        return empty_fig, empty_fig, msg

    # Rebuild prices DataFrame
    df = pd.DataFrame(prices_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    latest_row = df.iloc[-1]

    # Build simulation results depending on mode
    try:
        if mode == "suntzu":
            cash = float(PORTFOLIO_CONFIG.get("cash", 0.0))
            state = init_portfolio_state_from_config({"positions": POSITIONS, "cash": cash})
            state = update_state_with_market(state, latest_row, POSITIONS, cash)
            alloc_vector = compute_allocation_vector({}, state)
            sim_results = run_ai_simulation(state, alloc_vector, n_turns)
            sim_df = pd.DataFrame(sim_results)
            x_col = "turn"
            label_prefix = "Sun Tzu"
        else:  # mode == "rl"
            env_config = {
                "tickers": list(POSITIONS.keys()),
                "initial_cash": PORTFOLIO_CONFIG.get("cash", 0.0),
            }
            model_path = "models/ppo_portfolio.zip"
            from business_sim.dash_api import run_rl_simulation
            sim_results = run_rl_simulation(model_path, env_config, n_turns)
            sim_df = pd.DataFrame(sim_results)
            x_col = "step"
            label_prefix = "RL (PPO)"
    except Exception as e:
        msg = f"Error while running simulation ({mode}): {e}"
        empty_fig = go.Figure()
        return empty_fig, empty_fig, msg

    if sim_df.empty:
        empty_fig = go.Figure()
        return empty_fig, empty_fig, "Simulation returned no results."

    # ---------- Portfolio value figure ----------
    fig_port = go.Figure()
    fig_port.add_trace(go.Scatter(
        x=sim_df[x_col],
        y=sim_df["portfolio_value"],
        mode="lines+markers",
        name=f"{label_prefix} portfolio value",
    ))

    # Sun Tzu benchmark: simulated market index if available
    if mode == "suntzu" and "market_index_end" in sim_df.columns:
        idx_series = sim_df["market_index_end"].astype(float)
        idx_norm = idx_series / idx_series.iloc[0] * sim_df["portfolio_value"].iloc[0]
        fig_port.add_trace(go.Scatter(
            x=sim_df[x_col],
            y=idx_norm,
            mode="lines+markers",
            name="Simulated market index (normalized)",
            line=dict(dash="dash"),
        ))

    fig_port.update_layout(
        title=f"{label_prefix} simulation over {len(sim_df)} steps",
        xaxis_title="Step" if mode == "rl" else "Turn",
        yaxis_title="Value",
    )

    # ---------- Tension / risk_tension figure ----------
    fig_tension = go.Figure()
    if mode == "suntzu" and "risk_tension" in sim_df.columns:
        fig_tension.add_trace(go.Scatter(
            x=sim_df[x_col],
            y=sim_df["risk_tension"],
            mode="lines+markers",
            name="Risk tension",
        ))
        fig_tension.update_layout(
            title="Strategic tension (Sun Tzu AI)",
            xaxis_title="Turn",
            yaxis_title="Risk tension",
        )
    elif mode == "rl" and "reward" in sim_df.columns:
        fig_tension.add_trace(go.Scatter(
            x=sim_df[x_col],
            y=sim_df["reward"],
            mode="lines+markers",
            name="Reward",
        ))
        fig_tension.update_layout(
            title="RL reward per step (PPO)",
            xaxis_title="Step",
            yaxis_title="Reward",
        )
    else:
        fig_tension.update_layout(
            title="No secondary metric to display",
        )

    # ---------- Stats ----------
    n_eff = len(sim_df)
    beat_series = sim_df.get("beat_market", pd.Series([False] * n_eff))
    n_beat = int(beat_series.astype(bool).sum()) if not beat_series.empty else 0
    mdd = compute_max_drawdown(sim_df["portfolio_value"])

    last_msg = (
        f"Simulation ({mode}) completed: {n_eff} steps, "
        f"final value={sim_df['portfolio_value'].iloc[-1]:.2f}, "
        f"steps beating market={n_beat}, "
        f"max drawdown={mdd * 100:.2f}%"
    )
    
    if mode == "suntzu":
        explanation = explain_simulation_results(sim_df)
    else:
        explanation = "RL (PPO) simulation: summary explanation not implemented yet."


    return fig_port, fig_tension, last_msg, explanation


# ---------- Callbacks ----------

@app.callback(
    Output("price-chart", "figure"),
    Input("ticker-dropdown", "value"),
    State("prices-store", "data"),
)
def update_price_chart(ticker, prices_data):
    df = pd.DataFrame(prices_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    if ticker is None or ticker not in df.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[ticker],
        mode="lines",
        name=f"{ticker} price",
    ))
    fig.update_layout(
        title=f"{ticker} - historical price",
        xaxis_title="Date",
        yaxis_title="Price",
    )
    return fig


@app.callback(
    Output("portfolio-chart", "figure"),
    Input("ticker-dropdown", "value"),
    State("prices-store", "data"),
)
def update_portfolio_chart(_ticker, prices_data):
    df = pd.DataFrame(prices_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Portfolio value: sum(quantity_i * price_i) + cash
    # (for this curve, we ignore cash and plot only the invested value)
    port_values = pd.Series(0.0, index=df.index)
    for t, info in POSITIONS.items():
        qty = float(info.get("quantity", 0.0))
        if t in df.columns:
            port_values += qty * df[t]

    # Benchmark
    bench_values = None
    if BENCHMARK in df.columns:
        bench_values = df[BENCHMARK]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=port_values,
        mode="lines",
        name="Portfolio",
    ))

    if bench_values is not None and len(bench_values) > 0 and len(port_values) > 0:
        # Normalize benchmark to start at the same value as the portfolio
        fig.add_trace(go.Scatter(
            x=df.index,
            y=bench_values / bench_values.iloc[0] * port_values.iloc[0],
            mode="lines",
            name=f"Benchmark ({BENCHMARK}) - normalized",
            line=dict(dash="dash"),
        ))

    fig.update_layout(
        title="Portfolio value vs benchmark",
        xaxis_title="Date",
        yaxis_title="Value",
    )
    return fig


@app.callback(
    [
        Output("ai-last-run", "children"),
        Output("ai-phase", "children"),
        Output("ai-tension", "children"),
        Output("ai-center-control", "children"),
        Output("ai-personality", "children"),
        Output("ai-posture", "children"),
        Output("ai-tactics", "children"),
        Output("ai-explanation", "children"),
    ],
    Input("ai-run-button", "n_clicks"),
    State("prices-store", "data"),
    prevent_initial_call=True,
)
def run_ai_turn(n_clicks, prices_data):
    # No click yet
    if not n_clicks:
        return ["", "", "", "", "", "", []]

    # Defensive: check that we have prices
    if not prices_data:
        return (
            "Error: no price data available.",
            "", "", "", "", "", [],
        )

    # Rebuild prices DataFrame from the Store
    df = pd.DataFrame(prices_data)
    if "date" not in df.columns or len(df) == 0:
        return (
            "Error: invalid or empty price data.",
            "", "", "", "", "", [],
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Get cash from config (default to 0.0 if missing)
    try:
        cash = float(PORTFOLIO_CONFIG.get("cash", 0.0))
    except Exception:
        cash = 0.0

    # Build a fresh PortfolioState from current market data and positions
    try:
        state, result = get_ai_snapshot(df, POSITIONS, cash)
    except Exception as e:
        # En debug, tu peux aussi faire: raise
        return (
            f"Error while running AI snapshot: {e}",
            "", "", "", "", "", [],
        )

    # Extract fields with robust defaults
    turn = result.get("turn", 1)
    phase = result.get("ai_phase", "Unknown")
    tension = result.get("risk_tension", 0.0)
    center_control = result.get("center_control", 0.0)
    personality = result.get("ai_personality", "Unknown")
    tactics = result.get("tactics", [])
    portfolio_value = result.get("portfolio_value", 0.0)
    result_cash = result.get("cash", cash)

    # Ensure numeric types for formatting
    try:
        portfolio_value = float(portfolio_value)
    except Exception:
        portfolio_value = 0.0

    try:
        result_cash = float(result_cash)
    except Exception:
        result_cash = 0.0

    try:
        tension_float = float(tension)
    except Exception:
        tension_float = 0.0

    try:
        center_control_float = float(center_control)
    except Exception:
        center_control_float = 0.0

    # Derive a simple "posture" from tension
    if tension_float > 0.7:
        posture = "Attack"
    elif tension_float < 0.3:
        posture = "Defense"
    else:
        posture = "Stabilization"

    tactics_li = [html.Li(str(t)) for t in tactics]
    
    explanation = explain_snapshot_result(result)

    return (
        f"Last analysis: turn {turn} - value={portfolio_value:.2f}, cash={result_cash:.2f}",
        f"Phase: {phase}",
        f"Tension (risk_tension): {tension_float:.3f}",
        f"Center control: {center_control_float:.3f}",
        f"AI personality: {personality}",
        f"Suggested posture: {posture}",
        tactics_li,
        explanation,
    )

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

if __name__ == "__main__":
    #threading.Timer(1.0, open_browser).start()
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.0, open_browser).start()
    app.run(debug=True)
    
    
