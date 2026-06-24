import os
import csv
from datetime import datetime
from typing import List, Dict, Any

#TRADE_HISTORY_FILE = "trade_history.csv" 
TRADE_HISTORY_FILE = os.path.join("data", "trade_history.csv") 


def append_trades_to_csv(
    actions: List[Dict[str, Any]],
    prices_row: Dict[str, float],
    result: Dict[str, Any],
    mode: str,
    csv_path: str = TRADE_HISTORY_FILE,
) -> None:
    """
    Append executed actions to a CSV trade history file.

    actions: list of {ticker, side, quantity, reason}
    prices_row: dict mapping ticker -> last price
    result: Sun Tzu snapshot result (for context: phase, tension, center_control)
    mode: 'auto' or 'manual'
    """
    file_exists = os.path.exists(csv_path)

    phase = result.get("ai_phase")
    tension = result.get("risk_tension")
    center_control = result.get("center_control")

    # Ensure directory exists if you use a nested path like "data/trade_history.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header only once
        if not file_exists:
            writer.writerow([
                "timestamp",
                "mode",
                "ticker",
                "side",
                "quantity",
                "price",
                "phase",
                "tension",
                "center_control",
                "reason",
            ])

        timestamp = datetime.utcnow().isoformat()

        for a in actions:
            ticker = a["ticker"]
            side = a["side"]
            quantity = int(a["quantity"])
            price = float(prices_row.get(ticker, 0.0))
            reason = a.get("reason", "")

            writer.writerow([
                timestamp,
                mode,
                ticker,
                side,
                quantity,
                price,
                phase,
                tension,
                center_control,
                reason,
            ])