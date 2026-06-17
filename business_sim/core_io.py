import json
import csv
from datetime import datetime
from typing import List, Dict, Any

import numpy as np 

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def export_json(data: List[Dict[str, Any]], prefix: str) -> str:
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    return filename


def build_state_vector(turn_result: Dict[str, Any]) -> List[float]:
    state = [
        turn_result["cash"],
        turn_result["portfolio_value"],
        turn_result["market_fear"],
        turn_result["market_liquidity"],
        turn_result["market_volatility"],
        turn_result["center_control"],
        turn_result["risk_tension"],
    ]
    return state


def export_state_vectors_csv(
    turn_results: List[Dict[str, Any]], path: str
):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "turn",
            "cash",
            "portfolio_value",
            "market_fear",
            "market_liquidity",
            "market_volatility",
            "center_control",
            "risk_tension",
        ])
        for tr in turn_results:
            state = build_state_vector(tr)
            writer.writerow([tr["turn"], *state])