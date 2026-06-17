# Author(s): Dr. Patrick Lemoine
# Sun Tzu The Art of Investment War in real time + Chess AI logic


import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
import json

from business_sim.core_portfolio import PortfolioState, simulate_single_turn


from business_sim.core_io import export_json, export_state_vectors_csv

"""
def export_report(self):
    if not self.sim_data:
        messagebox.showwarning("No Data", "No data to export.")
        return

    json_file = export_json(self.sim_data, prefix="portfolio_report")
    csv_file = "portfolio_states_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    export_state_vectors_csv(self.sim_data, csv_file)

    self.log(f"Report exported to: {json_file}", event_type="event")
    self.log(f"State vectors exported to: {csv_file}", event_type="event")
    messagebox.showinfo("Export Complete", f"JSON: {json_file}\nCSV: {csv_file}")
"""



class PortfolioSimulatorGUI:
    LOG_COLORS = {
        'info': 'black',
        'win': 'blue',
        'loss': 'red',
        'trade': 'green',
        'ai': 'purple',
        'event': 'brown',
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Sun Tzu The Art of Investment War in real time + Chess AI logic")

        self.state = PortfolioState.default()

        self.log_text = ScrolledText(root, state='disabled', width=110, height=20, wrap='word')
        self.log_text.pack(padx=10, pady=5)

        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)
        
        tk.Label(control_frame, text="Custom tickers (comma-separated):").grid(row=2, column=0)
        self.tickers_var = tk.StringVar(value="")
        self.tickers_entry = tk.Entry(control_frame, width=30, textvariable=self.tickers_var)
        self.tickers_entry.grid(row=2, column=1, columnspan=3, sticky="w")

        tk.Label(control_frame, text="Number of Turns:").grid(row=0, column=0, padx=5)
        self.turns_var = tk.IntVar(value=12)
        self.turns_entry = tk.Entry(control_frame, width=5, textvariable=self.turns_var)
        self.turns_entry.grid(row=0, column=1)

        tk.Label(control_frame, text="Allocation % (AAPL/GOOG/TSLA/MSFT/SPY):").grid(row=1, column=0)
        self.alloc_var = tk.StringVar(value="30/20/10/20/20")
        self.alloc_entry = tk.Entry(control_frame, width=20, textvariable=self.alloc_var)
        self.alloc_entry.grid(row=1, column=1)

        self.run_button = tk.Button(control_frame, text="Run Simulation", command=self.run_simulation)
        self.run_button.grid(row=0, column=2, padx=5)

        self.export_button = tk.Button(control_frame, text="Export Report", command=self.export_report, state='disabled')
        self.export_button.grid(row=0, column=3, padx=5)

        self.load_button = tk.Button(control_frame, text="Load Portfolio", command=self.load_portfolio)
        self.load_button.grid(row=1, column=2, padx=5)

        self.save_button = tk.Button(control_frame, text="Save Portfolio", command=self.save_portfolio)
        self.save_button.grid(row=1, column=3, padx=5)

        self.logs = []
        self.sim_data = []


    def log(self, message, event_type="info"):
        self.logs.append(message)
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, message + "\n", event_type)
        self.log_text.tag_config(event_type, foreground=self.LOG_COLORS.get(event_type, 'black'))
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
        self.root.update_idletasks()

    def parse_allocation(self, dist: str, n_assets: int) -> list[float]:
        try:
            raw = [float(x) for x in dist.strip().split('/')]
        except Exception:
            # fallback: répartition uniforme
            raw = [100.0 / n_assets] * n_assets

        if len(raw) < n_assets:
            raw = raw + [0.0] * (n_assets - len(raw))
        elif len(raw) > n_assets:
            raw = raw[:n_assets]

        total = sum(raw)
        if total <= 0:
            return [1.0 / n_assets] * n_assets

        return [x / total for x in raw]
        
    

    def portfolio_turn(self, alloc_percent):
        turn_num = len(self.sim_data) + 1
        result = simulate_single_turn(self.state, alloc_percent, turn_num)
        self._log_turn(result)
        self.sim_data.append(result)

    def _log_turn(self, result):
        """
        Affiche le résultat retourné par simulate_single_turn.
        On suppose que result contient au moins les clés:
        - turn, portfolio_value, cash,
        - ai_personality, ai_phase, risk_tension, center_control,
        - tactics (liste de chaînes).
        """
        self.log(f"\n--- Turn {result['turn']} ---", event_type="info")
        self.log(
            f"Total portfolio value: ${result['portfolio_value']:.2f} "
            f"(Cash: ${result['cash']:.2f})",
            event_type="info",
        )

        self.log(
            f"AI: personality={result['ai_personality']}, "
            f"phase={result['ai_phase']}, "
            f"risk_tension={result['risk_tension']:.3f}, "
            f"center_control={result['center_control']:.3f}",
            event_type="ai",
        )

        for tact in result.get("tactics", []):
            self.log("AI Strategy: " + tact, event_type="ai")


    def run_simulation(self):
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.configure(state='disabled')
        self.logs.clear()
        self.sim_data.clear()

        tickers_str = self.tickers_var.get().strip()
        if tickers_str:
            tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
            if len(tickers) == 0:
                self.state = PortfolioState.default()
            else:
                self.state = PortfolioState.from_tickers(tickers)
        else:
            self.state = PortfolioState.default()

        try:
            turns = int(self.turns_var.get())
            assert turns > 0
        except Exception:
            messagebox.showerror("Error", "Invalid number of turns.")
            return

        n_assets = len(self.state.assets)
        alloc_percent = self.parse_allocation(self.alloc_var.get(), n_assets)
        self.log("=== Starting Sun Tzu + Chess AI simulator ===", event_type="info")

        for _ in range(turns):
            self.portfolio_turn(alloc_percent)
            if self.state.total_value() <= 0:
                self.log("Portfolio lost. Simulation ended.", event_type="loss")
                break

        self.log("\n=== Simulation Completed ===", event_type="event")
        self.export_button.config(state='normal')


    def export_report(self):
        if not self.sim_data:
            messagebox.showwarning("No Data", "No data to export.")
            return

        filename = export_json(self.sim_data, prefix="portfolio_report")
        self.log(f"Report exported to: {filename}", event_type="event")
        messagebox.showinfo("Export Complete", f"Report saved as:\n{filename}")


    def load_portfolio(self):
        file = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file:
            try:
                self.state.load_portfolio_from_file(file)
                self.log(f"Portfolio loaded from {file}", event_type="event")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load portfolio:\n{e}")

    def save_portfolio(self):
        file = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON Files", "*.json")]
        )
        if file:
            try:
                self.state.save_portfolio_to_file(file)
                self.log(f"Portfolio saved to {file}", event_type="event")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save portfolio:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioSimulatorGUI(root)
    root.mainloop()