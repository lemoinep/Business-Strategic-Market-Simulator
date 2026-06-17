import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
import json

from .core_portfolio import MultiAgentPortfolio


class MultiAgentPortfolioSimulatorGUI:
    LOG_COLORS = {
        "info": "black",
        "agent": "purple",
        "trade": "green",
        "socio": "brown",
        "event": "blue",
    }

    def __init__(self, root):
        self.root = root
        self.root.title(
            "Multi-Agent Sun Tzu Chess + Socio-Economic Portfolio Simulator"
        )

        self.state = MultiAgentPortfolio()

        self.log_text = ScrolledText(
            root, state="disabled", width=120, height=20, wrap="word"
        )
        self.log_text.pack(padx=10, pady=5)

        control_frame = tk.Frame(root)
        control_frame.pack(pady=5)

        tk.Label(control_frame, text="Number of Turns:").grid(row=0, column=0, padx=5)
        self.turns_var = tk.IntVar(value=12)
        self.turns_entry = tk.Entry(
            control_frame, width=5, textvariable=self.turns_var
        )
        self.turns_entry.grid(row=0, column=1)

        self.run_button = tk.Button(
            control_frame, text="Run Simulation", command=self.run_simulation
        )
        self.run_button.grid(row=0, column=2, padx=5)

        self.export_button = tk.Button(
            control_frame,
            text="Export Report",
            command=self.export_report,
            state="disabled",
        )
        self.export_button.grid(row=0, column=3, padx=5)

    def log(self, message, event_type="info"):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message + "\n", event_type)
        self.log_text.tag_config(
            event_type, foreground=self.LOG_COLORS.get(event_type, "black")
        )
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")
        self.root.update_idletasks()

    def run_simulation(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

        self.log(
            "=== Multi-Agent + Sun Tzu Chess + Socio-Economic Simulation Started ===",
            event_type="event",
        )

        try:
            turns = int(self.turns_var.get())
            assert turns > 0
        except Exception:
            messagebox.showerror("Error", "Invalid number of turns.")
            return

        self.state = MultiAgentPortfolio()

        for k in range(turns):
            turn_idx = k + 1
            turn_record = self.state.one_turn(turn_idx)

            socio = turn_record["socio"]
            self.log(
                f"Turn {turn_idx} - Socio: "
                f"unemployment={socio['unemployment_rate']:.3f}, "
                f"inflation={socio['inflation_rate']:.3f}, "
                f"consumption_index={socio['consumption_index']:.2f}",
                event_type="socio",
            )

            for log_msg in turn_record["logs"]:
                self.log(log_msg, event_type="agent")

            self.log(
                f"\nTotal market value: ${turn_record['total_market_value']:.2f}",
                event_type="event",
            )

        self.log("\n=== Simulation Completed ===", event_type="event")
        self.export_button.config(state="normal")

    def export_report(self):
        if not self.state.history:
            messagebox.showwarning("No Data", "No data to export.")
            return

        filename = (
            "multiagent_portfolio_report_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".json"
        )
        try:
            with open(filename, "w") as f:
                json.dump(self.state.history, f, indent=2)
            self.log(f"Report exported to: {filename}", event_type="event")
            messagebox.showinfo(
                "Export Complete",
                f"Report saved as:\n{filename}",
            )
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export report:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiAgentPortfolioSimulatorGUI(root)
    root.mainloop()