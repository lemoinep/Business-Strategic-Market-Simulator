# BusinessMultiAgentMarketGUI.py
# Wrapper to launch the multi-agent GUI based on business_sim.gui_multi

import tkinter as tk
from business_sim.gui_multi import MultiAgentPortfolioSimulatorGUI


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiAgentPortfolioSimulatorGUI(root)
    root.mainloop()
    
