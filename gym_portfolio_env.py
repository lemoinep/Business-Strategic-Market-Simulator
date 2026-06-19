# Gymnasium wrapper for PortfolioEnv (compatible with recent SB3)

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from business_sim.core_portfolio import PortfolioEnv as CorePortfolioEnv


class GymPortfolioEnv(gym.Env):
    """
    Gymnasium wrapper around PortfolioEnv to use with stable-baselines3.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_turns=30, tickers=None, cash=10000):
        super().__init__()
        self.core_env = CorePortfolioEnv(
            max_turns=max_turns,
            tickers=tickers,
            cash=cash,
        )
        dummy_state = self.core_env.reset()
        self.state_dim = len(dummy_state)
        self.n_assets = len(self.core_env.state_obj.assets)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        self.state = np.array(dummy_state, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = self.core_env.reset()
        self.state = np.array(state, dtype=np.float32)
        return self.state, {}

    def step(self, action):
        action = np.clip(action, 0.0, 1.0)
        next_state, reward, done, info = self.core_env.step(action.tolist())
        self.state = np.array(next_state, dtype=np.float32)
        # Gymnasium: (obs, reward, terminated, truncated, info)
        return self.state, float(reward), bool(done), False, info

    def render(self):
        pass

    def close(self):
        pass