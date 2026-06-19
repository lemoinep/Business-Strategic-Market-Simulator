# PPO training with tqdm progress bar

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from tqdm import tqdm

from gym_portfolio_env import GymPortfolioEnv


def make_env():
    return GymPortfolioEnv(
        max_turns=30,
        tickers=["NVDA", "AMD", "MSFT", "SPY"],
        cash=20000,
    )


class TqdmCallback(BaseCallback):
    """
    SB3 callback that updates a tqdm bar on timesteps.
    """

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="PPO training", unit="step")

    def _on_step(self) -> bool:
        # self.num_timesteps is maintained by SB3
        if self.pbar is not None:
            self.pbar.n = self.num_timesteps
            self.pbar.refresh()
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.n = self.total_timesteps
            self.pbar.close()


if __name__ == "__main__":
    total_timesteps = 200_000

    print("Creating environment...")
    env = DummyVecEnv([make_env])

    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,             
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )

    callback = TqdmCallback(total_timesteps=total_timesteps)

    print("Training...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("ppo_portfolio")
    print("\nModel saved as 'ppo_portfolio'")

    # Quick test
    print("\nQuick test of one episode with PPO policy...")
    test_env = make_env()
    obs, _ = test_env.reset()
    done = False
    total_reward = 0.0
    step_count = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        total_reward += reward
        step_count += 1

    print(f"Number of steps:", step_count)
    print("Total reward test episode:", total_reward)