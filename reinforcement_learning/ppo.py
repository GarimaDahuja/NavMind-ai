"""
NavMind AI - PPO Agent (Proximal Policy Optimization)
Wraps Stable-Baselines3 PPO for the NavMind navigation environment.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Callable


class PPOAgent:
    """
    PPO Agent using Stable-Baselines3.

    Provides a consistent interface matching the DQNAgent API
    for easy algorithm comparison in experiments.
    """

    def __init__(
        self,
        env_id: str = "NavMind-v0",
        env_kwargs: Optional[dict] = None,
        lr: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy: str = "MlpPolicy",
        device: str = "auto",
        verbose: int = 1,
    ):
        """
        Args:
            env_id: Gymnasium environment ID
            env_kwargs: Additional kwargs for environment creation
            lr: Learning rate
            n_steps: Steps per rollout buffer
            batch_size: Mini-batch size for updates
            n_epochs: Optimization epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_range: PPO clipping parameter
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function loss coefficient
            max_grad_norm: Gradient clipping
            policy: Policy architecture ('MlpPolicy' or 'CnnPolicy')
            device: Compute device ('auto', 'cpu', 'cuda')
            verbose: Verbosity level
        """
        self.env_id = env_id
        self.env_kwargs = env_kwargs or {}
        self.hyperparams = {
            "learning_rate": lr,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "device": device,
            "verbose": verbose,
        }
        self.policy = policy
        self.model = None
        self._training_rewards: list[float] = []

    def create_env(self, n_envs: int = 1):
        """Create vectorized environment for PPO training."""
        from stable_baselines3.common.env_util import make_vec_env
        from environment.robot_env import RobotNavEnv

        env = make_vec_env(
            RobotNavEnv,
            n_envs=n_envs,
            env_kwargs=self.env_kwargs,
        )
        return env

    def build_model(self, env=None):
        """Initialize the SB3 PPO model."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

        if env is None:
            env = self.create_env()

        self.model = PPO(
            policy=self.policy,
            env=env,
            **self.hyperparams,
        )
        return self.model

    def train(
        self,
        total_timesteps: int = 200_000,
        eval_freq: int = 10_000,
        save_path: str = "models/ppo_navmind",
        callback=None,
    ):
        """
        Train the PPO agent.

        Args:
            total_timesteps: Total training steps
            eval_freq: Evaluation frequency
            save_path: Path to save checkpoints
            callback: Optional SB3 callback
        """
        from stable_baselines3.common.callbacks import (
            EvalCallback,
            CheckpointCallback,
            CallbackList,
        )
        from stable_baselines3.common.monitor import Monitor
        from environment.robot_env import RobotNavEnv

        if self.model is None:
            self.build_model()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Evaluation environment
        eval_env = Monitor(RobotNavEnv(**self.env_kwargs))
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(Path(save_path).parent),
            log_path=str(Path(save_path).parent / "logs"),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=eval_freq,
            save_path=str(Path(save_path).parent / "checkpoints"),
            name_prefix="ppo_navmind",
        )

        callbacks = [eval_callback, checkpoint_callback]
        if callback:
            callbacks.append(callback)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=False,
        )
        self.model.save(save_path)
        print(f"[PPO] Model saved to {save_path}")

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        """
        Select action for a given state.

        Args:
            state: Current observation
            deterministic: If True, use greedy policy (no exploration)

        Returns:
            Action index
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() or load() first.")
        action, _ = self.model.predict(state, deterministic=deterministic)
        return int(action)

    def save(self, path: str):
        """Save model to path."""
        if self.model:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path)
            print(f"[PPO] Saved to {path}")

    def load(self, path: str):
        """Load model from path."""
        from stable_baselines3 import PPO
        self.model = PPO.load(path)
        print(f"[PPO] Loaded from {path}")

    def get_stats(self) -> dict:
        """Return current agent statistics."""
        if self.model is None:
            return {}
        return {
            "algorithm": "PPO",
            "total_timesteps": self.model.num_timesteps,
            "device": str(self.model.device),
            "policy": self.policy,
            **{k: v for k, v in self.hyperparams.items() if k != "verbose"},
        }

