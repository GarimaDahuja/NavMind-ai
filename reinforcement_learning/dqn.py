"""
NavMind AI - Deep Q-Network (DQN) Agent
Implements DQN with experience replay, target network, and epsilon-greedy exploration.
"""
from __future__ import annotations

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Optional

from reinforcement_learning.replay_buffer import ReplayBuffer


class DQNNetwork(nn.Module):
    """
    Deep Q-Network architecture.

    Input:  observation vector (state_dim,)
    Output: Q-values for each discrete action (action_dim,)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Optional[list[int]] = None):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]

        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)

        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN: separates value and advantage streams.
    
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.action_dim = action_dim

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        # Combine: subtract mean advantage for identifiability
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


class DQNAgent:
    """
    DQN Agent for discrete robot navigation.

    Features:
    - Experience replay buffer
    - Target network with periodic update
    - Epsilon-greedy exploration with decay
    - Optional Dueling DQN architecture
    - GPU acceleration if available
    """

    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        target_update_freq: int = 10,  # episodes
        dueling: bool = True,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        if dueling:
            self.policy_net = DuelingDQNNetwork(state_dim, action_dim).to(self.device)
            self.target_net = DuelingDQNNetwork(state_dim, action_dim).to(self.device)
        else:
            self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
            self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=200, gamma=0.5
        )

        # Replay buffer
        self.memory = ReplayBuffer(buffer_capacity)

        # Tracking
        self.steps_done = 0
        self.episodes_done = 0
        self.losses: list[float] = []
        self.q_values: list[float] = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: Current observation
            training: If False, always pick greedy action

        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals = self.policy_net(state_t)
            self.q_values.append(q_vals.max().item())
            return int(q_vals.argmax().item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
        self.steps_done += 1

    def learn(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient update.

        Returns:
            Loss value or None if buffer not ready.
        """
        if not self.memory.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # Target Q-values (Double DQN: use policy net to select, target net to evaluate)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        # Loss
        loss = nn.SmoothL1Loss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes_done += 1

    def update_target_network(self):
        """Hard-copy policy net weights to target net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update_target(self, tau: float = 0.005):
        """Soft update: target = tau * policy + (1-tau) * target."""
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(tau * pp.data + (1 - tau) * tp.data)

    def save(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps_done": self.steps_done,
                "episodes_done": self.episodes_done,
                "losses": self.losses[-1000:],
            },
            path,
        )
        print(f"[DQN] Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.steps_done = checkpoint.get("steps_done", 0)
        self.episodes_done = checkpoint.get("episodes_done", 0)
        self.losses = checkpoint.get("losses", [])
        print(f"[DQN] Model loaded from {path}")

    def get_stats(self) -> dict:
        """Return current training statistics."""
        return {
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "episodes_done": self.episodes_done,
            "buffer_size": len(self.memory),
            "avg_loss": np.mean(self.losses[-100:]) if self.losses else 0.0,
            "avg_q": np.mean(self.q_values[-100:]) if self.q_values else 0.0,
            "device": str(self.device),
        }

    def __repr__(self):
        return (
            f"DQNAgent(state={self.state_dim}, actions={self.action_dim}, "
            f"epsilon={self.epsilon:.3f}, device={self.device})"
        )

