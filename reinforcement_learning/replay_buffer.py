"""
NavMind AI - Experience Replay Buffer
Implements uniform and prioritized experience replay for DQN.
"""
from __future__ import annotations

import random
import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class Transition:
    """Single experience transition."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Uniform Experience Replay Buffer for DQN.

    Stores (s, a, r, s', done) transitions and samples random mini-batches
    to break temporal correlations during training.
    """

    def __init__(self, capacity: int = 50000):
        """
        Args:
            capacity: Maximum number of transitions to store.
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add a transition to the buffer."""
        self.buffer.append(Transition(
            state=np.array(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=bool(done),
        ))

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a random mini-batch.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays.
        """
        batch = random.sample(self.buffer, batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """True if buffer has enough samples for one mini-batch."""
        return len(self.buffer) >= batch_size

    def clear(self):
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer (PER).
    
    Samples transitions proportional to their TD error magnitude,
    giving more training attention to surprising or important experiences.
    """

    def __init__(self, capacity: int = 50000, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta: IS weight exponent (0 = no correction, 1 = full correction)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001

        self.buffer: list = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add transition with max priority."""
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = Transition(
            state=np.array(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=bool(done),
        )
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        """
        Sample proportionally to priorities.

        Returns:
            (states, actions, rewards, next_states, dones, indices, weights)
        """
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities after learning (using new TD errors)."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # small epsilon to avoid zero

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size

