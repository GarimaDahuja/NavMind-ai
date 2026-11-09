"""
NavMind AI - Robot Model
Implements differential-drive robot kinematics and state management.
"""

import math
import numpy as np


class Robot:
    """
    Differential-drive mobile robot model.

    State: (x, y, theta, velocity)
    Controls: discrete actions {forward, left, right, stop}
    """

    # Action constants
    ACTION_FORWARD = 0
    ACTION_TURN_LEFT = 1
    ACTION_TURN_RIGHT = 2
    ACTION_STOP = 3

    ACTION_NAMES = {
        ACTION_FORWARD: "Forward",
        ACTION_TURN_LEFT: "Turn Left",
        ACTION_TURN_RIGHT: "Turn Right",
        ACTION_STOP: "Stop",
    }

    def __init__(self, x: float, y: float, theta: float = 0.0, radius: float = 10.0):
        """
        Args:
            x: Initial x position (pixels)
            y: Initial y position (pixels)
            theta: Initial heading angle (radians)
            radius: Robot body radius (pixels)
        """
        self.x = x
        self.y = y
        self.theta = theta  # heading in radians
        self.radius = radius

        # Movement parameters
        self.linear_speed = 20.0    # pixels per step (forward)
        self.angular_speed = 0.2    # radians per step (turn)
        self.velocity = 0.0         # current speed scalar

        # Path tracking
        self.path: list[tuple[float, float]] = [(x, y)]
        self.total_distance = 0.0
        self.steps_taken = 0
        self.prev_action = Robot.ACTION_STOP

    def reset(self, x: float, y: float, theta: float = 0.0):
        """Reset robot to new start position."""
        self.x = x
        self.y = y
        self.theta = theta
        self.velocity = 0.0
        self.path = [(x, y)]
        self.total_distance = 0.0
        self.steps_taken = 0
        self.prev_action = Robot.ACTION_STOP

    def step(self, action: int) -> tuple[float, float]:
        """
        Execute one action step.

        Args:
            action: One of ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_STOP

        Returns:
            (new_x, new_y) position after action
        """
        old_x, old_y = self.x, self.y

        if action == Robot.ACTION_FORWARD:
            self.velocity = self.linear_speed
            self.x += self.velocity * math.cos(self.theta)
            self.y += self.velocity * math.sin(self.theta)
        elif action == Robot.ACTION_TURN_LEFT:
            self.theta -= self.angular_speed
            self.velocity = self.linear_speed * 0.5
            self.x += self.velocity * math.cos(self.theta)
            self.y += self.velocity * math.sin(self.theta)
        elif action == Robot.ACTION_TURN_RIGHT:
            self.theta += self.angular_speed
            self.velocity = self.linear_speed * 0.5
            self.x += self.velocity * math.cos(self.theta)
            self.y += self.velocity * math.sin(self.theta)
        elif action == Robot.ACTION_STOP:
            self.velocity = 0.0

        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # Track path
        dist = math.hypot(self.x - old_x, self.y - old_y)
        self.total_distance += dist
        self.steps_taken += 1
        self.prev_action = action

        if len(self.path) == 0 or dist > 2.0:
            self.path.append((self.x, self.y))

        return self.x, self.y

    def get_position(self) -> tuple[float, float]:
        return self.x, self.y

    def get_state_vector(self) -> np.ndarray:
        """Return (x, y, theta, velocity) as array."""
        return np.array([self.x, self.y, self.theta, self.velocity], dtype=np.float32)

    def distance_to(self, tx: float, ty: float) -> float:
        """Euclidean distance from robot center to target."""
        return math.hypot(tx - self.x, ty - self.y)

    def angle_to(self, tx: float, ty: float) -> float:
        """
        Relative angle from robot heading to target.
        Positive = target is to the right.
        """
        angle_to_target = math.atan2(ty - self.y, tx - self.x)
        rel_angle = angle_to_target - self.theta
        return math.atan2(math.sin(rel_angle), math.cos(rel_angle))

    def set_speed(self, speed_factor: float):
        """Scale robot speed. speed_factor in [0.1, 2.0]."""
        base_speed = 20.0
        self.linear_speed = base_speed * max(0.1, min(2.0, speed_factor))

    def __repr__(self):
        return (
            f"Robot(x={self.x:.1f}, y={self.y:.1f}, "
            f"theta={math.degrees(self.theta):.1f}°, v={self.velocity:.1f})"
        )

