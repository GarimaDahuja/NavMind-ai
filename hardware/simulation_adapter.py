"""
NavMind AI - Simulation Adapter
Provides a unified interface between simulation and hardware modes.
"""

import numpy as np
from typing import Optional, Callable


class SimulationAdapter:
    """
    Adapter layer between the simulation environment and agent.

    In simulation mode, this adapter directly accesses the Gymnasium environment.
    In hardware mode (Jetson Nano), it would be replaced by the JetsonAdapter.

    This design allows the same trained RL policy to work in both modes.
    """

    def __init__(self, env=None):
        """
        Args:
            env: RobotNavEnv gymnasium environment instance
        """
        self.env = env
        self._observation = None
        self._last_info = {}

    def set_env(self, env):
        """Set or replace the simulation environment."""
        self.env = env

    def get_observation(self) -> Optional[np.ndarray]:
        """
        Get current environment observation.

        Returns:
            Normalized observation vector [dist_goal, angle_goal, s1, s2, s3, s4, s5]
            or None if environment not set.
        """
        if self.env is None:
            return None
        return self.env._get_observation()

    def execute_action(self, action: int) -> tuple:
        """
        Execute an action in the simulation.

        Args:
            action: Discrete action index

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.env is None:
            raise RuntimeError("No environment set in SimulationAdapter")
        return self.env.step(action)

    def reset(self, seed: Optional[int] = None) -> tuple:
        """Reset the environment."""
        if self.env is None:
            raise RuntimeError("No environment set")
        return self.env.reset(seed=seed)

    def get_robot_state(self) -> dict:
        """
        Get current robot state.

        Returns:
            Dict with position, heading, velocity, sensor readings
        """
        if self.env is None:
            return {}
        robot = self.env.robot
        sensors = self.env.sensors
        return {
            "x": robot.x,
            "y": robot.y,
            "theta": robot.theta,
            "velocity": robot.velocity,
            "sensor_front": float(sensors.get_front()),
            "sensor_left": float(sensors.get_left()),
            "sensor_right": float(sensors.get_right()),
            "distance_to_goal": robot.distance_to(self.env.goal_x, self.env.goal_y),
            "angle_to_goal": robot.angle_to(self.env.goal_x, self.env.goal_y),
        }

    def set_goal(self, x: float, y: float):
        """Override the current goal position."""
        if self.env:
            self.env.goal_x = x
            self.env.goal_y = y

    def get_goal(self) -> tuple[float, float]:
        """Return current goal position."""
        if self.env:
            return self.env.goal_x, self.env.goal_y
        return 0.0, 0.0

    def render_frame(self) -> Optional[np.ndarray]:
        """Get rendered frame as numpy array (RGB)."""
        if self.env:
            return self.env.render()
        return None

    def is_simulation(self) -> bool:
        """Always True for simulation adapter."""
        return True

