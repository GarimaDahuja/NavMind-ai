"""
NavMind AI - Distance Sensors & Simulated LiDAR
Provides sensor models for the navigation environment.
"""

import math
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.obstacles import ObstacleManager


class DistanceSensorArray:
    """
    Simulates a set of directional distance sensors around the robot.
    
    Default configuration: front, front-left, front-right, left, right sensors.
    Can be extended to a full LiDAR ring.
    """

    # Predefined sensor configurations
    BASIC_5 = [0.0, -math.pi / 4, math.pi / 4, -math.pi / 2, math.pi / 2]
    LIDAR_8 = [i * math.pi / 4 for i in range(8)]
    LIDAR_16 = [i * math.pi / 8 for i in range(16)]
    LIDAR_36 = [i * math.pi / 18 for i in range(36)]

    def __init__(
        self,
        angles: list[float] = None,
        max_range: float = 150.0,
        noise_std: float = 0.0,
    ):
        """
        Args:
            angles: List of sensor angles relative to robot heading (radians).
                    0 = forward, negative = left, positive = right.
            max_range: Maximum detection range (pixels)
            noise_std: Standard deviation of Gaussian sensor noise (pixels)
        """
        self.angles = angles if angles is not None else self.BASIC_5
        self.max_range = max_range
        self.noise_std = noise_std
        self.readings: np.ndarray = np.full(len(self.angles), max_range, dtype=np.float32)

    @property
    def num_sensors(self) -> int:
        return len(self.angles)

    def update(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        obstacle_manager: "ObstacleManager",
    ) -> np.ndarray:
        """
        Update all sensor readings.

        Args:
            robot_x, robot_y: Robot position
            robot_theta: Robot heading (radians)
            obstacle_manager: Obstacle manager with ray-cast capability

        Returns:
            Array of sensor readings (normalized to [0, 1])
        """
        for i, rel_angle in enumerate(self.angles):
            global_angle = robot_theta + rel_angle
            dist = obstacle_manager.get_sensor_reading(
                robot_x, robot_y, global_angle, self.max_range
            )
            # Add noise if specified
            if self.noise_std > 0:
                dist += np.random.normal(0, self.noise_std)
                dist = np.clip(dist, 0, self.max_range)
            self.readings[i] = dist

        return self.readings / self.max_range  # normalize to [0, 1]

    def get_front(self) -> float:
        """Return normalized front sensor reading."""
        return self.readings[0] / self.max_range

    def get_left(self) -> float:
        """Return normalized left sensor reading (index 3 in BASIC_5)."""
        if len(self.readings) > 3:
            return self.readings[3] / self.max_range
        return 1.0

    def get_right(self) -> float:
        """Return normalized right sensor reading (index 4 in BASIC_5)."""
        if len(self.readings) > 4:
            return self.readings[4] / self.max_range
        return 1.0

    def get_front_left(self) -> float:
        """Return normalized front-left sensor reading."""
        if len(self.readings) > 1:
            return self.readings[1] / self.max_range
        return 1.0

    def get_front_right(self) -> float:
        """Return normalized front-right sensor reading."""
        if len(self.readings) > 2:
            return self.readings[2] / self.max_range
        return 1.0

    def is_obstacle_ahead(self, threshold: float = 0.3) -> bool:
        """True if front sensor reads below threshold (obstacle close ahead)."""
        return self.get_front() < threshold

    def get_min_reading(self) -> float:
        """Return the minimum (closest obstacle) normalized reading."""
        return float(np.min(self.readings)) / self.max_range

    def get_readings_array(self) -> np.ndarray:
        """Return normalized readings as numpy array."""
        return self.readings / self.max_range

    def set_noise(self, noise_std: float):
        """Enable or update sensor noise level."""
        self.noise_std = noise_std

    def __repr__(self):
        readings_str = ", ".join(f"{r/self.max_range:.2f}" for r in self.readings)
        return f"SensorArray({self.num_sensors} sensors: [{readings_str}])"

