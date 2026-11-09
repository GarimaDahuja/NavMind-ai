"""
NavMind AI - Obstacle Management
Handles static and dynamic obstacles in the navigation environment.
"""

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Obstacle:
    """A rectangular obstacle."""
    x: float          # top-left x
    y: float          # top-left y
    width: float
    height: float
    is_dynamic: bool = False
    dx: float = 0.0   # velocity x (for dynamic obstacles)
    dy: float = 0.0   # velocity y (for dynamic obstacles)
    color: tuple = field(default_factory=lambda: (100, 100, 180))

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def rect(self) -> tuple[float, float, float, float]:
        """(x, y, width, height)"""
        return self.x, self.y, self.width, self.height

    def contains_point(self, px: float, py: float, margin: float = 0.0) -> bool:
        """Check if point (px, py) is inside (or within margin of) this obstacle."""
        return (
            self.x - margin <= px <= self.x + self.width + margin
            and self.y - margin <= py <= self.y + self.height + margin
        )

    def distance_to_point(self, px: float, py: float) -> float:
        """Shortest distance from point to rectangle boundary."""
        cx, cy = self.center
        dx = max(abs(px - cx) - self.width / 2, 0)
        dy = max(abs(py - cy) - self.height / 2, 0)
        return math.hypot(dx, dy)

    def update(self, env_width: float, env_height: float):
        """Update position for dynamic obstacles (bouncing)."""
        if not self.is_dynamic:
            return
        self.x += self.dx
        self.y += self.dy
        # Bounce off environment boundaries
        if self.x <= 0 or self.x + self.width >= env_width:
            self.dx *= -1
        if self.y <= 0 or self.y + self.height >= env_height:
            self.dy *= -1
        self.x = max(0, min(env_width - self.width, self.x))
        self.y = max(0, min(env_height - self.height, self.y))


class ObstacleManager:
    """Manages all obstacles in the environment."""

    def __init__(self, env_width: int, env_height: int):
        self.env_width = env_width
        self.env_height = env_height
        self.obstacles: list[Obstacle] = []
        self.border_thickness = 10

    def clear(self):
        self.obstacles = []

    def add_obstacle(self, obstacle: Obstacle):
        self.obstacles.append(obstacle)

    def add_border_walls(self):
        """Add boundary walls around the environment."""
        t = self.border_thickness
        w, h = self.env_width, self.env_height
        walls = [
            Obstacle(0, 0, w, t, color=(60, 60, 80)),       # top
            Obstacle(0, h - t, w, t, color=(60, 60, 80)),   # bottom
            Obstacle(0, 0, t, h, color=(60, 60, 80)),       # left
            Obstacle(w - t, 0, t, h, color=(60, 60, 80)),  # right
        ]
        for wall in walls:
            self.obstacles.append(wall)

    def generate_random_obstacles(
        self,
        count: int,
        min_size: int = 20,
        max_size: int = 60,
        robot_pos: Optional[tuple] = None,
        goal_pos: Optional[tuple] = None,
        clear_radius: float = 60.0,
        dynamic_fraction: float = 0.0,
    ) -> list[Obstacle]:
        """
        Generate random non-overlapping obstacles, keeping start and goal clear.

        Args:
            count: Number of obstacles to generate
            min_size: Minimum obstacle dimension
            max_size: Maximum obstacle dimension
            robot_pos: (x, y) of robot start — kept clear
            goal_pos: (x, y) of goal — kept clear
            clear_radius: Radius around start/goal to keep obstacle-free
            dynamic_fraction: Fraction of obstacles that are dynamic [0, 1]
        """
        generated = []
        attempts = 0
        max_attempts = count * 20

        while len(generated) < count and attempts < max_attempts:
            attempts += 1
            margin = self.border_thickness + 5
            w = random.randint(min_size, max_size)
            h = random.randint(min_size, max_size)
            x = random.randint(margin, self.env_width - w - margin)
            y = random.randint(margin, self.env_height - h - margin)

            obs = Obstacle(x, y, w, h)
            cx, cy = obs.center

            # Check clearance from robot start
            if robot_pos:
                if math.hypot(cx - robot_pos[0], cy - robot_pos[1]) < clear_radius:
                    continue

            # Check clearance from goal
            if goal_pos:
                if math.hypot(cx - goal_pos[0], cy - goal_pos[1]) < clear_radius:
                    continue

            # Check no overlap with existing obstacles
            overlap = False
            for existing in generated:
                if (
                    abs(obs.center[0] - existing.center[0]) < (w + existing.width) / 2 + 10
                    and abs(obs.center[1] - existing.center[1]) < (h + existing.height) / 2 + 10
                ):
                    overlap = True
                    break
            if overlap:
                continue

            # Make some dynamic
            is_dynamic = random.random() < dynamic_fraction
            if is_dynamic:
                speed = random.uniform(0.5, 2.0)
                angle = random.uniform(0, 2 * math.pi)
                obs.is_dynamic = True
                obs.dx = speed * math.cos(angle)
                obs.dy = speed * math.sin(angle)
                obs.color = (180, 100, 100)

            generated.append(obs)
            self.obstacles.append(obs)

        return generated

    def check_collision(self, x: float, y: float, radius: float) -> bool:
        """
        Check if a circle (robot) at (x, y) with given radius collides with any obstacle.

        Returns:
            True if collision detected
        """
        for obs in self.obstacles:
            if obs.distance_to_point(x, y) < radius:
                return True
        return False

    def update_dynamic(self):
        """Update all dynamic obstacles for one time step."""
        for obs in self.obstacles:
            if obs.is_dynamic:
                obs.update(self.env_width, self.env_height)

    def get_sensor_reading(
        self,
        robot_x: float,
        robot_y: float,
        angle: float,
        max_range: float = 150.0,
        num_steps: int = 50,
    ) -> float:
        """
        Ray-cast a sensor beam from robot position in given angle.

        Args:
            robot_x, robot_y: Robot position
            angle: Beam angle (radians, global frame)
            max_range: Maximum sensor range
            num_steps: Ray-marching resolution

        Returns:
            Distance to nearest obstacle (or max_range if none)
        """
        step_size = max_range / num_steps
        for i in range(1, num_steps + 1):
            dist = i * step_size
            px = robot_x + dist * math.cos(angle)
            py = robot_y + dist * math.sin(angle)
            if px < 0 or px >= self.env_width or py < 0 or py >= self.env_height:
                return dist
            for obs in self.obstacles:
                if obs.contains_point(px, py):
                    return dist
        return max_range

