"""
NavMind AI - Random Map Generator
Generates diverse navigation environments for training and testing.
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional

from environment.obstacles import Obstacle, ObstacleManager


@dataclass
class MapConfig:
    """Configuration for map generation."""
    width: int = 600
    height: int = 600
    num_obstacles: int = 8
    min_obstacle_size: int = 20
    max_obstacle_size: int = 60
    clear_radius: float = 70.0
    dynamic_fraction: float = 0.0
    maze_like: bool = False
    seed: Optional[int] = None


class MapGenerator:
    """
    Generates 2D navigation maps with configurable complexity.
    
    Supported layouts:
    - Random: Randomly placed rectangular obstacles
    - Maze-like: Corridor-based maze structure
    - Custom: User-defined obstacle placement
    """

    DIFFICULTY_CONFIGS = {
        "easy": MapConfig(num_obstacles=4, max_obstacle_size=40),
        "medium": MapConfig(num_obstacles=8, max_obstacle_size=55),
        "hard": MapConfig(num_obstacles=12, max_obstacle_size=65),
        "dynamic": MapConfig(num_obstacles=8, dynamic_fraction=0.3),
        "maze": MapConfig(num_obstacles=0, maze_like=True),
    }

    def __init__(self, width: int = 600, height: int = 600):
        self.width = width
        self.height = height

    def generate(
        self,
        difficulty: str = "medium",
        seed: Optional[int] = None,
        robot_pos: Optional[tuple] = None,
        goal_pos: Optional[tuple] = None,
    ) -> tuple[ObstacleManager, tuple[float, float], tuple[float, float]]:
        """
        Generate a complete map including obstacles, robot start, and goal.

        Args:
            difficulty: One of "easy", "medium", "hard", "dynamic", "maze"
            seed: Random seed for reproducibility
            robot_pos: Override start position
            goal_pos: Override goal position

        Returns:
            (obstacle_manager, robot_start, goal_position)
        """
        config = self.DIFFICULTY_CONFIGS.get(difficulty, MapConfig())
        config.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        manager = ObstacleManager(self.width, self.height)
        manager.add_border_walls()

        # Generate start and goal positions
        margin = 60
        if robot_pos is None:
            robot_pos = self._random_free_position(margin)
        if goal_pos is None:
            goal_pos = self._random_goal_far_from(robot_pos, margin, min_dist=200)

        # Generate obstacles
        if config.maze_like:
            self._generate_maze(manager, robot_pos, goal_pos)
        else:
            manager.generate_random_obstacles(
                count=config.num_obstacles,
                min_size=config.min_obstacle_size,
                max_size=config.max_obstacle_size,
                robot_pos=robot_pos,
                goal_pos=goal_pos,
                clear_radius=config.clear_radius,
                dynamic_fraction=config.dynamic_fraction,
            )

        return manager, robot_pos, goal_pos

    def _random_free_position(self, margin: int) -> tuple[float, float]:
        """Get a random position within the environment boundaries."""
        x = random.uniform(margin, self.width - margin)
        y = random.uniform(margin, self.height - margin)
        return (x, y)

    def _random_goal_far_from(
        self,
        origin: tuple[float, float],
        margin: int,
        min_dist: float = 150.0,
    ) -> tuple[float, float]:
        """Get a goal position that is at least min_dist from origin."""
        for _ in range(100):
            x = random.uniform(margin, self.width - margin)
            y = random.uniform(margin, self.height - margin)
            if math.hypot(x - origin[0], y - origin[1]) >= min_dist:
                return (x, y)
        # Fallback: place goal at opposite corner
        return (self.width - margin, self.height - margin)

    def _generate_maze(
        self,
        manager: ObstacleManager,
        robot_pos: tuple,
        goal_pos: tuple,
    ):
        """Generate a simple maze-like layout with corridors."""
        corridor_width = 80
        wall_thickness = 15
        margin = 20

        # Horizontal walls at regular intervals
        num_walls = 4
        y_positions = np.linspace(margin + 80, self.height - margin - 80, num_walls)

        for i, wy in enumerate(y_positions):
            # Left half wall with gap on right
            if i % 2 == 0:
                wall_w = int(self.width * 0.6)
                x_start = margin
            else:
                # Right half wall with gap on left
                wall_w = int(self.width * 0.6)
                x_start = self.width - margin - wall_w

            obs = Obstacle(x_start, int(wy), wall_w, wall_thickness)
            # Check clearance from start and goal
            cx, cy = obs.center
            if (math.hypot(cx - robot_pos[0], cy - robot_pos[1]) > 60
                    and math.hypot(cx - goal_pos[0], cy - goal_pos[1]) > 60):
                manager.add_obstacle(obs)

    def generate_from_grid(
        self,
        grid: np.ndarray,
        cell_size: int = 30,
    ) -> ObstacleManager:
        """
        Generate environment from a binary grid (1 = obstacle, 0 = free).

        Args:
            grid: 2D numpy array
            cell_size: Size of each grid cell in pixels

        Returns:
            Configured ObstacleManager
        """
        rows, cols = grid.shape
        manager = ObstacleManager(cols * cell_size, rows * cell_size)
        manager.add_border_walls()

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 1:
                    manager.add_obstacle(
                        Obstacle(
                            c * cell_size,
                            r * cell_size,
                            cell_size,
                            cell_size,
                        )
                    )
        return manager

    @staticmethod
    def get_predefined_maps() -> dict[str, np.ndarray]:
        """
        Return a set of hand-crafted test maps as binary grids.
        0 = free, 1 = obstacle (10x10 grids, scaled to env size)
        """
        return {
            "open_field": np.zeros((10, 10), dtype=int),
            "corridor": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ], dtype=int),
            "cluttered": np.array([
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ], dtype=int),
        }

