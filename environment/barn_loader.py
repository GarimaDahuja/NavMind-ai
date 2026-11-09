"""
NavMind AI - BARN Dataset & Benchmark Map Loader
Loads and parses occupancy grid maps from the BARN (Benchmark for Autonomous Robot Navigation)
dataset, as well as generated synthetic benchmark environments.

Reference:
Perille et al., "Benchmarking Reinforcement Learning Techniques for Autonomous Navigation
in Challenging Environments", ICRA 2022.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from environment.obstacles import Obstacle, ObstacleManager


# Built-in BARN-style benchmark maps (sampled representing easy, medium, hard, and dense obstacle fields)
# 1 = obstacle, 0 = free space
BARN_SAMPLE_MAPS = {
    "barn_world_001_easy": np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8),

    "barn_world_050_corridor": np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8),

    "barn_world_120_dense_pillars": np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8),

    "barn_world_200_complex_maze": np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8),
}


class BARNLoader:
    """
    Loads BARN occupancy grid maps or benchmark arrays into NavMind AI's ObstacleManager.
    """

    def __init__(self, env_width: int = 600, env_height: int = 600):
        self.env_width = env_width
        self.env_height = env_height

    def get_benchmark_names(self) -> list[str]:
        """Return names of all available built-in BARN benchmark maps."""
        return list(BARN_SAMPLE_MAPS.keys())

    def load_grid_to_environment(
        self,
        grid: np.ndarray,
        robot_pos: Optional[Tuple[float, float]] = None,
        goal_pos: Optional[Tuple[float, float]] = None,
    ) -> Tuple[ObstacleManager, Tuple[float, float], Tuple[float, float]]:
        """
        Converts a binary occupancy grid (2D numpy array) into an ObstacleManager instance.

        Args:
            grid: 2D numpy array with 1 for obstacles and 0 for free space
            robot_pos: Starting (x, y) coordinates. Defaults to top-left free cell.
            goal_pos: Goal (x, y) coordinates. Defaults to bottom-right free cell.

        Returns:
            Tuple of (ObstacleManager, robot_pos, goal_pos)
        """
        rows, cols = grid.shape
        cell_w = self.env_width / cols
        cell_h = self.env_height / rows

        manager = ObstacleManager(self.env_width, self.env_height)
        manager.add_border_walls()

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 1:
                    x = c * cell_w
                    y = r * cell_h
                    manager.add_obstacle(Obstacle(x, y, cell_w, cell_h, color=(90, 110, 160)))

        # Default start and goal if not provided
        if robot_pos is None:
            robot_pos = (cell_w * 0.5, cell_h * 0.5)
        if goal_pos is None:
            goal_pos = (self.env_width - cell_w * 0.5, self.env_height - cell_h * 0.5)

        return manager, robot_pos, goal_pos

    def load_preset(
        self,
        preset_name: str,
    ) -> Tuple[ObstacleManager, Tuple[float, float], Tuple[float, float]]:
        """
        Load one of the predefined BARN benchmark environments.
        """
        if preset_name not in BARN_SAMPLE_MAPS:
            raise ValueError(f"Unknown preset {preset_name}. Available: {self.get_benchmark_names()}")

        grid = BARN_SAMPLE_MAPS[preset_name]
        return self.load_grid_to_environment(grid)

    def load_from_file(
        self,
        filepath: str,
    ) -> Tuple[ObstacleManager, Tuple[float, float], Tuple[float, float]]:
        """
        Load an occupancy grid from a text file, .npy array, or CSV.
        """
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Map file {filepath} not found.")

        if p.suffix == ".npy":
            grid = np.load(p)
        else:
            # Assume text/csv format of 0s and 1s
            grid = np.loadtxt(p, delimiter="," if "," in p.read_text() else None, dtype=np.uint8)

        return self.load_grid_to_environment(grid)

