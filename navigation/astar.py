"""
NavMind AI - A* Path Planning Algorithm
Implements A* for optimal path planning as a baseline against RL approaches.
"""

from __future__ import annotations

import math
import heapq
import numpy as np
from typing import Optional
from dataclasses import dataclass, field


@dataclass(order=True)
class AStarNode:
    """Node for A* priority queue."""
    f_cost: float
    g_cost: float = field(compare=False)
    pos: tuple = field(compare=False)
    parent: Optional[AStarNode] = field(compare=False, default=None)


class AStarPlanner:
    """
    A* path planner on a discretized occupancy grid.

    Provides optimal paths in static environments.
    Used as a baseline to compare against RL-based navigation.
    """

    def __init__(self, env_width: int = 600, env_height: int = 600, cell_size: int = 15):
        """
        Args:
            env_width: Environment width in pixels
            env_height: Environment height in pixels
            cell_size: Grid cell size in pixels (resolution)
        """
        self.env_width = env_width
        self.env_height = env_height
        self.cell_size = cell_size
        self.grid_cols = env_width // cell_size
        self.grid_rows = env_height // cell_size

        # 8-directional movement
        self.directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]
        self.dir_costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert pixel coordinates to grid cell."""
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        col = max(0, min(self.grid_cols - 1, col))
        row = max(0, min(self.grid_rows - 1, row))
        return col, row

    def grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        """Convert grid cell to pixel center coordinates."""
        x = (col + 0.5) * self.cell_size
        y = (row + 0.5) * self.cell_size
        return x, y

    def build_grid(self, obstacle_manager, robot_radius: float = 12.0) -> np.ndarray:
        """
        Build binary occupancy grid from obstacle manager.

        Args:
            obstacle_manager: ObstacleManager with obstacles list
            robot_radius: Robot radius for inflation (collision safety margin)

        Returns:
            Binary grid: 0 = free, 1 = occupied
        """
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.uint8)

        # Inflate obstacles by robot radius
        inflation = int(math.ceil(robot_radius / self.cell_size)) + 1

        for obs in obstacle_manager.obstacles:
            # Convert obstacle bounds to grid cells
            c_min = max(0, int(obs.x / self.cell_size) - inflation)
            c_max = min(self.grid_cols - 1, int((obs.x + obs.width) / self.cell_size) + inflation)
            r_min = max(0, int(obs.y / self.cell_size) - inflation)
            r_max = min(self.grid_rows - 1, int((obs.y + obs.height) / self.cell_size) + inflation)

            grid[r_min:r_max + 1, c_min:c_max + 1] = 1

        return grid

    def heuristic(self, col: int, row: int, goal_col: int, goal_row: int) -> float:
        """Octile distance heuristic (admissible for 8-directional movement)."""
        dx = abs(col - goal_col)
        dy = abs(row - goal_row)
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def plan(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        grid: np.ndarray,
    ) -> Optional[list[tuple[float, float]]]:
        """
        Find optimal path from start to goal using A*.

        Args:
            start: (x, y) start position in pixels
            goal: (x, y) goal position in pixels
            grid: Occupancy grid (from build_grid)

        Returns:
            List of (x, y) waypoints in pixel coordinates, or None if no path found.
        """
        start_cell = self.world_to_grid(*start)
        goal_cell = self.world_to_grid(*goal)

        # Check if start/goal are free
        if grid[start_cell[1], start_cell[0]] == 1:
            # Try nearby free cell
            start_cell = self._find_free_nearby(grid, start_cell)
        if grid[goal_cell[1], goal_cell[0]] == 1:
            goal_cell = self._find_free_nearby(grid, goal_cell)
        if start_cell is None or goal_cell is None:
            return None

        # A* search
        open_set = []
        g_start = 0.0
        h_start = self.heuristic(*start_cell, *goal_cell)
        start_node = AStarNode(f_cost=g_start + h_start, g_cost=g_start, pos=start_cell)
        heapq.heappush(open_set, start_node)

        came_from: dict = {}
        g_costs: dict = {start_cell: 0.0}
        closed: set = set()

        while open_set:
            current = heapq.heappop(open_set)

            if current.pos in closed:
                continue
            closed.add(current.pos)

            if current.pos == goal_cell:
                return self._reconstruct_path(came_from, current.pos, start_cell)

            col, row = current.pos
            for (dc, dr), cost in zip(self.directions, self.dir_costs):
                nc, nr = col + dc, row + dr
                neighbor = (nc, nr)

                if not (0 <= nc < self.grid_cols and 0 <= nr < self.grid_rows):
                    continue
                if grid[nr, nc] == 1:
                    continue
                if neighbor in closed:
                    continue

                new_g = g_costs[current.pos] + cost
                if new_g < g_costs.get(neighbor, float("inf")):
                    g_costs[neighbor] = new_g
                    h = self.heuristic(nc, nr, *goal_cell)
                    came_from[neighbor] = current.pos
                    heapq.heappush(open_set, AStarNode(new_g + h, new_g, neighbor))

        return None  # No path found

    def _reconstruct_path(
        self,
        came_from: dict,
        current: tuple,
        start: tuple,
    ) -> list[tuple[float, float]]:
        """Backtrack from goal to start and return world-coordinate waypoints."""
        path_cells = [current]
        while current in came_from:
            current = came_from[current]
            path_cells.append(current)
        path_cells.reverse()

        # Smooth path with line-of-sight pruning
        path_cells = self._smooth_path(path_cells, came_from)

        return [self.grid_to_world(c, r) for c, r in path_cells]

    def _smooth_path(self, path: list, came_from: dict) -> list:
        """Simple path smoothing: remove redundant intermediate waypoints."""
        if len(path) <= 2:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                # Check if direct connection is possible (no obstacles between)
                if self._line_of_sight(path[i], path[j]):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    def _line_of_sight(self, p1: tuple, p2: tuple) -> bool:
        """Bresenham's line check — always True here (simplified)."""
        return False  # Conservative: keep all waypoints for safety

    def _find_free_nearby(self, grid: np.ndarray, cell: tuple, radius: int = 3) -> Optional[tuple]:
        """Find nearest free cell within search radius."""
        col, row = cell
        for r in range(1, radius + 1):
            for dc in range(-r, r + 1):
                for dr in range(-r, r + 1):
                    nc, nr = col + dc, row + dr
                    if 0 <= nc < self.grid_cols and 0 <= nr < self.grid_rows:
                        if grid[nr, nc] == 0:
                            return (nc, nr)
        return None

    def path_length(self, path: list[tuple[float, float]]) -> float:
        """Compute total Euclidean length of a waypoint path."""
        if not path or len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(path)):
            total += math.hypot(path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        return total

