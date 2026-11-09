"""
NavMind AI - Path Following Controller
Converts A* waypoints into discrete robot actions.
"""
from __future__ import annotations

import math
from typing import Optional
from environment.robot import Robot


class PathController:
    """
    Pure-pursuit-style controller that converts a waypoint path
    into discrete robot actions for the navigation environment.
    
    Uses angular error to decide turn vs. forward.
    """

    # Thresholds
    WAYPOINT_REACH_DIST = 25.0   # pixels — when to advance to next waypoint
    TURN_THRESHOLD = 0.25        # radians — when to turn instead of go forward
    STOP_DIST = 20.0             # pixels — when to issue STOP near final goal

    def __init__(self, lookahead: int = 2):
        """
        Args:
            lookahead: Number of waypoints to look ahead (pure pursuit)
        """
        self.lookahead = lookahead
        self.current_waypoint_idx = 0
        self.path: list = []

    def set_path(self, path: list[tuple[float, float]]):
        """Set a new path and reset tracker."""
        self.path = path or []
        self.current_waypoint_idx = 0

    def get_action(self, robot: "Robot", path: Optional[list] = None) -> int:
        """
        Compute the best discrete action to follow the path.

        Args:
            robot: Current robot object
            path: Optional path override (uses stored path if None)

        Returns:
            Action index (0=Forward, 1=TurnLeft, 2=TurnRight, 3=Stop)
        """
        from environment.robot import Robot as RobotClass

        active_path = path if path is not None else self.path
        if not active_path:
            return RobotClass.ACTION_STOP

        # Advance waypoint index
        while self.current_waypoint_idx < len(active_path) - 1:
            wp = active_path[self.current_waypoint_idx]
            dist = robot.distance_to(*wp)
            if dist < self.WAYPOINT_REACH_DIST:
                self.current_waypoint_idx += 1
            else:
                break

        # Target lookahead waypoint
        target_idx = min(
            self.current_waypoint_idx + self.lookahead - 1,
            len(active_path) - 1,
        )
        target = active_path[target_idx]
        dist_to_target = robot.distance_to(*target)

        # At final waypoint — stop
        if target_idx == len(active_path) - 1 and dist_to_target < self.STOP_DIST:
            return RobotClass.ACTION_STOP

        # Compute heading error
        angle_error = robot.angle_to(*target)

        # Action selection
        if abs(angle_error) > self.TURN_THRESHOLD:
            # Turn toward waypoint
            if angle_error < 0:
                return RobotClass.ACTION_TURN_LEFT
            else:
                return RobotClass.ACTION_TURN_RIGHT
        else:
            return RobotClass.ACTION_FORWARD

    def is_path_complete(self, robot: "Robot") -> bool:
        """True if robot has reached the end of the path."""
        if not self.path:
            return True
        last_wp = self.path[-1]
        return robot.distance_to(*last_wp) < self.STOP_DIST

    def reset(self):
        """Reset path tracker."""
        self.path = []
        self.current_waypoint_idx = 0

