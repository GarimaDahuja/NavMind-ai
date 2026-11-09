"""
NavMind AI - Navigation Task Planner
Converts NLP intents into concrete navigation tasks.
"""

import math
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class NavigationIntent(Enum):
    NAVIGATE = "navigate"
    STOP = "stop"
    SET_CONSTRAINT = "set_constraint"
    OPTIMIZE = "optimize"
    DIRECTION = "direction"
    QUERY = "query"
    UNKNOWN = "unknown"


@dataclass
class NavigationTask:
    """Represents a parsed navigation task."""
    intent: NavigationIntent
    destination_name: Optional[str] = None
    goal_x: Optional[float] = None
    goal_y: Optional[float] = None
    speed_factor: float = 1.0
    avoid_obstacles: bool = True
    optimize_for: str = "distance"  # "distance" | "time" | "safety"
    direction: Optional[str] = None  # "left", "right", "forward", "back"
    raw_command: str = ""
    confidence: float = 1.0

    @property
    def has_goal(self) -> bool:
        return self.goal_x is not None and self.goal_y is not None


# Named locations in the environment (can be customized)
DEFAULT_LOCATIONS = {
    "charging station": (100, 100),
    "home": (100, 100),
    "start": (100, 100),
    "kitchen": (500, 100),
    "office": (500, 500),
    "entrance": (300, 550),
    "center": (300, 300),
    "goal": (500, 500),
    "destination": (500, 500),
}


class NavigationPlanner:
    """
    Maps NLP output to executable navigation tasks.

    Receives structured NLP intent + slots and translates them
    into NavigationTask objects that the RL agent or A* planner can execute.
    """

    def __init__(
        self,
        env_width: int = 600,
        env_height: int = 600,
        custom_locations: Optional[dict] = None,
    ):
        self.env_width = env_width
        self.env_height = env_height
        self.locations = {**DEFAULT_LOCATIONS}
        if custom_locations:
            self.locations.update(custom_locations)

    def plan(self, nlp_result: dict) -> NavigationTask:
        """
        Convert NLP result into a NavigationTask.

        Args:
            nlp_result: Output from NLP module with keys:
                - intent: Intent label string
                - destination: Optional destination name
                - speed: Optional speed modifier
                - constraint: Optional constraint string
                - direction: Optional direction string
                - confidence: Confidence score
                - raw_command: Original text

        Returns:
            NavigationTask object
        """
        intent_str = nlp_result.get("intent", "unknown").lower()
        intent = self._map_intent(intent_str)

        destination = nlp_result.get("destination", "").lower().strip()
        speed_str = nlp_result.get("speed", "normal")
        constraint = nlp_result.get("constraint", "")
        direction = nlp_result.get("direction", None)
        confidence = nlp_result.get("confidence", 1.0)
        raw_command = nlp_result.get("raw_command", "")

        # Resolve goal coordinates
        goal_x, goal_y = self._resolve_destination(destination)

        # Speed mapping
        speed_factor = self._map_speed(speed_str)

        task = NavigationTask(
            intent=intent,
            destination_name=destination or None,
            goal_x=goal_x,
            goal_y=goal_y,
            speed_factor=speed_factor,
            avoid_obstacles="avoid" in constraint.lower() if constraint else True,
            optimize_for=self._map_optimization(nlp_result.get("optimize", "")),
            direction=direction,
            raw_command=raw_command,
            confidence=confidence,
        )

        return task

    def plan_from_text(self, text: str) -> NavigationTask:
        """
        Simple rule-based planning directly from text (no NLP model needed for basics).
        Used as fallback when NLP model is not available.
        """
        text_lower = text.lower().strip()

        # Intent detection
        if any(w in text_lower for w in ["go to", "navigate", "move to", "reach", "find"]):
            intent = NavigationIntent.NAVIGATE
        elif any(w in text_lower for w in ["stop", "halt", "freeze", "pause"]):
            intent = NavigationIntent.STOP
        elif any(w in text_lower for w in ["turn left", "go left"]):
            return NavigationTask(intent=NavigationIntent.DIRECTION, direction="left", raw_command=text)
        elif any(w in text_lower for w in ["turn right", "go right"]):
            return NavigationTask(intent=NavigationIntent.DIRECTION, direction="right", raw_command=text)
        elif any(w in text_lower for w in ["forward", "go straight", "ahead"]):
            return NavigationTask(intent=NavigationIntent.DIRECTION, direction="forward", raw_command=text)
        elif any(w in text_lower for w in ["avoid", "obstacle"]):
            intent = NavigationIntent.SET_CONSTRAINT
        else:
            intent = NavigationIntent.UNKNOWN

        # Destination extraction
        destination = ""
        for loc_name in sorted(self.locations.keys(), key=len, reverse=True):
            if loc_name in text_lower:
                destination = loc_name
                break

        goal_x, goal_y = self._resolve_destination(destination)

        # Speed
        speed_factor = 1.0
        if "slowly" in text_lower or "slow" in text_lower:
            speed_factor = 0.5
        elif "fast" in text_lower or "quickly" in text_lower:
            speed_factor = 1.5

        return NavigationTask(
            intent=intent,
            destination_name=destination or None,
            goal_x=goal_x,
            goal_y=goal_y,
            speed_factor=speed_factor,
            raw_command=text,
            confidence=0.8,
        )

    def _map_intent(self, intent_str: str) -> NavigationIntent:
        mapping = {
            "navigate": NavigationIntent.NAVIGATE,
            "stop": NavigationIntent.STOP,
            "set_constraint": NavigationIntent.SET_CONSTRAINT,
            "optimize": NavigationIntent.OPTIMIZE,
            "direction": NavigationIntent.DIRECTION,
            "query": NavigationIntent.QUERY,
        }
        return mapping.get(intent_str, NavigationIntent.UNKNOWN)

    def _resolve_destination(self, name: str) -> tuple[Optional[float], Optional[float]]:
        """Resolve destination name to (x, y) coordinates."""
        if not name:
            return None, None
        for loc_name, coords in self.locations.items():
            if loc_name in name or name in loc_name:
                return float(coords[0]), float(coords[1])
        return None, None

    def _map_speed(self, speed_str: str) -> float:
        mapping = {
            "slow": 0.5, "slowly": 0.5, "low": 0.5,
            "normal": 1.0, "medium": 1.0,
            "fast": 1.5, "quickly": 1.5, "high": 1.5,
        }
        return mapping.get(speed_str.lower(), 1.0)

    def _map_optimization(self, opt_str: str) -> str:
        if "short" in opt_str or "distance" in opt_str:
            return "distance"
        elif "safe" in opt_str or "obstacle" in opt_str:
            return "safety"
        elif "time" in opt_str or "fast" in opt_str:
            return "time"
        return "distance"

    def add_location(self, name: str, x: float, y: float):
        """Register a new named location."""
        self.locations[name.lower()] = (x, y)

    def get_locations(self) -> dict:
        """Return all registered named locations."""
        return dict(self.locations)

    def describe_task(self, task: NavigationTask) -> str:
        """Generate a human-readable description of the task."""
        if task.intent == NavigationIntent.STOP:
            return "Stopping the robot."
        elif task.intent == NavigationIntent.NAVIGATE:
            dest = task.destination_name or f"({task.goal_x:.0f}, {task.goal_y:.0f})"
            speed_desc = {0.5: "slowly", 1.0: "", 1.5: "quickly"}.get(task.speed_factor, "")
            return f"Navigating {speed_desc} to {dest}.".strip()
        elif task.intent == NavigationIntent.DIRECTION:
            return f"Turning {task.direction}."
        elif task.intent == NavigationIntent.SET_CONSTRAINT:
            return "Setting navigation constraint: avoid obstacles."
        else:
            return f"Command understood: {task.raw_command}"

