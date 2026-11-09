"""
NavMind AI - Slot Extractor
Extracts destination, speed, constraint, and direction slots from commands.
"""

import re
from typing import Optional


DESTINATION_KEYWORDS = {
    "charging station": ["charging station", "charging dock", "charger", "charge point"],
    "kitchen": ["kitchen", "cooking area"],
    "office": ["office", "workstation", "desk"],
    "home": ["home", "home position", "base", "start", "starting point", "home base"],
    "entrance": ["entrance", "exit", "door", "entry"],
    "center": ["center", "middle", "centre", "central"],
    "goal": ["goal", "target", "destination", "waypoint"],
}

SPEED_KEYWORDS = {
    "slow": ["slowly", "slow", "carefully", "cautiously", "gently", "at low speed"],
    "fast": ["quickly", "fast", "rapidly", "hurry", "speed up", "at full speed"],
    "normal": ["normally", "normal speed", "regular speed"],
}

DIRECTION_KEYWORDS = {
    "left": ["left", "leftward", "counterclockwise"],
    "right": ["right", "rightward", "clockwise"],
    "forward": ["forward", "straight", "ahead", "front"],
    "backward": ["backward", "back", "behind", "reverse"],
}

CONSTRAINT_KEYWORDS = {
    "avoid_obstacles": [
        "avoid obstacles", "avoiding obstacles", "avoid collisions", "avoiding collisions",
        "avoid", "avoiding", "obstacle avoidance", "be careful", "carefully",
        "cautiously", "safely", "safe mode", "don't hit", "watch out",
        "no collisions", "avoid walls",
    ],
}


class SlotExtractor:
    """
    Rule-based slot extractor for navigation commands.

    Extracts: destination, speed, constraint, direction, coordinates
    """

    def __init__(self):
        # Compile regex patterns for coordinates
        self.coord_pattern = re.compile(
            r"(?:position|pos|coordinates?|coords?|at|to)?\s*"
            r"[(\[]?\s*(\d+\.?\d*)\s*[,\s]\s*(\d+\.?\d*)\s*[)\]]?",
            re.IGNORECASE,
        )

    def extract(self, text: str) -> dict:
        """
        Extract all slots from a navigation command.

        Args:
            text: Input command text

        Returns:
            Dict with keys: destination, speed, constraint, direction,
                            goal_x, goal_y, raw_text
        """
        text_lower = text.lower()
        result = {
            "destination": "",
            "speed": "normal",
            "constraint": "",
            "direction": "",
            "goal_x": None,
            "goal_y": None,
            "raw_text": text,
        }

        # Extract destination
        result["destination"] = self._extract_destination(text_lower)

        # Extract speed modifier
        result["speed"] = self._extract_speed(text_lower)

        # Extract constraints
        result["constraint"] = self._extract_constraint(text_lower)

        # Extract direction
        result["direction"] = self._extract_direction(text_lower)

        # Extract explicit coordinates
        coords = self._extract_coordinates(text)
        if coords:
            result["goal_x"], result["goal_y"] = coords

        return result

    def _extract_destination(self, text: str) -> str:
        """Find the best matching destination name."""
        best_match = ""
        best_score = 0
        for dest_name, keywords in DESTINATION_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    # Longer match = higher priority
                    if len(kw) > best_score:
                        best_score = len(kw)
                        best_match = dest_name
        return best_match

    def _extract_speed(self, text: str) -> str:
        """Detect speed modifier in text."""
        for speed_level, keywords in SPEED_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    return speed_level
        return "normal"

    def _extract_constraint(self, text: str) -> str:
        """Detect navigation constraints."""
        for constraint_type, keywords in CONSTRAINT_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    return constraint_type
        return ""

    def _extract_direction(self, text: str) -> str:
        """Detect explicit direction commands."""
        # Check for 'turn/go/move + direction' patterns
        for direction, keywords in DIRECTION_KEYWORDS.items():
            for kw in keywords:
                # Check for direction word with an optional preceding verb
                pattern = r"\b(?:turn|go|move|rotate|face|spin)?\s*\b" + re.escape(kw) + r"\b"
                if re.search(pattern, text):
                    return direction
        return ""

    def _extract_coordinates(self, text: str) -> Optional[tuple[float, float]]:
        """Extract explicit (x, y) coordinate pairs from text."""
        match = self.coord_pattern.search(text)
        if match:
            try:
                x, y = float(match.group(1)), float(match.group(2))
                return x, y
            except (ValueError, IndexError):
                pass
        return None

