"""
NavMind AI - Gymnasium-Compatible Robot Navigation Environment
Core RL environment implementing the Gymnasium interface.
"""

import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Any

from environment.robot import Robot
from environment.obstacles import ObstacleManager
from environment.sensors import DistanceSensorArray
from environment.map_generator import MapGenerator


class RobotNavEnv(gym.Env):
    """
    2D Robot Navigation Environment — Gymnasium Interface.

    Observation space (7-dimensional):
        [dist_to_goal, angle_to_goal, sensor_front, sensor_front_left,
         sensor_front_right, sensor_left, sensor_right]

    Action space (discrete):
        0: Forward
        1: Turn Left
        2: Turn Right
        3: Stop

    Reward:
        +100 : Goal reached
        -100 : Collision
        +5   : Moving closer to goal
        -2   : Moving away from goal
        -0.1 : Per-step penalty
        -20  : Max steps exceeded
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Environment dimensions
    ENV_WIDTH = 600
    ENV_HEIGHT = 600

    # Goal tolerance (pixels)
    GOAL_RADIUS = 25.0

    # Max steps per episode
    MAX_STEPS = 500

    def __init__(
        self,
        render_mode: Optional[str] = None,
        difficulty: str = "medium",
        dynamic_obstacles: bool = False,
        sensor_noise: float = 0.0,
        max_steps: int = MAX_STEPS,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.difficulty = difficulty
        self.dynamic_obstacles = dynamic_obstacles
        self.sensor_noise = sensor_noise
        self.max_steps = max_steps

        # Observation: [dist_goal, angle_goal, 5 sensors]
        # All normalized to reasonable ranges
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Actions: 0=Forward, 1=TurnLeft, 2=TurnRight, 3=Stop
        self.action_space = spaces.Discrete(4)

        # Components
        self.map_generator = MapGenerator(self.ENV_WIDTH, self.ENV_HEIGHT)
        self.robot = Robot(300, 300, radius=12)
        self.sensors = DistanceSensorArray(
            angles=DistanceSensorArray.BASIC_5,
            max_range=150.0,
            noise_std=sensor_noise,
        )
        self.obstacle_manager: Optional[ObstacleManager] = None

        # Episode state
        self.goal_x: float = 400.0
        self.goal_y: float = 400.0
        self.step_count: int = 0
        self.prev_dist: float = 0.0
        self.episode_reward: float = 0.0
        self.collision_flag: bool = False
        self.path_history: list = []

        # Metrics
        self.episode_metrics: dict = {}

        # Pygame rendering
        self._pygame_initialized = False
        self._screen = None
        self._clock = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Generate new map
        dyn_frac = 0.25 if self.dynamic_obstacles else 0.0
        difficulty = options.get("difficulty", self.difficulty) if options else self.difficulty

        self.obstacle_manager, robot_pos, goal_pos = self.map_generator.generate(
            difficulty=difficulty,
            seed=seed,
        )
        self.obstacle_manager.obstacles[0].is_dynamic  # touch reference

        # Adjust dynamic fraction
        if self.dynamic_obstacles:
            for obs in self.obstacle_manager.obstacles[4:]:  # skip border walls
                if random.random() < dyn_frac:
                    obs.is_dynamic = True
                    speed = random.uniform(0.5, 2.0)
                    angle = random.uniform(0, 2 * math.pi)
                    obs.dx = speed * math.cos(angle)
                    obs.dy = speed * math.sin(angle)
                    obs.color = (180, 100, 100)

        # Place robot
        self.robot.reset(robot_pos[0], robot_pos[1])
        self.goal_x, self.goal_y = goal_pos

        # Reset episode variables
        self.step_count = 0
        self.prev_dist = self.robot.distance_to(self.goal_x, self.goal_y)
        self.episode_reward = 0.0
        self.collision_flag = False
        self.path_history = [robot_pos]

        obs = self._get_observation()
        info = {
            "robot_pos": robot_pos,
            "goal_pos": goal_pos,
            "difficulty": difficulty,
        }

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.step_count += 1

        # Update dynamic obstacles
        if self.dynamic_obstacles and self.obstacle_manager:
            self.obstacle_manager.update_dynamic()

        # Execute robot action
        new_x, new_y = self.robot.step(action)

        # Collision check
        collision = self.obstacle_manager.check_collision(
            new_x, new_y, self.robot.radius
        )

        # Update sensors
        sensor_readings = self.sensors.update(
            new_x, new_y, self.robot.theta, self.obstacle_manager
        )

        # Track path
        self.path_history.append((new_x, new_y))

        # Compute reward
        reward, terminated, reason = self._compute_reward(new_x, new_y, collision)
        truncated = self.step_count >= self.max_steps

        if truncated and not terminated:
            reward += -20.0  # max steps penalty
            reason = "timeout"

        self.episode_reward += reward
        self.prev_dist = self.robot.distance_to(self.goal_x, self.goal_y)

        obs = self._get_observation()
        info = {
            "step": self.step_count,
            "reward": reward,
            "episode_reward": self.episode_reward,
            "collision": collision,
            "distance_to_goal": self.prev_dist,
            "reason": reason if (terminated or truncated) else "",
            "path_length": self.robot.total_distance,
        }

        if terminated or truncated:
            self.episode_metrics = {
                "success": reason == "goal_reached",
                "collision": reason == "collision",
                "timeout": reason == "timeout",
                "steps": self.step_count,
                "episode_reward": self.episode_reward,
                "path_length": self.robot.total_distance,
            }
            info["episode"] = self.episode_metrics

        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self, new_x: float, new_y: float, collision: bool
    ) -> tuple[float, bool, str]:
        """Compute reward for current step."""
        current_dist = self.robot.distance_to(self.goal_x, self.goal_y)

        # Terminal: Collision
        if collision:
            self.collision_flag = True
            return -100.0, True, "collision"

        # Terminal: Goal reached
        if current_dist < self.GOAL_RADIUS:
            return +100.0, True, "goal_reached"

        # Progress reward
        dist_delta = self.prev_dist - current_dist
        if dist_delta > 0:
            reward = +5.0 * (dist_delta / 10.0)  # scale by movement
        else:
            reward = -2.0 * (abs(dist_delta) / 10.0)

        # Per-step penalty
        reward -= 0.1

        return reward, False, ""

    def _get_observation(self) -> np.ndarray:
        """
        Construct normalized observation vector.
        
        Returns:
            [dist_goal, angle_goal, sensor_f, sensor_fl, sensor_fr, sensor_l, sensor_r]
        """
        # Distance to goal (normalized by map diagonal)
        max_dist = math.hypot(self.ENV_WIDTH, self.ENV_HEIGHT)
        dist_norm = self.robot.distance_to(self.goal_x, self.goal_y) / max_dist

        # Angle to goal (normalized to [-1, 1])
        angle = self.robot.angle_to(self.goal_x, self.goal_y) / math.pi

        # Sensor readings (already normalized in sensors.update)
        if self.obstacle_manager is not None:
            sensors = self.sensors.update(
                self.robot.x, self.robot.y, self.robot.theta, self.obstacle_manager
            )
        else:
            sensors = np.ones(5, dtype=np.float32)

        obs = np.array(
            [dist_norm, angle] + list(sensors[:5]),
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def render(self):
        """Render environment to screen (pygame) or numpy array."""
        if self.render_mode is None:
            return None

        if not self._pygame_initialized:
            self._init_pygame()

        return self._draw_frame()

    def _init_pygame(self):
        """Initialize pygame display."""
        try:
            import pygame
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode(
                    (self.ENV_WIDTH, self.ENV_HEIGHT)
                )
                pygame.display.set_caption("NavMind AI — Navigation Simulator")
            else:
                self._screen = pygame.Surface((self.ENV_WIDTH, self.ENV_HEIGHT))
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("monospace", 14)
            self._pygame_initialized = True
        except ImportError:
            pass

    def _draw_frame(self):
        """Draw one frame of the simulation."""
        try:
            import pygame

            # Background
            self._screen.fill((20, 20, 35))

            # Draw path history
            if len(self.path_history) > 1:
                for i in range(1, len(self.path_history)):
                    alpha = int(255 * i / len(self.path_history))
                    pygame.draw.line(
                        self._screen,
                        (alpha // 3, alpha // 2, alpha),
                        (int(self.path_history[i - 1][0]), int(self.path_history[i - 1][1])),
                        (int(self.path_history[i][0]), int(self.path_history[i][1])),
                        2,
                    )

            # Draw obstacles
            if self.obstacle_manager:
                for obs in self.obstacle_manager.obstacles:
                    pygame.draw.rect(
                        self._screen,
                        obs.color,
                        pygame.Rect(int(obs.x), int(obs.y), int(obs.width), int(obs.height)),
                    )

            # Draw goal
            goal_color = (100, 220, 100)
            pygame.draw.circle(
                self._screen,
                goal_color,
                (int(self.goal_x), int(self.goal_y)),
                int(self.GOAL_RADIUS),
            )
            # Goal marker
            pygame.draw.line(
                self._screen, (255, 255, 255),
                (int(self.goal_x) - 8, int(self.goal_y)),
                (int(self.goal_x) + 8, int(self.goal_y)), 2
            )
            pygame.draw.line(
                self._screen, (255, 255, 255),
                (int(self.goal_x), int(self.goal_y) - 8),
                (int(self.goal_x), int(self.goal_y) + 8), 2
            )

            # Draw sensor beams
            for i, rel_angle in enumerate(self.sensors.angles):
                global_angle = self.robot.theta + rel_angle
                dist = self.sensors.readings[i]
                end_x = int(self.robot.x + dist * math.cos(global_angle))
                end_y = int(self.robot.y + dist * math.sin(global_angle))
                alpha = max(50, int(255 * (1 - dist / self.sensors.max_range)))
                pygame.draw.line(
                    self._screen, (alpha, alpha // 2, 50),
                    (int(self.robot.x), int(self.robot.y)),
                    (end_x, end_y), 1,
                )

            # Draw robot
            robot_color = (80, 160, 240)
            pygame.draw.circle(
                self._screen,
                robot_color,
                (int(self.robot.x), int(self.robot.y)),
                int(self.robot.radius),
            )
            # Direction indicator
            dir_x = int(self.robot.x + self.robot.radius * math.cos(self.robot.theta))
            dir_y = int(self.robot.y + self.robot.radius * math.sin(self.robot.theta))
            pygame.draw.line(
                self._screen, (255, 255, 100),
                (int(self.robot.x), int(self.robot.y)),
                (dir_x, dir_y), 3,
            )

            # HUD
            hud_lines = [
                f"Step: {self.step_count}/{self.max_steps}",
                f"Dist: {self.robot.distance_to(self.goal_x, self.goal_y):.1f}px",
                f"Reward: {self.episode_reward:.1f}",
                f"Action: {Robot.ACTION_NAMES.get(self.robot.prev_action, '?')}",
            ]
            for i, line in enumerate(hud_lines):
                surf = self._font.render(line, True, (200, 200, 200))
                self._screen.blit(surf, (10, 10 + i * 18))

            if self.render_mode == "human":
                pygame.display.flip()
                self._clock.tick(self.metadata["render_fps"])
                return None
            else:
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2)
                )
        except Exception:
            return None

    def close(self):
        """Clean up pygame resources."""
        if self._pygame_initialized:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._pygame_initialized = False


# Import Robot for HUD
from environment.robot import Robot


# Register environment with Gymnasium
def register_env():
    """Register NavMindEnv with Gymnasium registry."""
    try:
        gym.register(
            id="NavMind-v0",
            entry_point="environment.robot_env:RobotNavEnv",
            max_episode_steps=500,
        )
    except gym.error.Error:
        pass  # Already registered


register_env()

