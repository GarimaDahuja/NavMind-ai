"""
NavMind AI - Automated Test Suite
Verifies core functionality across environment, navigation, RL, and NLP modules.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.robot import Robot
from environment.obstacles import Obstacle, ObstacleManager
from environment.sensors import DistanceSensorArray
from environment.robot_env import RobotNavEnv
from environment.barn_loader import BARNLoader, BARN_SAMPLE_MAPS
from navigation.astar import AStarPlanner
from navigation.controller import PathController
from navigation.planner import NavigationPlanner, NavigationIntent
from nlp.intent_classifier import IntentClassifier, RuleBasedClassifier
from nlp.slot_extractor import SlotExtractor
from reinforcement_learning.dqn import DQNAgent, DQNNetwork, DuelingDQNNetwork
from reinforcement_learning.replay_buffer import ReplayBuffer


class TestEnvironment:
    def test_robot_kinematics(self):
        robot = Robot(x=100.0, y=100.0, theta=0.0)
        # Move forward
        new_x, new_y = robot.step(Robot.ACTION_FORWARD)
        assert new_x > 100.0
        assert math.isclose(new_y, 100.0, abs_tol=1e-3)
        assert robot.velocity > 0.0

        # Turn left
        old_theta = robot.theta
        robot.step(Robot.ACTION_TURN_LEFT)
        assert robot.theta < old_theta

        # Reset
        robot.reset(x=50.0, y=50.0)
        assert robot.x == 50.0 and robot.y == 50.0
        assert robot.steps_taken == 0

    def test_obstacle_collision(self):
        manager = ObstacleManager(env_width=600, env_height=600)
        obs = Obstacle(x=100, y=100, width=50, height=50)
        manager.add_obstacle(obs)

        # Collision directly inside obstacle
        assert manager.check_collision(120, 120, radius=10) is True
        # Far away
        assert manager.check_collision(300, 300, radius=10) is False

    def test_sensor_ray_cast(self):
        manager = ObstacleManager(env_width=600, env_height=600)
        manager.add_obstacle(Obstacle(x=200, y=100, width=50, height=50))

        # Sensor looking directly towards obstacle
        reading = manager.get_sensor_reading(robot_x=100, robot_y=125, angle=0.0, max_range=200.0)
        assert reading < 200.0
        assert math.isclose(reading, 100.0, abs_tol=10.0)

    def test_barn_loader(self):
        loader = BARNLoader(env_width=600, env_height=600)
        names = loader.get_benchmark_names()
        assert len(names) >= 4
        assert "barn_world_001_easy" in names

        manager, start, goal = loader.load_preset("barn_world_001_easy")
        assert len(manager.obstacles) > 4  # Includes borders + obstacles
        assert start[0] > 0 and start[1] > 0
        assert goal[0] > 0 and goal[1] > 0

    def test_gym_environment_step(self):
        env = RobotNavEnv(render_mode=None, difficulty="easy", max_steps=50)
        obs, info = env.reset(seed=42)
        assert obs.shape == (7,)
        assert env.observation_space.contains(obs)

        # Step
        next_obs, reward, terminated, truncated, step_info = env.step(Robot.ACTION_FORWARD)
        assert next_obs.shape == (7,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()


class TestNavigation:
    def test_astar_pathfinding(self):
        planner = AStarPlanner(env_width=300, env_height=300, cell_size=15)
        manager = ObstacleManager(300, 300)
        # Place a wall in the middle with a gap
        manager.add_obstacle(Obstacle(x=140, y=0, width=20, height=200))
        grid = planner.build_grid(manager, robot_radius=5.0)

        start = (50.0, 50.0)
        goal = (250.0, 50.0)
        path = planner.plan(start, goal, grid)

        assert path is not None
        assert len(path) > 2
        length = planner.path_length(path)
        assert length > 200.0  # Euclidean distance must be at least start-to-goal around wall

    def test_path_controller(self):
        controller = PathController()
        path = [(100.0, 100.0), (150.0, 100.0), (200.0, 100.0)]
        controller.set_path(path)
        robot = Robot(x=100.0, y=100.0, theta=0.0)

        action = controller.get_action(robot)
        # Facing toward (150, 100) along theta=0 should move Forward
        assert action == Robot.ACTION_FORWARD


class TestRL:
    def test_replay_buffer(self):
        buf = ReplayBuffer(capacity=100)
        state = np.zeros(7, dtype=np.float32)
        next_state = np.ones(7, dtype=np.float32)

        for _ in range(10):
            buf.push(state, action=0, reward=1.0, next_state=next_state, done=False)

        assert len(buf) == 10
        assert buf.is_ready(batch_size=5) is True

        s, a, r, ns, d = buf.sample(batch_size=4)
        assert s.shape == (4, 7)
        assert a.shape == (4,)
        assert r.shape == (4,)

    def test_dqn_agent(self):
        agent = DQNAgent(state_dim=7, action_dim=4, dueling=True)
        state = np.random.randn(7).astype(np.float32)

        # Select action
        action = agent.select_action(state, training=False)
        assert action in [0, 1, 2, 3]

        # Feed some transitions and test learn step
        for _ in range(70):
            s = np.random.randn(7).astype(np.float32)
            ns = np.random.randn(7).astype(np.float32)
            agent.remember(s, action=1, reward=0.5, next_state=ns, done=False)

        loss = agent.learn()
        assert loss is not None
        assert isinstance(loss, float)


class TestNLP:
    def test_rule_classifier(self):
        clf = RuleBasedClassifier()

        intent, conf = clf.classify("Navigate to the charging station")
        assert intent == "NAVIGATE"
        assert conf > 0.0

        intent, conf = clf.classify("Emergency stop immediately!")
        assert intent == "STOP"

        intent, conf = clf.classify("Please avoid all obstacles and be careful")
        assert intent == "SET_CONSTRAINT"

        intent, conf = clf.classify("Turn left now")
        assert intent == "DIRECTION"

    def test_slot_extractor(self):
        extractor = SlotExtractor()
        slots = extractor.extract("Move slowly to the office avoiding obstacles")
        assert slots["destination"] == "office"
        assert slots["speed"] == "slow"
        assert slots["constraint"] == "avoid_obstacles"

    def test_task_planner(self):
        planner = NavigationPlanner(env_width=600, env_height=600)
        task = planner.plan_from_text("Go to the kitchen quickly")
        assert task.intent == NavigationIntent.NAVIGATE
        assert task.destination_name == "kitchen"
        assert task.has_goal is True
        assert task.speed_factor == 1.5


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))

