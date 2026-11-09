"""
NavMind AI - RL Agent Evaluation
Evaluates trained DQN, PPO and A* agents under identical conditions.
"""
from __future__ import annotations

import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.robot_env import RobotNavEnv
from reinforcement_learning.dqn import DQNAgent


def evaluate_dqn(
    model_path: str,
    num_episodes: int = 100,
    difficulty: str = "medium",
    render: bool = False,
    dynamic_obstacles: bool = False,
    sensor_noise: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    Evaluate a trained DQN agent.

    Args:
        model_path: Path to saved DQN model
        num_episodes: Number of evaluation episodes
        difficulty: Environment difficulty
        render: Enable Pygame rendering
        dynamic_obstacles: Enable moving obstacles
        sensor_noise: Sensor noise level
        seed: Random seed for reproducibility

    Returns:
        Dictionary of evaluation metrics
    """
    env = RobotNavEnv(
        render_mode="human" if render else None,
        difficulty=difficulty,
        dynamic_obstacles=dynamic_obstacles,
        sensor_noise=sensor_noise,
    )

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )
    agent.load(model_path)
    agent.epsilon = 0.0  # Greedy evaluation

    results = _run_evaluation(env, agent, num_episodes, seed, algorithm="DQN")
    env.close()
    return results


def evaluate_ppo(
    model_path: str,
    num_episodes: int = 100,
    difficulty: str = "medium",
    dynamic_obstacles: bool = False,
    sensor_noise: float = 0.0,
    seed: int = 42,
) -> dict:
    """Evaluate a trained PPO agent."""
    from reinforcement_learning.ppo import PPOAgent

    env = RobotNavEnv(
        difficulty=difficulty,
        dynamic_obstacles=dynamic_obstacles,
        sensor_noise=sensor_noise,
    )

    ppo = PPOAgent()
    ppo.load(model_path)

    results = _run_evaluation(env, ppo, num_episodes, seed, algorithm="PPO")
    env.close()
    return results


def evaluate_astar(
    num_episodes: int = 100,
    difficulty: str = "medium",
    dynamic_obstacles: bool = False,
    seed: int = 42,
) -> dict:
    """Evaluate the A* path planner as a baseline."""
    from navigation.astar import AStarPlanner
    from navigation.controller import PathController

    env = RobotNavEnv(
        difficulty=difficulty,
        dynamic_obstacles=dynamic_obstacles,
    )
    planner = AStarPlanner(env.ENV_WIDTH, env.ENV_HEIGHT)
    controller = PathController()

    metrics = defaultdict(list)
    start_time = time.time()

    for ep in range(num_episodes):
        state, info = env.reset(seed=seed + ep)
        robot_pos = info["robot_pos"]
        goal_pos = info["goal_pos"]

        # Plan path
        grid = planner.build_grid(env.obstacle_manager)
        path = planner.plan(robot_pos, goal_pos, grid)

        if path is None:
            # No path found — count as failure
            metrics["success"].append(0)
            metrics["collision"].append(0)
            metrics["steps"].append(500)
            metrics["path_length"].append(0.0)
            metrics["timeout"].append(1)
            continue

        # Follow path
        done = False
        step = 0
        episode_collision = False
        while not done and step < 500:
            action = controller.get_action(env.robot, path)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            if info.get("collision"):
                episode_collision = True

        ep_metrics = info.get("episode", {})
        metrics["success"].append(int(ep_metrics.get("success", False)))
        metrics["collision"].append(int(episode_collision))
        metrics["steps"].append(step)
        metrics["path_length"].append(ep_metrics.get("path_length", 0.0))
        metrics["timeout"].append(int(ep_metrics.get("timeout", False)))

    env.close()

    return _aggregate_metrics(metrics, "A*", num_episodes, time.time() - start_time)


def _run_evaluation(env, agent, num_episodes: int, seed: int, algorithm: str) -> dict:
    """Run evaluation loop for any agent with select_action() interface."""
    metrics = defaultdict(list)
    start_time = time.time()

    for ep in range(num_episodes):
        state, info = env.reset(seed=seed + ep)
        done = False
        step = 0

        while not done:
            action = agent.select_action(state, **({} if algorithm == "A*" else {"training": False}))
            state, reward, terminated, truncated, ep_info = env.step(action)
            done = terminated or truncated
            step += 1

        ep_metrics = ep_info.get("episode", {})
        metrics["success"].append(int(ep_metrics.get("success", False)))
        metrics["collision"].append(int(ep_metrics.get("collision", False)))
        metrics["timeout"].append(int(ep_metrics.get("timeout", False)))
        metrics["steps"].append(ep_metrics.get("steps", step))
        metrics["episode_reward"].append(ep_metrics.get("episode_reward", 0.0))
        metrics["path_length"].append(ep_metrics.get("path_length", 0.0))

    return _aggregate_metrics(metrics, algorithm, num_episodes, time.time() - start_time)


def _aggregate_metrics(metrics: dict, algorithm: str, num_episodes: int, elapsed: float) -> dict:
    """Aggregate raw per-episode metrics into summary statistics."""
    return {
        "algorithm": algorithm,
        "num_episodes": num_episodes,
        "evaluation_time_s": round(elapsed, 2),
        "success_rate": round(np.mean(metrics["success"]) * 100, 2),
        "collision_rate": round(np.mean(metrics["collision"]) * 100, 2),
        "timeout_rate": round(np.mean(metrics.get("timeout", [0])) * 100, 2),
        "avg_steps": round(float(np.mean(metrics["steps"])), 2),
        "avg_path_length": round(float(np.mean(metrics["path_length"])), 2),
        "avg_episode_reward": round(float(np.mean(metrics.get("episode_reward", [0]))), 2),
        # Raw arrays for plotting
        "success_per_episode": metrics["success"],
        "steps_per_episode": metrics["steps"],
        "path_length_per_episode": metrics["path_length"],
    }


def compare_algorithms(
    dqn_path: Optional[str] = None,
    ppo_path: Optional[str] = None,
    num_episodes: int = 50,
    difficulty: str = "medium",
    seed: int = 42,
    save_path: str = "results/comparison.json",
) -> dict:
    """
    Compare DQN, PPO and A* under identical conditions.

    Args:
        dqn_path: Path to DQN model (None to skip)
        ppo_path: Path to PPO model (None to skip)
        num_episodes: Evaluation episodes per algorithm
        difficulty: Environment difficulty
        seed: Random seed

    Returns:
        Comparison dictionary
    """
    comparison = {}

    print(f"\n{'='*60}")
    print("  NavMind AI — Algorithm Comparison")
    print(f"{'='*60}\n")

    # A* Baseline
    print("▶ Evaluating A* baseline...")
    try:
        comparison["A*"] = evaluate_astar(num_episodes, difficulty, seed=seed)
        print(f"  ✅ Success: {comparison['A*']['success_rate']:.1f}%")
    except Exception as e:
        print(f"  ⚠️ A* evaluation failed: {e}")

    # DQN
    if dqn_path and Path(dqn_path).exists():
        print("▶ Evaluating DQN agent...")
        comparison["DQN"] = evaluate_dqn(dqn_path, num_episodes, difficulty, seed=seed)
        print(f"  ✅ Success: {comparison['DQN']['success_rate']:.1f}%")

    # PPO
    if ppo_path and Path(ppo_path).exists():
        print("▶ Evaluating PPO agent...")
        comparison["PPO"] = evaluate_ppo(ppo_path, num_episodes, difficulty, seed=seed)
        print(f"  ✅ Success: {comparison['PPO']['success_rate']:.1f}%")

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        # Remove raw arrays for JSON (keep only summary)
        summary = {
            alg: {k: v for k, v in metrics.items()
                  if not isinstance(v, list) or len(v) < 10}
            for alg, metrics in comparison.items()
        }
        json.dump(summary, f, indent=2)
    print(f"\n[Comparison] Results saved to {save_path}")

    return comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn", type=str, default=None, help="DQN model path")
    parser.add_argument("--ppo", type=str, default=None, help="PPO model path")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--difficulty", type=str, default="medium")
    args = parser.parse_args()

    compare_algorithms(
        dqn_path=args.dqn,
        ppo_path=args.ppo,
        num_episodes=args.episodes,
        difficulty=args.difficulty,
    )

