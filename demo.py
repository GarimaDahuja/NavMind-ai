"""
NavMind AI - Standalone Simulation Demo
Run this to test the navigation environment without Streamlit.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from environment.robot_env import RobotNavEnv


def run_demo(
    algorithm: str = "random",
    episodes: int = 3,
    max_steps: int = 300,
    render: bool = True,
):
    """
    Run a quick demo of the navigation environment.

    Args:
        algorithm: 'random', 'keyboard', or 'astar'
        episodes: Number of episodes to run
        max_steps: Steps per episode
        render: Enable Pygame rendering
    """
    print("\n" + "=" * 60)
    print("  NavMind AI — Simulation Demo")
    print("=" * 60)
    print(f"  Algorithm: {algorithm}")
    print(f"  Episodes:  {episodes}")
    print(f"  Render:    {render}")
    print("=" * 60 + "\n")

    env = RobotNavEnv(
        render_mode="human" if render else None,
        difficulty="medium",
        max_steps=max_steps,
    )

    if algorithm == "astar":
        from navigation.astar import AStarPlanner
        from navigation.controller import PathController
        planner = AStarPlanner(600, 600)
        controller = PathController()

    total_successes = 0
    total_collisions = 0

    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}", end="")
        obs, info = env.reset(seed=episode * 7)

        if algorithm == "astar":
            grid = planner.build_grid(env.obstacle_manager)
            path = planner.plan(info["robot_pos"], info["goal_pos"], grid)
            if path:
                controller.set_path(path)
                print(f" | A* path: {len(path)} waypoints")
            else:
                print(" | No A* path found, using random fallback")
                path = None

        done = False
        step = 0
        ep_reward = 0.0

        while not done and step < max_steps:
            if algorithm == "random":
                action = env.action_space.sample()
            elif algorithm == "astar" and path:
                action = controller.get_action(env.robot, path)
            else:
                action = 0  # Forward

            obs, reward, terminated, truncated, ep_info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step += 1

            if render:
                env.render()
                time.sleep(0.02)

        metrics = ep_info.get("episode", {})
        success = metrics.get("success", False)
        collision = metrics.get("collision", False)
        total_successes += int(success)
        total_collisions += int(collision)

        status = "✅ Goal" if success else ("💥 Collision" if collision else "⏰ Timeout")
        print(f" | Steps: {step:4d} | Reward: {ep_reward:8.1f} | {status}")

    env.close()
    print(f"\n{'='*60}")
    print(f"  Results: {total_successes}/{episodes} successes, {total_collisions} collisions")
    print(f"{'='*60}\n")

    return total_successes, total_collisions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NavMind AI Demo")
    parser.add_argument("--algorithm", choices=["random", "astar"], default="astar")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    run_demo(
        algorithm=args.algorithm,
        episodes=args.episodes,
        max_steps=args.steps,
        render=not args.no_render,
    )

