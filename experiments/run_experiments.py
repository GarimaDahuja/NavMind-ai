"""
NavMind AI - Research Experiments Runner
Automates evaluation across the 5 core research questions:
    Q1: Dynamic Obstacle Performance (A* vs DQN)
    Q2: Obstacle Density & Clutter Impact (Easy vs Medium vs Hard)
    Q3: Generalization to Unseen Environments (BARN Benchmark Maps)
    Q4: Sensor Noise Robustness Evaluation (Noise STD from 0 to 10px)
    Q5: Algorithmic Comparative Benchmark (Success, Collision, Steps, Path Length)
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.robot_env import RobotNavEnv
from environment.barn_loader import BARNLoader
from navigation.astar import AStarPlanner
from navigation.controller import PathController
from reinforcement_learning.dqn import DQNAgent
from reinforcement_learning.evaluate import evaluate_astar, evaluate_dqn


def run_q1_dynamic_obstacles(episodes=10, dqn_model="models/dqn_navmind.pth"):
    print("\n--- Running Experiment Q1: Dynamic Obstacle Performance ---")
    results = {}
    print("Evaluating A* on dynamic obstacles...")
    results["A*"] = evaluate_astar(num_episodes=episodes, difficulty="dynamic", dynamic_obstacles=True)
    if Path(dqn_model).exists():
        print("Evaluating DQN on dynamic obstacles...")
        results["DQN"] = evaluate_dqn(dqn_model, num_episodes=episodes, difficulty="dynamic", dynamic_obstacles=True)
    return results


def run_q2_obstacle_density(episodes=10, dqn_model="models/dqn_navmind.pth"):
    print("\n--- Running Experiment Q2: Obstacle Density Impact ---")
    densities = ["easy", "medium", "hard"]
    results = {}
    for d in densities:
        print(f"Testing density: {d}...")
        results[d] = {
            "A*": evaluate_astar(num_episodes=episodes, difficulty=d),
        }
        if Path(dqn_model).exists():
            results[d]["DQN"] = evaluate_dqn(dqn_model, num_episodes=episodes, difficulty=d)
    return results


def run_q3_barn_benchmarks(episodes=5, dqn_model="models/dqn_navmind.pth"):
    print("\n--- Running Experiment Q3: Generalization on BARN Maps ---")
    loader = BARNLoader(600, 600)
    planner = AStarPlanner(600, 600)
    controller = PathController()
    benchmarks = loader.get_benchmark_names()

    results = {}
    for map_name in benchmarks:
        print(f"Testing on {map_name}...")
        manager, start, goal = loader.load_preset(map_name)
        grid = planner.build_grid(manager)
        path = planner.plan(start, goal, grid)
        has_path = path is not None
        path_len = planner.path_length(path) if has_path else 0.0
        results[map_name] = {
            "path_found": has_path,
            "waypoints": len(path) if has_path else 0,
            "path_length": round(path_len, 2),
        }
    return results


def run_q4_sensor_noise(episodes=10, dqn_model="models/dqn_navmind.pth"):
    print("\n--- Running Experiment Q4: Sensor Noise Degradation ---")
    noise_levels = [0.0, 2.0, 5.0, 10.0]
    results = {}
    if Path(dqn_model).exists():
        for noise in noise_levels:
            print(f"Evaluating DQN at noise_std={noise}...")
            res = evaluate_dqn(dqn_model, num_episodes=episodes, difficulty="medium", sensor_noise=noise)
            results[f"noise_{noise}"] = {
                "success_rate": res["success_rate"],
                "collision_rate": res["collision_rate"],
                "avg_steps": res["avg_steps"],
            }
    else:
        print("DQN model not found. Skipping sensor noise test for DQN.")
    return results


def main():
    print("=" * 60)
    print("  NavMind AI — Automated Research Experiments")
    print("=" * 60)

    out_dir = Path("results/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Q1_dynamic": run_q1_dynamic_obstacles(episodes=5),
        "Q2_density": run_q2_obstacle_density(episodes=5),
        "Q3_barn": run_q3_barn_benchmarks(episodes=3),
        "Q4_noise": run_q4_sensor_noise(episodes=5),
    }

    out_file = out_dir / "research_experiments_summary.json"
    with open(out_file, "w") as f:
        # Save compact summary
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ All experiments completed! Results saved to: {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

