"""
NavMind AI - RL Training Loop
Trains DQN and PPO agents on the robot navigation environment.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from collections import deque

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.robot_env import RobotNavEnv
from reinforcement_learning.dqn import DQNAgent


def train_dqn(
    num_episodes: int = 1000,
    max_steps: int = 500,
    difficulty: str = "medium",
    save_path: str = "models/dqn_navmind.pth",
    results_path: str = "results/dqn_training.json",
    target_update_freq: int = 10,
    print_freq: int = 50,
    render: bool = False,
    dynamic_obstacles: bool = False,
    sensor_noise: float = 0.0,
) -> dict:
    """
    Train the DQN agent.

    Args:
        num_episodes: Total training episodes
        max_steps: Max steps per episode
        difficulty: Environment difficulty level
        save_path: Path to save best model
        results_path: Path to save training results JSON
        target_update_freq: How often to update target network (episodes)
        print_freq: Logging frequency (episodes)
        render: Enable Pygame rendering during training
        dynamic_obstacles: Enable moving obstacles
        sensor_noise: Sensor noise standard deviation

    Returns:
        Dictionary of training statistics
    """
    print(f"\n{'='*60}")
    print("  NavMind AI — DQN Training")
    print(f"{'='*60}")
    print(f"  Episodes:    {num_episodes}")
    print(f"  Difficulty:  {difficulty}")
    print(f"  Dynamic obs: {dynamic_obstacles}")
    print(f"  Sensor noise:{sensor_noise}")
    print(f"{'='*60}\n")

    # Create environment
    env = RobotNavEnv(
        render_mode="human" if render else None,
        difficulty=difficulty,
        dynamic_obstacles=dynamic_obstacles,
        sensor_noise=sensor_noise,
        max_steps=max_steps,
    )

    # Create agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_capacity=50_000,
        target_update_freq=target_update_freq,
        dueling=True,
    )

    # Tracking
    episode_rewards = []
    episode_lengths = []
    success_flags = []
    collision_flags = []
    path_lengths = []
    losses = []

    best_success_rate = 0.0
    recent_rewards = deque(maxlen=100)
    recent_success = deque(maxlen=100)

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        episode_reward = 0.0
        episode_loss = []

        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, training=True)

            # Execute step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Learn
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        # Episode ended
        metrics = info.get("episode", {})
        success = metrics.get("success", False)
        collision = metrics.get("collision", False)
        path_len = metrics.get("path_length", 0.0)

        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        success_flags.append(int(success))
        collision_flags.append(int(collision))
        path_lengths.append(path_len)
        if episode_loss:
            losses.append(np.mean(episode_loss))

        recent_rewards.append(episode_reward)
        recent_success.append(int(success))

        # Decay epsilon
        agent.decay_epsilon()

        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # Save best model
        current_success_rate = np.mean(recent_success)
        if current_success_rate > best_success_rate and len(recent_success) >= 50:
            best_success_rate = current_success_rate
            agent.save(save_path)

        # Logging
        if episode % print_freq == 0:
            elapsed = time.time() - start_time
            avg_reward = np.mean(recent_rewards)
            avg_success = np.mean(recent_success) * 100
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(
                f"Episode {episode:5d}/{num_episodes} | "
                f"Reward: {avg_reward:7.1f} | "
                f"Success: {avg_success:5.1f}% | "
                f"ε: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {elapsed:.0f}s"
            )

    # Final save
    agent.save(save_path.replace(".pth", "_final.pth"))

    training_results = {
        "algorithm": "DQN",
        "num_episodes": num_episodes,
        "difficulty": difficulty,
        "dynamic_obstacles": dynamic_obstacles,
        "sensor_noise": sensor_noise,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "success_flags": success_flags,
        "collision_flags": collision_flags,
        "path_lengths": path_lengths,
        "losses": losses,
        "final_epsilon": agent.epsilon,
        "best_success_rate": best_success_rate,
        "total_time_seconds": time.time() - start_time,
    }

    # Save results
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(training_results, f, indent=2)
    print(f"\n[Training] Results saved to {results_path}")

    env.close()
    return training_results


def train_ppo(
    total_timesteps: int = 200_000,
    difficulty: str = "medium",
    save_path: str = "models/ppo_navmind",
    dynamic_obstacles: bool = False,
    sensor_noise: float = 0.0,
) -> dict:
    """
    Train the PPO agent using Stable-Baselines3.

    Args:
        total_timesteps: Total environment steps for training
        difficulty: Environment difficulty
        save_path: Path prefix for saving model
        dynamic_obstacles: Enable moving obstacles
        sensor_noise: Sensor noise level

    Returns:
        Dictionary of training statistics
    """
    print(f"\n{'='*60}")
    print("  NavMind AI — PPO Training (Stable-Baselines3)")
    print(f"{'='*60}")
    print(f"  Timesteps:   {total_timesteps}")
    print(f"  Difficulty:  {difficulty}")
    print(f"{'='*60}\n")

    from reinforcement_learning.ppo import PPOAgent

    env_kwargs = {
        "difficulty": difficulty,
        "dynamic_obstacles": dynamic_obstacles,
        "sensor_noise": sensor_noise,
    }

    agent = PPOAgent(
        env_kwargs=env_kwargs,
        lr=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
    )
    agent.build_model()
    agent.train(
        total_timesteps=total_timesteps,
        save_path=save_path,
    )

    results = agent.get_stats()
    results["algorithm"] = "PPO"
    results["total_timesteps"] = total_timesteps
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NavMind AI — RL Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algorithm", choices=["dqn", "ppo"], default="dqn",
        help="RL algorithm to train"
    )
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes (DQN)")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total timesteps (PPO)")
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard", "dynamic", "maze"],
        default="medium", help="Environment difficulty"
    )
    parser.add_argument("--render", action="store_true", help="Enable Pygame rendering")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic obstacles")
    parser.add_argument("--noise", type=float, default=0.0, help="Sensor noise std dev")
    parser.add_argument("--save", type=str, default=None, help="Model save path")
    args = parser.parse_args()

    if args.algorithm == "dqn":
        save_path = args.save or f"models/dqn_{args.difficulty}.pth"
        results = train_dqn(
            num_episodes=args.episodes,
            difficulty=args.difficulty,
            save_path=save_path,
            render=args.render,
            dynamic_obstacles=args.dynamic,
            sensor_noise=args.noise,
        )
        final_success = np.mean(results["success_flags"][-100:]) * 100
        print(f"\n✅ DQN Training Complete!")
        print(f"   Final success rate: {final_success:.1f}%")
        print(f"   Best success rate:  {results['best_success_rate']*100:.1f}%")

    elif args.algorithm == "ppo":
        save_path = args.save or f"models/ppo_{args.difficulty}"
        results = train_ppo(
            total_timesteps=args.timesteps,
            difficulty=args.difficulty,
            save_path=save_path,
            dynamic_obstacles=args.dynamic,
            sensor_noise=args.noise,
        )
        print(f"\n✅ PPO Training Complete!")

