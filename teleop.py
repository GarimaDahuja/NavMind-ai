"""
NavMind AI - Interactive Keyboard Teleoperation
Allows manual driving of the robot using Arrow Keys or WASD.
Visualizes the robot trajectory, sensor beams, obstacles, and collision detection.

Controls:
    UP / W     : Move Forward
    LEFT / A   : Turn Left
    RIGHT / D  : Turn Right
    DOWN / S   : Stop
    R          : Reset Environment
    ESC / Q    : Quit
"""

import sys
import time
import math
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import pygame
except ImportError:
    print("[Error] Pygame is required for teleoperation. Run: pip install pygame")
    sys.exit(1)

from environment.robot_env import RobotNavEnv
from environment.robot import Robot


def run_teleop(difficulty: str = "medium", dynamic: bool = False):
    print("\n" + "=" * 60)
    print("  NavMind AI — Keyboard Teleoperation Mode")
    print("=" * 60)
    print("  Controls:")
    print("    UP / W    : Move Forward")
    print("    LEFT / A  : Turn Left")
    print("    RIGHT / D : Turn Right")
    print("    DOWN / S  : Stop")
    print("    R         : Reset Environment / New Map")
    print("    ESC / Q   : Quit")
    print("=" * 60 + "\n")

    env = RobotNavEnv(
        render_mode="human",
        difficulty=difficulty,
        dynamic_obstacles=dynamic,
        max_steps=1000,
    )

    pygame.init()
    obs, info = env.reset()
    env.render()
    running = True
    clock = pygame.time.Clock()

    current_action = Robot.ACTION_STOP
    step_count = 0
    total_reward = 0.0

    while running:
        # Event processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    print("\n[Teleop] Resetting environment...")
                    obs, info = env.reset()
                    step_count = 0
                    total_reward = 0.0
                    current_action = Robot.ACTION_STOP

        # Continuous key state handling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            current_action = Robot.ACTION_FORWARD
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            current_action = Robot.ACTION_TURN_LEFT
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            current_action = Robot.ACTION_TURN_RIGHT
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            current_action = Robot.ACTION_STOP
        else:
            current_action = Robot.ACTION_STOP

        # Step the environment
        obs, reward, terminated, truncated, ep_info = env.step(current_action)
        step_count += 1
        total_reward += reward

        # Render display
        env.render()
        clock.tick(30)  # 30 FPS

        if terminated or truncated:
            reason = ep_info.get("reason", "done")
            if reason == "goal_reached":
                print(f"🎉 GOAL REACHED! Steps: {step_count}, Reward: {total_reward:.1f}")
            elif reason == "collision":
                print(f"💥 COLLISION! Robot hit an obstacle! Steps: {step_count}")
            elif reason == "timeout":
                print("⏰ Maximum steps reached.")
            
            time.sleep(1.0)
            obs, info = env.reset()
            step_count = 0
            total_reward = 0.0
            current_action = Robot.ACTION_STOP

    env.close()
    print("[Teleop] Closed successfully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NavMind AI Teleoperation")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard", "dynamic", "maze"])
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic moving obstacles")
    args = parser.parse_args()

    run_teleop(difficulty=args.difficulty, dynamic=args.dynamic)

