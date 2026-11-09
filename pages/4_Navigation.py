"""
NavMind AI - Navigation Simulator Page
Live robot navigation visualization with DQN, PPO, and A* agents.
"""

import sys
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="NavMind AI — Simulator", page_icon="🤖", layout="wide")

st.markdown("# 🤖 Navigation Simulator")
st.markdown("*Watch the robot navigate in real time*")
st.divider()

# ─── Sidebar controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎮 Simulation Controls")
    algorithm = st.selectbox("Algorithm", ["A* (Classical)", "DQN (RL)", "PPO (RL)"])
    difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard", "dynamic", "maze"])
    num_steps = st.slider("Max Steps", 50, 500, 200)
    step_delay = st.slider("Step Delay (ms)", 0, 200, 30)
    seed = st.number_input("Seed", 0, 9999, 42)
    show_sensors = st.checkbox("Show Sensor Beams", value=True)
    show_path = st.checkbox("Show Path Trail", value=True)

    st.divider()
    # Check for model files
    dqn_path = Path("models/dqn_navmind.pth")
    ppo_path = Path("models/ppo_navmind.zip")
    if algorithm == "DQN (RL)" and not dqn_path.exists():
        st.warning("⚠️ DQN model not found. Train first on the RL Training page.")
    if algorithm == "PPO (RL)" and not ppo_path.exists():
        st.warning("⚠️ PPO model not found. Train first on the RL Training page.")

col_sim, col_stats = st.columns([2, 1])

with col_sim:
    st.markdown("### 🌍 Simulation View")
    sim_placeholder = st.empty()

with col_stats:
    st.markdown("### 📊 Live Stats")
    stats_placeholder = st.empty()

start_btn = st.button("▶️ Start Simulation", type="primary", use_container_width=True)
stop_btn = st.button("⏹️ Stop", use_container_width=True)

# ─── Simulation Loop ─────────────────────────────────────────────────────────
if start_btn:
    st.session_state["sim_running"] = True
    st.session_state["sim_results"] = []

if stop_btn:
    st.session_state["sim_running"] = False


def render_env_plotly(
    robot_x, robot_y, robot_theta,
    goal_x, goal_y,
    obstacles,
    path_history,
    sensor_readings,
    sensor_angles_global,
    sensor_max_range,
    show_sensors=True,
    show_path=True,
    width=580, height=580,
):
    """Render navigation environment as Plotly figure."""
    import math
    fig = go.Figure()

    fig.update_layout(
        width=width, height=height,
        xaxis=dict(range=[0, 600], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 600], showgrid=False, zeroline=False, scaleanchor="x", showticklabels=False),
        plot_bgcolor="#141428",
        paper_bgcolor="#0d1117",
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )

    # Obstacles
    for obs in obstacles:
        fig.add_shape(
            type="rect",
            x0=obs.x, y0=obs.y, x1=obs.x + obs.width, y1=obs.y + obs.height,
            fillcolor="rgba(100,100,180,0.75)" if not obs.is_dynamic else "rgba(200,80,80,0.75)",
            line=dict(color="rgba(200,200,255,0.3)", width=1),
        )

    # Path trail
    if show_path and len(path_history) > 1:
        px_v = [p[0] for p in path_history]
        py_v = [p[1] for p in path_history]
        fig.add_trace(go.Scatter(
            x=px_v, y=py_v, mode="lines",
            line=dict(color="rgba(100,180,255,0.4)", width=2),
        ))

    # Sensor beams
    if show_sensors and sensor_readings is not None:
        for i, (angle, dist) in enumerate(zip(sensor_angles_global, sensor_readings)):
            end_x = robot_x + dist * math.cos(angle)
            end_y = robot_y + dist * math.sin(angle)
            alpha = max(0.05, 1 - dist / sensor_max_range)
            fig.add_trace(go.Scatter(
                x=[robot_x, end_x], y=[robot_y, end_y], mode="lines",
                line=dict(color=f"rgba(255,160,50,{alpha:.2f})", width=1),
            ))

    # Goal
    theta_arr = np.linspace(0, 2 * np.pi, 30)
    goal_r = 25
    fig.add_trace(go.Scatter(
        x=goal_x + goal_r * np.cos(theta_arr),
        y=goal_y + goal_r * np.sin(theta_arr),
        fill="toself",
        fillcolor="rgba(50,220,50,0.3)",
        line=dict(color="limegreen", width=2),
        mode="lines",
    ))
    fig.add_annotation(x=goal_x, y=goal_y, text="⭐", showarrow=False, font=dict(size=18))

    # Robot body
    robot_r = 12
    fig.add_trace(go.Scatter(
        x=robot_x + robot_r * np.cos(theta_arr),
        y=robot_y + robot_r * np.sin(theta_arr),
        fill="toself",
        fillcolor="rgba(80,160,240,0.8)",
        line=dict(color="white", width=1.5),
        mode="lines",
    ))

    # Direction arrow
    import math
    arrow_len = 20
    fig.add_annotation(
        x=robot_x + arrow_len * math.cos(robot_theta),
        y=robot_y + arrow_len * math.sin(robot_theta),
        ax=robot_x, ay=robot_y,
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=2, arrowwidth=3, arrowcolor="yellow",
        showarrow=True,
    )

    return fig


if st.session_state.get("sim_running", False):
    try:
        import math
        from environment.robot_env import RobotNavEnv

        # Initialize environment
        env = RobotNavEnv(
            render_mode=None,
            difficulty=difficulty,
            dynamic_obstacles=(difficulty == "dynamic"),
            max_steps=num_steps,
        )
        obs, info = env.reset(seed=int(seed))

        # Load agent
        agent = None
        astar_path = None

        if algorithm == "DQN (RL)" and dqn_path.exists():
            from reinforcement_learning.dqn import DQNAgent
            agent = DQNAgent(state_dim=7, action_dim=4)
            agent.load(str(dqn_path))
            agent.epsilon = 0.0

        elif algorithm == "PPO (RL)" and ppo_path.exists():
            from reinforcement_learning.ppo import PPOAgent
            agent = PPOAgent()
            agent.load(str(ppo_path))

        elif algorithm == "A* (Classical)":
            from navigation.astar import AStarPlanner
            from navigation.controller import PathController
            planner = AStarPlanner(600, 600)
            controller = PathController()
            grid = planner.build_grid(env.obstacle_manager)
            astar_path = planner.plan(info["robot_pos"], info["goal_pos"], grid)
            if astar_path:
                controller.set_path(astar_path)
            else:
                st.warning("⚠️ No A* path found. Try a different map or seed.")

        # Run simulation
        step = 0
        path_hist = [(env.robot.x, env.robot.y)]
        done = False
        episode_reward = 0.0
        result_text = ""

        while not done and step < num_steps and st.session_state.get("sim_running", True):
            # Select action
            if algorithm == "A* (Classical)" and controller:
                action = controller.get_action(env.robot, astar_path)
            elif agent:
                if algorithm == "DQN (RL)":
                    action = agent.select_action(obs, training=False)
                else:
                    action = agent.select_action(obs, deterministic=True)
            else:
                # Fallback: simple greedy
                action = 0

            obs, reward, terminated, truncated, ep_info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
            path_hist.append((env.robot.x, env.robot.y))

            # Render
            sensor_global_angles = [env.robot.theta + a for a in env.sensors.angles]
            fig = render_env_plotly(
                robot_x=env.robot.x,
                robot_y=env.robot.y,
                robot_theta=env.robot.theta,
                goal_x=env.goal_x,
                goal_y=env.goal_y,
                obstacles=env.obstacle_manager.obstacles,
                path_history=path_hist,
                sensor_readings=env.sensors.readings,
                sensor_angles_global=sensor_global_angles,
                sensor_max_range=env.sensors.max_range,
                show_sensors=show_sensors,
                show_path=show_path,
            )
            sim_placeholder.plotly_chart(fig, use_container_width=True)

            # Stats
            dist_to_goal = env.robot.distance_to(env.goal_x, env.goal_y)
            with stats_placeholder.container():
                st.metric("Step", f"{step}/{num_steps}")
                st.metric("Distance to Goal", f"{dist_to_goal:.1f} px")
                st.metric("Episode Reward", f"{episode_reward:.1f}")
                st.metric("Robot Position", f"({env.robot.x:.0f}, {env.robot.y:.0f})")
                action_names = {0: "Forward", 1: "Turn Left", 2: "Turn Right", 3: "Stop"}
                st.metric("Action", action_names.get(action, "?"))

            if step_delay > 0:
                time.sleep(step_delay / 1000.0)

            if done:
                ep_metrics = ep_info.get("episode", {})
                if ep_metrics.get("success"):
                    result_text = "✅ **Goal Reached!**"
                elif ep_metrics.get("collision"):
                    result_text = "💥 **Collision!** Robot stopped."
                else:
                    result_text = "⏰ **Timeout.** Max steps reached."

        if result_text:
            st.markdown(f"### {result_text}")
            st.markdown(f"**Total steps:** {step} | **Path length:** {env.robot.total_distance:.1f}px | **Episode reward:** {episode_reward:.1f}")

        env.close()
        st.session_state["sim_running"] = False

    except Exception as e:
        st.error(f"Simulation error: {e}")
        st.exception(e)
        st.session_state["sim_running"] = False

