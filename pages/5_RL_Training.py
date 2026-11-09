"""
NavMind AI - RL Training Page
Interactive training dashboard for DQN and PPO agents.
"""

import sys
import json
import time
import threading
import warnings
warnings.filterwarnings("ignore", message=".*use_container_width.*")

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="NavMind AI — RL Training", page_icon="🧠", layout="wide")

st.markdown("# 🧠 Reinforcement Learning Training")
st.markdown("*Train DQN and PPO agents to navigate autonomously*")
st.divider()

# ─── Training Configuration ──────────────────────────────────────────────────
col_cfg, col_prog = st.columns([1, 2])

with col_cfg:
    st.markdown("### ⚙️ Training Configuration")

    algorithm = st.selectbox("Algorithm", ["DQN (Recommended)", "PPO"], index=0)

    difficulty = st.selectbox(
        "Environment Difficulty",
        ["easy", "medium", "hard", "dynamic", "maze"],
        index=1,
    )

    if algorithm == "DQN (Recommended)":
        num_episodes = st.slider("Training Episodes", 100, 2000, 500, step=100)
        lr = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
        gamma = st.slider("Discount Factor (γ)", 0.90, 0.999, 0.99)
        epsilon_decay = st.slider("Epsilon Decay", 0.990, 0.999, 0.995, step=0.001)
        batch_size = st.selectbox("Batch Size", [32, 64, 128], index=1)
        use_dueling = st.checkbox("Dueling DQN", value=True)
    else:
        total_timesteps = st.slider("Total Timesteps", 10_000, 500_000, 100_000, step=10_000)
        lr = st.select_slider("Learning Rate", options=[1e-4, 3e-4, 1e-3], value=3e-4)
        n_steps = st.selectbox("Steps per Rollout", [512, 1024, 2048], index=2)

    dynamic_obs = st.checkbox("Dynamic Obstacles", value=False)
    sensor_noise = st.slider("Sensor Noise", 0.0, 10.0, 0.0, step=0.5)

    start_training = st.button("🚀 Start Training", type="primary", width="stretch")
    st.markdown("---")
    st.info(
        "💡 **Tip:** Start with DQN on 'easy' for 300 episodes "
        "to verify everything works. Then switch to 'medium' for full training."
    )

with col_prog:
    st.markdown("### 📈 Training Progress")

    progress_bar = st.progress(0, text="Ready to train...")
    reward_chart = st.empty()
    success_chart = st.empty()
    metrics_row = st.empty()

# ─── Training ────────────────────────────────────────────────────────────────
if start_training:
    if algorithm == "DQN (Recommended)":
        from environment.robot_env import RobotNavEnv
        from reinforcement_learning.dqn import DQNAgent
        from collections import deque

        env = RobotNavEnv(
            render_mode=None,
            difficulty=difficulty,
            dynamic_obstacles=dynamic_obs,
            sensor_noise=sensor_noise,
        )
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            lr=float(lr),
            gamma=float(gamma),
            epsilon_decay=float(epsilon_decay),
            batch_size=int(batch_size),
            dueling=bool(use_dueling),
        )

        episode_rewards = []
        success_flags = []
        recent_rewards = deque(maxlen=100)
        recent_success = deque(maxlen=100)
        losses = []

        Path("models").mkdir(exist_ok=True)

        for episode in range(1, num_episodes + 1):
            state, info = env.reset()
            ep_reward = 0.0
            ep_losses = []
            done = False

            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, ep_info = env.step(action)
                done = terminated or truncated
                agent.remember(state, action, reward, next_state, done)
                loss = agent.learn()
                if loss:
                    ep_losses.append(loss)
                ep_reward += reward
                state = next_state

            ep_metrics = ep_info.get("episode", {})
            success = ep_metrics.get("success", False)
            episode_rewards.append(ep_reward)
            success_flags.append(int(success))
            recent_rewards.append(ep_reward)
            recent_success.append(int(success))
            if ep_losses:
                losses.append(float(np.mean(ep_losses)))

            agent.decay_epsilon()
            if episode % 10 == 0:
                agent.update_target_network()

            # Save best model
            if episode % 50 == 0 and len(recent_success) >= 50:
                agent.save("models/dqn_navmind.pth")

            # Update UI every 5 episodes
            if episode % 5 == 0:
                progress = episode / num_episodes
                progress_bar.progress(
                    progress,
                    text=f"Episode {episode}/{num_episodes} | ε={agent.epsilon:.3f}",
                )

                # Reward chart
                window = max(10, len(episode_rewards) // 20)
                smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(
                    y=episode_rewards, name="Raw", opacity=0.3,
                    line=dict(color="steelblue", width=1),
                ))
                fig_r.add_trace(go.Scatter(
                    y=smoothed, name="Smoothed",
                    line=dict(color="orange", width=2),
                ))
                fig_r.update_layout(
                    title="Episode Reward",
                    template="plotly_dark", height=250,
                    margin=dict(l=40, r=10, t=40, b=40),
                )
                reward_chart.plotly_chart(fig_r, width="stretch")

                # Success rate chart
                win2 = max(5, len(success_flags) // 10)
                rolling_succ = [
                    np.mean(success_flags[max(0, i - win2):i + 1]) * 100
                    for i in range(len(success_flags))
                ]
                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(
                    y=rolling_succ, name="Success Rate %",
                    fill="tozeroy",
                    line=dict(color="mediumseagreen", width=2),
                ))
                fig_s.update_layout(
                    title="Rolling Success Rate",
                    yaxis=dict(range=[0, 100]),
                    template="plotly_dark", height=250,
                    margin=dict(l=40, r=10, t=40, b=40),
                )
                success_chart.plotly_chart(fig_s, width="stretch")

                # Metrics
                with metrics_row.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Avg Reward (100)", f"{np.mean(recent_rewards):.1f}")
                    m2.metric("Success Rate", f"{np.mean(recent_success)*100:.1f}%")
                    m3.metric("Epsilon", f"{agent.epsilon:.3f}")
                    m4.metric("Buffer Size", f"{len(agent.memory):,}")

        agent.save("models/dqn_navmind.pth")
        env.close()
        progress_bar.progress(1.0, text="✅ Training Complete!")
        st.success(
            f"🎉 DQN Training Complete! "
            f"Final success rate: {np.mean(success_flags[-100:])*100:.1f}%"
        )

        # Save results
        Path("results").mkdir(exist_ok=True)
        with open("results/dqn_training.json", "w") as f:
            json.dump({
                "algorithm": "DQN",
                "num_episodes": num_episodes,
                "difficulty": difficulty,
                "episode_rewards": episode_rewards,
                "success_flags": success_flags,
                "losses": losses,
            }, f)

    else:
        st.info("🔄 PPO training via Stable-Baselines3 — runs in a separate process.")
        st.code(
            f"python reinforcement_learning/train.py "
            f"--algorithm ppo --timesteps {total_timesteps} --difficulty {difficulty}",
            language="bash",
        )
        st.warning("Launch this command in your terminal with the conda environment active.")

# ─── Load Existing Results ────────────────────────────────────────────────────
st.divider()
st.markdown("### 📂 Load Saved Training Results")
results_files = list(Path("results").glob("*.json")) if Path("results").exists() else []

if results_files:
    selected_result = st.selectbox(
        "Select results file:", [f.name for f in results_files]
    )
    if st.button("📊 Load & Visualize"):
        with open(Path("results") / selected_result) as f:
            data = json.load(f)
        st.json({k: v for k, v in data.items() if not isinstance(v, list)})
        st.success(f"Loaded {len(data.get('episode_rewards', []))} episodes.")
else:
    st.info("No results files found. Train an agent first.")

