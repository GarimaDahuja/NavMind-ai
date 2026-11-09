"""
NavMind AI - Dashboard Page
Overview statistics and project summary.
"""

import warnings
warnings.filterwarnings("ignore", message=".*use_container_width.*")

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path

st.set_page_config(page_title="NavMind AI — Dashboard", page_icon="📊", layout="wide")

st.markdown("# 📊 NavMind AI — Dashboard")
st.markdown("*Real-time navigation metrics and training overview*")
st.divider()

# ─── Key Metrics ────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

# Load results if available
results_dir = Path("results")
dqn_results = None
if (results_dir / "dqn_training.json").exists():
    with open(results_dir / "dqn_training.json") as f:
        dqn_results = json.load(f)

if dqn_results:
    success_flags = dqn_results.get("success_flags", [])
    collision_flags = dqn_results.get("collision_flags", [])
    rewards = dqn_results.get("episode_rewards", [])

    recent_success = np.mean(success_flags[-100:]) * 100 if success_flags else 0
    recent_collision = np.mean(collision_flags[-100:]) * 100 if collision_flags else 0
    avg_reward = np.mean(rewards[-100:]) if rewards else 0
    total_eps = len(rewards)
else:
    recent_success = 0.0
    recent_collision = 0.0
    avg_reward = 0.0
    total_eps = 0

with col1:
    st.metric("🎯 Success Rate", f"{recent_success:.1f}%",
              delta=f"+{recent_success:.1f}%" if recent_success > 50 else None)
with col2:
    st.metric("💥 Collision Rate", f"{recent_collision:.1f}%",
              delta=f"-{recent_collision:.1f}%" if recent_collision > 0 else None,
              delta_color="inverse")
with col3:
    st.metric("🏆 Avg Episode Reward", f"{avg_reward:.1f}")
with col4:
    st.metric("📈 Episodes Trained", f"{total_eps:,}")

st.divider()

# ─── Training Curves ────────────────────────────────────────────────────────
if dqn_results:
    st.markdown("## 📈 DQN Training Progress")
    col1, col2 = st.columns(2)

    with col1:
        # Reward curve
        rewards = dqn_results.get("episode_rewards", [])
        if rewards:
            # Smooth with rolling average
            window = min(50, len(rewards) // 10 + 1)
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=rewards, name="Raw Reward", opacity=0.3,
                line=dict(color="steelblue", width=1),
            ))
            fig.add_trace(go.Scatter(
                y=smoothed, name=f"Smoothed (w={window})",
                line=dict(color="orange", width=2),
            ))
            fig.update_layout(
                title="Episode Reward",
                xaxis_title="Episode",
                yaxis_title="Total Reward",
                template="plotly_dark",
                height=350,
            )
            st.plotly_chart(fig, width="stretch")

    with col2:
        # Success rate rolling average
        success = dqn_results.get("success_flags", [])
        if success:
            window = min(50, len(success) // 5 + 1)
            rolling_success = [
                np.mean(success[max(0, i - window):i + 1]) * 100
                for i in range(len(success))
            ]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                y=rolling_success,
                name="Success Rate %",
                fill="tozeroy",
                line=dict(color="mediumseagreen", width=2),
            ))
            fig2.update_layout(
                title="Rolling Success Rate",
                xaxis_title="Episode",
                yaxis_title="Success Rate (%)",
                yaxis=dict(range=[0, 100]),
                template="plotly_dark",
                height=350,
            )
            st.plotly_chart(fig2, width="stretch")

    # Loss curve
    losses = dqn_results.get("losses", [])
    if losses:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            y=losses, name="DQN Loss",
            line=dict(color="tomato", width=1),
        ))
        fig3.update_layout(
            title="DQN Training Loss",
            xaxis_title="Update Step",
            yaxis_title="Huber Loss",
            template="plotly_dark",
            height=250,
        )
        st.plotly_chart(fig3, width="stretch")

else:
    st.info("📌 No training results found yet. Train a DQN agent first via **RL Training** page.")
    st.markdown("**Quick Start Training:**")
    st.code("python reinforcement_learning/train.py --algorithm dqn --episodes 500", language="bash")

# ─── Algorithm Comparison ───────────────────────────────────────────────────
comparison_path = results_dir / "comparison.json"
if comparison_path.exists():
    st.divider()
    st.markdown("## ⚖️ Algorithm Comparison")
    with open(comparison_path) as f:
        comparison = json.load(f)

    algorithms = list(comparison.keys())
    metrics_to_show = ["success_rate", "collision_rate", "avg_steps", "avg_path_length"]
    metric_labels = ["Success Rate (%)", "Collision Rate (%)", "Avg Steps", "Avg Path Length"]

    for metric, label in zip(metrics_to_show, metric_labels):
        values = [comparison[alg].get(metric, 0) for alg in algorithms]
        fig = px.bar(
            x=algorithms, y=values,
            labels={"x": "Algorithm", "y": label},
            title=label,
            color=algorithms,
            color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_dark",
            height=300,
        )
        st.plotly_chart(fig, width="stretch")

# ─── System Info ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("## 🖥️ System Information")
col1, col2 = st.columns(2)
with col1:
    try:
        import torch
        gpu = torch.cuda.is_available()
        st.markdown(f"**PyTorch:** {torch.__version__}")
        if gpu:
            st.success(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.markdown(f"**VRAM:** {mem:.1f} GB")
        else:
            st.warning("⚠️ No GPU detected — running on CPU")
    except ImportError:
        st.error("PyTorch not installed")

with col2:
    import sys
    st.markdown(f"**Python:** {sys.version.split()[0]}")
    try:
        import gymnasium
        st.markdown(f"**Gymnasium:** {gymnasium.__version__}")
    except ImportError:
        st.error("Gymnasium not installed")
    try:
        import stable_baselines3
        st.markdown(f"**SB3:** {stable_baselines3.__version__}")
    except ImportError:
        st.warning("SB3 not installed")

