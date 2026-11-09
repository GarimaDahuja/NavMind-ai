"""
NavMind AI - Performance Analytics Page
Detailed performance analysis and algorithm comparison.
"""

import sys
import json
import warnings
warnings.filterwarnings("ignore", message=".*use_container_width.*")

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="NavMind AI — Analytics", page_icon="📈", layout="wide")

st.markdown("# 📈 Performance Analytics")
st.markdown("*In-depth analysis of navigation agent performance*")
st.divider()

# ─── Load Data ────────────────────────────────────────────────────────────────
results_dir = Path("results")

def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

dqn_data = load_json(results_dir / "dqn_training.json")
comparison_data = load_json(results_dir / "comparison.json")

# ─── Algorithm Comparison ─────────────────────────────────────────────────────
st.markdown("## ⚖️ Algorithm Comparison")

if comparison_data:
    algorithms = list(comparison_data.keys())
    colors = {"A*": "#4CAF50", "DQN": "#2196F3", "PPO": "#FF9800"}

    # Success Rate comparison
    col1, col2 = st.columns(2)
    with col1:
        success_vals = [comparison_data[a].get("success_rate", 0) for a in algorithms]
        fig = go.Figure(go.Bar(
            x=algorithms, y=success_vals,
            marker_color=[colors.get(a, "#888") for a in algorithms],
            text=[f"{v:.1f}%" for v in success_vals],
            textposition="outside",
        ))
        fig.update_layout(
            title="✅ Success Rate (%)",
            yaxis=dict(range=[0, 110]),
            template="plotly_dark", height=350,
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        collision_vals = [comparison_data[a].get("collision_rate", 0) for a in algorithms]
        fig2 = go.Figure(go.Bar(
            x=algorithms, y=collision_vals,
            marker_color=["tomato"] * len(algorithms),
            text=[f"{v:.1f}%" for v in collision_vals],
            textposition="outside",
        ))
        fig2.update_layout(
            title="💥 Collision Rate (%)",
            yaxis=dict(range=[0, 110]),
            template="plotly_dark", height=350,
        )
        st.plotly_chart(fig2, width="stretch")

    # Steps and Path Length
    col3, col4 = st.columns(2)
    with col3:
        steps_vals = [comparison_data[a].get("avg_steps", 0) for a in algorithms]
        fig3 = go.Figure(go.Bar(
            x=algorithms, y=steps_vals,
            marker_color=["mediumpurple"] * len(algorithms),
            text=[f"{v:.0f}" for v in steps_vals],
            textposition="outside",
        ))
        fig3.update_layout(
            title="🔢 Avg Steps to Goal",
            template="plotly_dark", height=300,
        )
        st.plotly_chart(fig3, width="stretch")

    with col4:
        path_vals = [comparison_data[a].get("avg_path_length", 0) for a in algorithms]
        fig4 = go.Figure(go.Bar(
            x=algorithms, y=path_vals,
            marker_color=["steelblue"] * len(algorithms),
            text=[f"{v:.0f}px" for v in path_vals],
            textposition="outside",
        ))
        fig4.update_layout(
            title="📏 Avg Path Length (px)",
            template="plotly_dark", height=300,
        )
        st.plotly_chart(fig4, width="stretch")

    # Summary table
    st.markdown("### 📋 Summary Table")
    table_data = {
        "Algorithm": algorithms,
        "Success Rate": [f"{comparison_data[a].get('success_rate', 0):.1f}%" for a in algorithms],
        "Collision Rate": [f"{comparison_data[a].get('collision_rate', 0):.1f}%" for a in algorithms],
        "Avg Steps": [f"{comparison_data[a].get('avg_steps', 0):.0f}" for a in algorithms],
        "Avg Path Length": [f"{comparison_data[a].get('avg_path_length', 0):.1f}px" for a in algorithms],
        "Eval Time (s)": [f"{comparison_data[a].get('evaluation_time_s', 0):.1f}s" for a in algorithms],
    }
    st.dataframe(table_data, width="stretch")

else:
    st.info("📌 No comparison data found. Run the evaluation to compare algorithms.")
    st.code(
        "python reinforcement_learning/evaluate.py --dqn models/dqn_navmind.pth --episodes 50",
        language="bash"
    )

st.divider()

# ─── DQN Training Analysis ───────────────────────────────────────────────────
if dqn_data:
    st.markdown("## 🧠 DQN Training Deep Dive")

    rewards = dqn_data.get("episode_rewards", [])
    successes = dqn_data.get("success_flags", [])
    collisions = dqn_data.get("collision_flags", [])
    losses = dqn_data.get("losses", [])
    n = len(rewards)

    # Reward distribution
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            x=rewards,
            nbins=50,
            title="Reward Distribution",
            labels={"x": "Episode Reward", "count": "Frequency"},
            color_discrete_sequence=["steelblue"],
            template="plotly_dark",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")

    with col2:
        # Cumulative success
        cum_success = np.cumsum(successes)
        fig2 = go.Figure(go.Scatter(
            y=cum_success,
            mode="lines",
            name="Cumulative Successes",
            line=dict(color="limegreen", width=2),
            fill="tozeroy",
        ))
        fig2.update_layout(
            title="Cumulative Successful Episodes",
            xaxis_title="Episode",
            yaxis_title="Total Successes",
            template="plotly_dark",
            height=300,
        )
        st.plotly_chart(fig2, width="stretch")

    # Training phases
    if n >= 100:
        third = n // 3
        phases = {
            "Early (1–33%)": rewards[:third],
            "Mid (33–67%)": rewards[third:2*third],
            "Late (67–100%)": rewards[2*third:],
        }
        phase_means = {k: np.mean(v) for k, v in phases.items()}
        phase_success = {
            "Early": np.mean(successes[:third]) * 100,
            "Mid": np.mean(successes[third:2*third]) * 100,
            "Late": np.mean(successes[2*third:]) * 100,
        }

        col3, col4 = st.columns(2)
        with col3:
            fig3 = go.Figure(go.Bar(
                x=list(phase_means.keys()),
                y=list(phase_means.values()),
                marker_color=["#E74C3C", "#F39C12", "#27AE60"],
                text=[f"{v:.1f}" for v in phase_means.values()],
                textposition="outside",
            ))
            fig3.update_layout(
                title="Avg Reward by Training Phase",
                template="plotly_dark", height=300,
            )
            st.plotly_chart(fig3, width="stretch")

        with col4:
            fig4 = go.Figure(go.Bar(
                x=list(phase_success.keys()),
                y=list(phase_success.values()),
                marker_color=["#E74C3C", "#F39C12", "#27AE60"],
                text=[f"{v:.1f}%" for v in phase_success.values()],
                textposition="outside",
            ))
            fig4.update_layout(
                title="Success Rate by Training Phase",
                yaxis=dict(range=[0, 110]),
                template="plotly_dark", height=300,
            )
            st.plotly_chart(fig4, width="stretch")

    # Training summary stats
    st.markdown("### 📊 Training Statistics Summary")
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Total Episodes", f"{n:,}")
    col6.metric("Total Successes", f"{sum(successes):,}")
    col7.metric("Total Collisions", f"{sum(collisions):,}")
    col8.metric("Best Avg Reward", f"{max(rewards):.1f}" if rewards else "—")

else:
    st.info("📌 Train the DQN agent to see detailed analytics here.")

# ─── Research Questions Analysis ──────────────────────────────────────────────
st.divider()
st.markdown("## 🔬 Research Questions")

research_questions = [
    "Does DQN learn better navigation strategies than A* under dynamic obstacles?",
    "How does obstacle density affect the RL agent's success rate?",
    "Does training on randomized maps improve generalization?",
    "How do different reward functions affect collision avoidance?",
    "How does simulated sensor noise degrade agent performance?",
]

for i, q in enumerate(research_questions, 1):
    with st.expander(f"Q{i}: {q}"):
        st.markdown(f"""
        **Experiment Design:**
        - Train DQN/PPO on standard environment
        - Evaluate on controlled test conditions
        - Compare success rate, collision rate, path length

        **How to Test:**
        ```python
        python reinforcement_learning/evaluate.py \\
            --dqn models/dqn_navmind.pth \\
            --episodes 100 \\
            --difficulty {'dynamic' if 'dynamic' in q.lower() else 'medium'}
        ```
        
        **Status:** ⏳ Awaiting training + evaluation data
        """)

