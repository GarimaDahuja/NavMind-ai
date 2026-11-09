"""
NavMind AI - Environment Editor Page
Create custom maps, place obstacles, and configure navigation environments.
"""

import sys
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="NavMind AI — Environment", page_icon="🗺️", layout="wide")

st.markdown("# 🗺️ Navigation Environment Editor")
st.markdown("*Design custom maps, place obstacles, and configure the robot environment*")
st.divider()

# ─── Environment Configuration ──────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Environment Settings")

    difficulty = st.selectbox(
        "Difficulty Level",
        ["easy", "medium", "hard", "dynamic", "maze"],
        index=1,
        help="Controls number and complexity of obstacles"
    )

    num_obstacles = st.slider("Number of Obstacles", 2, 20, 8)
    dynamic_obs = st.checkbox("Enable Dynamic Obstacles", value=False)
    sensor_noise = st.slider("Sensor Noise (std dev)", 0.0, 15.0, 0.0, step=0.5)
    seed = st.number_input("Random Seed", value=42, min_value=0, max_value=9999)

    st.divider()
    st.markdown("### 🤖 Robot Settings")
    start_x = st.slider("Start X", 50, 550, 80)
    start_y = st.slider("Start Y", 50, 550, 80)
    goal_x = st.slider("Goal X", 50, 550, 500)
    goal_y = st.slider("Goal Y", 50, 550, 500)

    generate_btn = st.button("🗺️ Generate Map", use_container_width=True, type="primary")

    st.divider()
    predefined = st.selectbox(
        "Predefined Maps",
        ["Custom", "open_field", "corridor", "cluttered"],
        help="Load a predefined test map"
    )
    load_predefined = st.button("📂 Load Predefined Map", use_container_width=True)

with col2:
    st.markdown("### 🌍 Environment Preview")

    if "current_map" not in st.session_state:
        st.session_state.current_map = None

    if generate_btn or load_predefined:
        try:
            from environment.map_generator import MapGenerator
            from environment.obstacles import ObstacleManager

            gen = MapGenerator(600, 600)
            np.random.seed(int(seed))

            if load_predefined and predefined != "Custom":
                predefined_maps = gen.get_predefined_maps()
                grid = predefined_maps[predefined]
                manager = gen.generate_from_grid(grid, cell_size=60)
                robot_pos = (45, 45)
                goal_p = (555, 555)
            else:
                manager, robot_pos, goal_p = gen.generate(
                    difficulty=difficulty,
                    seed=int(seed),
                    robot_pos=(float(start_x), float(start_y)),
                    goal_pos=(float(goal_x), float(goal_y)),
                )

            st.session_state.current_map = {
                "manager": manager,
                "robot_pos": robot_pos,
                "goal_pos": goal_p,
            }
            st.success("✅ Map generated successfully!")
        except Exception as e:
            st.error(f"Map generation error: {e}")

    # ─── Plotly Map Visualization ────────────────────────────────────────
    fig = go.Figure()
    fig.update_layout(
        width=580, height=580,
        xaxis=dict(range=[0, 600], showgrid=False, zeroline=False),
        yaxis=dict(range=[0, 600], showgrid=False, zeroline=False, scaleanchor="x"),
        plot_bgcolor="#141428",
        paper_bgcolor="#0d1117",
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        title="Navigation Environment (600×600)",
    )

    if st.session_state.current_map:
        manager = st.session_state.current_map["manager"]
        rp = st.session_state.current_map["robot_pos"]
        gp = st.session_state.current_map["goal_pos"]

        # Draw obstacles
        for obs in manager.obstacles:
            color = "rgba(100,100,180,0.7)" if not obs.is_dynamic else "rgba(200,80,80,0.7)"
            fig.add_shape(
                type="rect",
                x0=obs.x, y0=obs.y,
                x1=obs.x + obs.width, y1=obs.y + obs.height,
                fillcolor=color,
                line=dict(color="white", width=0.5),
            )

        # Robot start
        fig.add_trace(go.Scatter(
            x=[rp[0]], y=[rp[1]],
            mode="markers+text",
            marker=dict(size=20, color="dodgerblue", symbol="circle"),
            text=["🤖 Start"], textposition="top center",
            name="Robot Start",
        ))

        # Goal
        fig.add_trace(go.Scatter(
            x=[gp[0]], y=[gp[1]],
            mode="markers+text",
            marker=dict(size=20, color="limegreen", symbol="star"),
            text=["⭐ Goal"], textposition="top center",
            name="Goal",
        ))

        # Obstacle legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color="rgba(100,100,180,0.7)", symbol="square"),
            name="Static Obstacle",
        ))
        if dynamic_obs:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color="rgba(200,80,80,0.7)", symbol="square"),
                name="Dynamic Obstacle",
            ))
    else:
        fig.add_annotation(
            text="Click 'Generate Map' to preview the environment",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray"),
        )

    st.plotly_chart(fig, use_container_width=True)

# ─── A* Path Preview ─────────────────────────────────────────────────────────
st.divider()
st.markdown("### 🗺️ A* Path Preview")
run_astar = st.button("🔍 Run A* Path Planning", use_container_width=True)

if run_astar and st.session_state.current_map:
    try:
        from navigation.astar import AStarPlanner

        map_data = st.session_state.current_map
        planner = AStarPlanner(600, 600, cell_size=15)
        grid = planner.build_grid(map_data["manager"])
        path = planner.plan(map_data["robot_pos"], map_data["goal_pos"], grid)

        fig2 = go.Figure()
        fig2.update_layout(
            width=580, height=580,
            xaxis=dict(range=[0, 600], showgrid=False),
            yaxis=dict(range=[0, 600], showgrid=False, scaleanchor="x"),
            plot_bgcolor="#141428", paper_bgcolor="#0d1117",
            margin=dict(l=0, r=0, t=30, b=0),
            title="A* Optimal Path",
        )

        # Obstacles
        for obs in map_data["manager"].obstacles:
            fig2.add_shape(
                type="rect",
                x0=obs.x, y0=obs.y, x1=obs.x + obs.width, y1=obs.y + obs.height,
                fillcolor="rgba(100,100,180,0.7)", line=dict(color="white", width=0.5),
            )

        if path:
            px_vals = [p[0] for p in path]
            py_vals = [p[1] for p in path]
            fig2.add_trace(go.Scatter(
                x=px_vals, y=py_vals, mode="lines+markers",
                line=dict(color="orange", width=2),
                marker=dict(size=5, color="orange"),
                name=f"A* Path ({len(path)} waypoints)",
            ))
            path_len = planner.path_length(path)
            st.success(f"✅ A* path found: {len(path)} waypoints, length = {path_len:.1f} px")
        else:
            st.warning("⚠️ No valid path found. Try adjusting obstacle placement.")

        fig2.add_trace(go.Scatter(
            x=[map_data["robot_pos"][0]], y=[map_data["robot_pos"][1]],
            mode="markers", marker=dict(size=18, color="dodgerblue"), name="Start",
        ))
        fig2.add_trace(go.Scatter(
            x=[map_data["goal_pos"][0]], y=[map_data["goal_pos"][1]],
            mode="markers", marker=dict(size=18, color="limegreen", symbol="star"), name="Goal",
        ))
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"A* planning error: {e}")
elif run_astar:
    st.info("Generate a map first.")

