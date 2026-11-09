"""
NavMind AI - Main Streamlit Application Entry Point
Voice-Controlled Autonomous Robot Navigation Using Deep Reinforcement Learning
"""

import warnings
warnings.filterwarnings("ignore", message=".*use_container_width.*")

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="NavMind AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/GarimaDahuja/NavMind-ai",
        "Report a bug": "https://github.com/GarimaDahuja/NavMind-ai/issues",
        "About": "NavMind AI — Voice-Controlled Autonomous Robot Navigation",
    },
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .metric-card {
        background: #0f3460;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # Hero section
    st.markdown('<h1 class="main-header">🤖 NavMind AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Voice-Controlled Autonomous Robot Navigation Using Deep Reinforcement Learning</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Quick navigation
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🗺️ Quick Actions")
        if st.button("🚀 Launch Simulator", width="stretch"):
            st.switch_page("pages/4_Navigation.py")
        if st.button("🎤 Voice Assistant", width="stretch"):
            st.switch_page("pages/3_Voice_Assistant.py")
        if st.button("🧠 Train RL Agent", width="stretch"):
            st.switch_page("pages/5_RL_Training.py")

    with col2:
        st.markdown("### 📊 System Status")
        import importlib.util

        # PyTorch status
        try:
            if importlib.util.find_spec("torch") is not None:
                torch_mod = importlib.import_module("torch")
                if torch_mod.cuda.is_available():
                    gpu_name = torch_mod.cuda.get_device_name(0)
                    st.success(f"✅ GPU: {gpu_name}")
                else:
                    st.warning("⚠️ Running on CPU")
            else:
                st.error("❌ PyTorch not installed")
        except Exception:
            st.error("❌ PyTorch error")

        # Gymnasium status
        try:
            if importlib.util.find_spec("gymnasium") is not None:
                gym_mod = importlib.import_module("gymnasium")
                ver = getattr(gym_mod, "__version__", "")
                st.success(f"✅ Gymnasium {ver}".strip())
            else:
                st.error("❌ Gymnasium not installed")
        except Exception:
            st.error("❌ Gymnasium error")

        # Stable-Baselines3 status
        try:
            if importlib.util.find_spec("stable_baselines3") is not None:
                sb3_mod = importlib.import_module("stable_baselines3")
                ver = getattr(sb3_mod, "__version__", "")
                st.success(f"✅ SB3 {ver}".strip())
            else:
                st.info("ℹ️ Stable-Baselines3 optional (for PPO)")
        except Exception:
            st.info("ℹ️ Stable-Baselines3 optional")

    with col3:
        st.markdown("### 🏆 Project Info")
        st.info(
            """
            **NavMind AI**
            
            Semester Project — AI & Robotics
            
            **Algorithms:** DQN, PPO, A*
            
            **Features:**
            - 🗣️ Voice Commands (Whisper)
            - 🧠 NLP (DistilBERT)
            - 🤖 Deep RL Navigation
            - 📈 Real-time Analytics
            """
        )

    st.divider()

    # Feature overview
    st.markdown("## 🎯 System Modules")

    modules = [
        {
            "icon": "🌍",
            "name": "Navigation Environment",
            "desc": "2D Pygame + Gymnasium environment with robot kinematics, collision detection, and sensor simulation.",
            "tech": "Pygame · Gymnasium · NumPy",
        },
        {
            "icon": "🧠",
            "name": "Reinforcement Learning",
            "desc": "DQN and PPO agents learn to navigate from scratch through trial and error with custom reward shaping.",
            "tech": "PyTorch · DQN · PPO · SB3",
        },
        {
            "icon": "🗺️",
            "name": "Path Planning",
            "desc": "A* algorithm provides an optimal baseline for comparison with RL-learned navigation policies.",
            "tech": "A* · PythonRobotics",
        },
        {
            "icon": "💬",
            "name": "NLP Module",
            "desc": "DistilBERT-based intent classifier extracts navigation commands, destinations and constraints.",
            "tech": "DistilBERT · HuggingFace · scikit-learn",
        },
        {
            "icon": "🎤",
            "name": "Speech Processing",
            "desc": "Whisper Tiny converts voice to text; pyttsx3 provides spoken feedback on navigation status.",
            "tech": "Whisper · pyttsx3",
        },
        {
            "icon": "📊",
            "name": "Analytics Dashboard",
            "desc": "Real-time performance metrics: success rate, collision rate, path efficiency, training curves.",
            "tech": "Streamlit · Plotly · Matplotlib",
        },
    ]

    cols = st.columns(3)
    for i, module in enumerate(modules):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="feature-card">
                    <h3>{module['icon']} {module['name']}</h3>
                    <p>{module['desc']}</p>
                    <small style="color: #667eea;">🔧 {module['tech']}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Architecture diagram
    st.markdown("## 🏗️ System Architecture")
    st.code(
        """
    ┌─────────────────────────────────────────────────────────┐
    │                   Streamlit Frontend                     │
    │  Dashboard │ Environment │ Voice │ Simulator │ Analytics │
    └──────────────────────┬──────────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
      ┌──────▼──────┐           ┌────────▼────────┐
      │ Speech STT  │           │ NLP Module      │
      │ Whisper     │           │ DistilBERT      │
      │ pyttsx3 TTS │           │ Intent + Slots  │
      └──────┬──────┘           └────────┬────────┘
             │                           │
             └───────────┬───────────────┘
                         │
              ┌──────────▼──────────┐
              │  Navigation Planner │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   RL Agent (DQN/PPO)│
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   Gymnasium Env     │
              │   Pygame Renderer   │
              └─────────────────────┘
    """,
        language="text",
    )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            NavMind AI<br>
            <a href="https://github.com/GarimaDahuja/NavMind-ai" target="_blank">GitHub Repository</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

