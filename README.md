# 🤖 NavMind AI — Voice-Controlled Autonomous Navigation Simulator

[![NavMind AI CI](https://github.com/GarimaDahuja/NavMind-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/GarimaDahuja/NavMind-ai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Deep Reinforcement Learning + Natural Language Processing + Speech Processing**
> 
> A software-first autonomous mobile robot navigation system where the robot learns to navigate through environments, avoid obstacles, and reach targets using Reinforcement Learning — controlled by your voice.

---

## 🎯 Project Overview

NavMind AI combines three cutting-edge AI disciplines:

| Module | Technology | Purpose |
|--------|-----------|---------|
| **Navigation Environment** | Pygame + Gymnasium + BARN | 2D robot kinematics & benchmark maps |
| **Reinforcement Learning** | Double Dueling DQN, PPO (PyTorch) | Autonomous obstacle avoidance & navigation |
| **Path Planning** | 8-Directional A* Algorithm | Optimal classical baseline comparison |
| **NLP** | DistilBERT + Rule-Based Engine | Intent classification & slot extraction |
| **Speech** | Whisper Tiny + pyttsx3 | Voice command interface & audio feedback |
| **Frontend** | Streamlit + Plotly | Interactive multi-page dashboard |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                     │
│  Dashboard │ Environment │ Voice │ Simulator │ Analytics │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
  ┌──────▼──────┐           ┌────────▼────────┐
  │ Speech      │           │ NLP Module      │
  │ Whisper STT │           │ DistilBERT      │
  │ pyttsx3 TTS │           │ Intent + Slots  │
  └──────┬──────┘           └────────┬────────┘
         │                           │
         └───────────┬───────────────┘
                     │
          ┌──────────▼──────────┐
          │  Navigation Planner │
          │  Goal + Constraints │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   RL Agent          │
          │ Double Dueling DQN  │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   Robot Environment │
          │   Pygame / Gymnasium│
          │   BARN Benchmarks   │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Hardware Adapter   │
          │  NVIDIA Jetson Nano │
          └─────────────────────┘
```

---

## 📁 Project Structure

```
NavMind-AI/
├── .github/workflows/
│   └── ci.yml                  # GitHub Actions CI automated testing
├── docs/
│   ├── PROJECT_REPORT.md       # Comprehensive academic project report
│   └── PRESENTATION_GUIDE.md   # Presentation script & demo talking points
├── app.py                      # Streamlit main entry point
├── demo.py                     # Headless/render CLI navigation demo
├── teleop.py                   # Keyboard manual driving (Pygame teleop)
├── verify.py                   # System & GPU verification check
│
├── pages/
│   ├── 1_Dashboard.py          # Overview & statistics
│   ├── 2_Environment.py        # Map editor & BARN benchmark loader
│   ├── 3_Voice_Assistant.py    # Voice/text command interface
│   ├── 4_Navigation.py         # Live navigation simulator
│   ├── 5_RL_Training.py        # In-browser training dashboard
│   └── 6_Analytics.py          # Performance analytics & comparisons
│
├── environment/
│   ├── robot_env.py            # Gymnasium environment (NavMind-v0)
│   ├── robot.py                # Differential-drive robot kinematics
│   ├── obstacles.py            # Static and dynamic bouncing obstacles
│   ├── sensors.py              # 5-beam distance sensor / LiDAR ray cast
│   ├── map_generator.py        # Procedural random map generator
│   └── barn_loader.py          # ICRA 2022 BARN benchmark dataset loader
│
├── reinforcement_learning/
│   ├── dqn.py                  # Double Dueling DQN agent (PyTorch)
│   ├── ppo.py                  # PPO wrapper (Stable-Baselines3)
│   ├── replay_buffer.py        # Uniform & Prioritized Replay Buffer
│   ├── train.py                # Training loop with metrics logging
│   └── evaluate.py             # Evaluation & comparison benchmarks
│
├── navigation/
│   ├── astar.py                # 8-directional A* path planner
│   ├── planner.py              # Task planner resolving NLP to coords
│   └── controller.py           # Pure-pursuit waypoint follower
│
├── nlp/
│   ├── intent_classifier.py    # Hybrid rule + DistilBERT classifier
│   ├── slot_extractor.py       # Slot entity extraction (destination/speed)
│   ├── command_data.py         # Intent documentation & samples
│   └── commands.csv            # 80 annotated navigation commands
│
├── speech/
│   ├── speech_to_text.py       # Whisper Tiny speech-to-text
│   └── text_to_speech.py       # pyttsx3 offline voice synthesis
│
├── hardware/
│   ├── simulation_adapter.py   # Simulation bridge interface
│   └── jetson_adapter.py       # Jetson Nano hardware bridge scaffold
│
├── experiments/
│   └── run_experiments.py      # Automated runner for 5 research questions
├── tests/
│   └── test_all.py             # Automated unit test suite (12 test cases)
├── models/                     # Saved model checkpoints
├── results/                    # Exported metrics & evaluation summaries
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Recommended Conda Environment (Python 3.10 for full RTX 3050 CUDA GPU support):
conda create -n navmind python=3.10 -y
conda activate navmind

# Install dependencies:
pip install -r requirements.txt
```

### 2. Verify Installation & Tests
```bash
# Run system check
python verify.py

# Run automated test suite (12 unit tests)
python tests/test_all.py
```

### 3. Drive the Robot Manually (Keyboard Teleoperation)
```bash
python teleop.py --difficulty medium
```
* **`W` / `Up`**: Move Forward
* **`A` / `Left`**: Turn Left
* **`D` / `Right`**: Turn Right
* **`S` / `Down`**: Stop
* **`R`**: Reset map

### 4. Run Headless / CLI Simulation
```bash
# Test classical A* path planning:
python demo.py --algorithm astar --episodes 3

# Test random exploration:
python demo.py --algorithm random --episodes 3
```

### 5. Launch the Streamlit Dashboard
```bash
python -m streamlit run app.py
```

---

## 🧠 Reinforcement Learning Formulation

### State Space ($\mathcal{S} \subset \mathbb{R}^7$)
$$S_t = \left[ \frac{d_g}{D_{\max}}, \frac{\theta_g}{\pi}, \frac{s_f}{R_{\max}}, \frac{s_{fl}}{R_{\max}}, \frac{s_{fr}}{R_{\max}}, \frac{s_l}{R_{\max}}, \frac{s_r}{R_{\max}} \right]$$

### Action Space ($\mathcal{A}$)
Discrete movement primitives:
* `0`: Move Forward
* `1`: Turn Left ($0.2\text{ rad}$)
* `2`: Turn Right ($0.2\text{ rad}$)
* `3`: Stop

### Potential-Based Dense Reward Function
$$R(s, a, s') = \begin{cases}
+100.0 & \text{Goal reached } (d_g < 25\text{px}) \\
-100.0 & \text{Collision detected} \\
+5.0 \times \Delta d_g / 10 & \text{Progress toward target} \\
-2.0 \times \Delta d_g / 10 & \text{Moving away from target} \\
-0.1 & \text{Per-step penalty} \\
-20.0 & \text{Timeout (exceeding maximum steps)}
\end{cases}$$

---

## 🔬 Research Experiments & Benchmarks

Run the automated suite to benchmark the 5 core research questions:
```bash
python experiments/run_experiments.py
```

* **Q1: Dynamic Obstacles:** A* vs. DQN performance in environments with moving hazards.
* **Q2: Obstacle Density:** Impact of clutter (`easy` vs `medium` vs `hard`).
* **Q3: Generalization on BARN:** Testing zero-shot navigation on unseen ICRA 2022 BARN benchmark maps.
* **Q4: Sensor Noise:** Policy degradation under simulated Gaussian sensor noise ($\sigma \in [0, 10\text{px}]$).

---

## 🎤 Voice & NLP Capabilities

| User Command | Intent | Extracted Entities | Action |
|:---|:---|:---|:---|
| *"Navigate to the charging station"* | `NAVIGATE` | Goal: `charging station` | Plans path to $(100, 100)$ |
| *"Move slowly to the office avoiding obstacles"* | `NAVIGATE` | Goal: `office`, Speed: `0.5x`, Constraint: `avoid_obstacles` | Restricts max velocity & avoids hazards |
| *"Emergency stop immediately"* | `STOP` | Action: `stop` | Halts robot motors immediately |
| *"Turn left"* | `DIRECTION` | Direction: `left` | Executes heading rotation |
| *"Find the shortest route"* | `OPTIMIZE` | Objective: `distance` | Executes A* minimum-distance route |

---

## 📄 Documentation

* **[Academic Project Report](docs/PROJECT_REPORT.md)**: Mathematical formulation, Bellman targets, Double Dueling DQN derivation, and experimental analysis.
* **[Presentation & Demo Guide](docs/PRESENTATION_GUIDE.md)**: 5-minute presentation script, live demo walkthrough, and defense Q&A.

---

## 👩‍💻 Author

**Garima Dahuja**  
NavMind AI — Voice-Controlled Autonomous Robot Navigation Using Deep Reinforcement Learning  
*GitHub: [@GarimaDahuja](https://github.com/GarimaDahuja)*
