# 🤖 NavMind AI — Voice-Controlled Autonomous Navigation Simulator

> **Deep Reinforcement Learning + Natural Language Processing + Speech Processing**
> 
> A software-first autonomous mobile robot navigation system where the robot learns to navigate through environments, avoid obstacles, and reach targets using Reinforcement Learning — controlled by your voice.

---

## 🎯 Project Overview

NavMind AI combines three cutting-edge AI disciplines:

| Module | Technology | Purpose |
|--------|-----------|---------|
| **Navigation Environment** | Pygame + Gymnasium | 2D robot simulation |
| **Reinforcement Learning** | DQN, PPO (PyTorch) | Autonomous navigation |
| **Path Planning** | A* Algorithm | Baseline comparison |
| **NLP** | DistilBERT | Intent & slot extraction |
| **Speech** | Whisper Tiny + pyttsx3 | Voice command interface |
| **Frontend** | Streamlit | Interactive dashboard |

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
          │   DQN / PPO         │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   Robot Environment │
          │   Pygame / Gymnasium│
          └─────────────────────┘
```

---

## 📁 Project Structure

```
NavMind-AI/
├── app.py                      # Streamlit main entry point
├── pages/
│   ├── 1_Dashboard.py          # Overview & statistics
│   ├── 2_Environment.py        # Map editor
│   ├── 3_Voice_Assistant.py    # Voice/text command interface
│   ├── 4_Navigation.py         # Live navigation simulator
│   ├── 5_RL_Training.py        # Training dashboard
│   └── 6_Analytics.py          # Performance analytics
│
├── environment/
│   ├── robot_env.py            # Gymnasium environment
│   ├── robot.py                # Robot model & kinematics
│   ├── obstacles.py            # Obstacle management
│   ├── sensors.py              # Distance sensors / LiDAR
│   └── map_generator.py        # Random map generation
│
├── reinforcement_learning/
│   ├── dqn.py                  # DQN agent (PyTorch)
│   ├── ppo.py                  # PPO wrapper (SB3)
│   ├── replay_buffer.py        # Experience replay
│   ├── train.py                # Training loop
│   └── evaluate.py             # Evaluation metrics
│
├── navigation/
│   ├── astar.py                # A* path planning
│   ├── planner.py              # Navigation task planner
│   └── controller.py           # Movement controller
│
├── nlp/
│   ├── intent_classifier.py    # DistilBERT intent classifier
│   ├── slot_extractor.py       # Slot/entity extraction
│   └── commands.csv            # Training command dataset
│
├── speech/
│   ├── speech_to_text.py       # Whisper STT
│   └── text_to_speech.py       # pyttsx3 TTS
│
├── hardware/
│   ├── simulation_adapter.py   # Simulation interface
│   └── jetson_adapter.py       # Jetson Nano interface (future)
│
├── models/                     # Saved model weights
├── assets/                     # Images, icons
├── results/                    # Evaluation results
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Create Conda Environment
```bash
conda create -n navmind python=3.10 -y
conda activate navmind
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify GPU
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 4. Train the RL Agent
```bash
python reinforcement_learning/train.py --algorithm dqn --episodes 1000
```

### 5. Launch Streamlit App
```bash
streamlit run app.py
```

---

## 🧠 Reinforcement Learning

### State Space
$$S_t = [d_g, \theta_g, d_f, d_l, d_r, v, a_{prev}]$$

| Variable | Description |
|----------|-------------|
| $d_g$ | Distance to goal |
| $\theta_g$ | Angle to goal |
| $d_f$ | Front obstacle distance |
| $d_l$ | Left obstacle distance |
| $d_r$ | Right obstacle distance |
| $v$ | Robot velocity |
| $a_{prev}$ | Previous action |

### Action Space (Discrete)

| Action | Movement |
|--------|----------|
| 0 | Move Forward |
| 1 | Turn Left |
| 2 | Turn Right |
| 3 | Stop |

### Reward Function

| Condition | Reward |
|-----------|--------|
| Goal reached | +100 |
| Collision | -100 |
| Moves closer to goal | +5 |
| Moves away from goal | -2 |
| Normal step | -0.1 |
| Max steps exceeded | -20 |

---

## 📊 Algorithm Comparison

| Metric | A* | DQN | PPO |
|--------|-----|-----|-----|
| Optimal path | ✅ Static only | ❌ | ❌ |
| Dynamic obstacles | ❌ | ✅ | ✅ |
| Training required | ❌ | ✅ | ✅ |
| Generalizes | ❌ | ✅ | ✅ |

---

## 🎤 Voice Commands

| Command | Intent | Extracted Info |
|---------|--------|---------------|
| "Navigate to the charging station" | NAVIGATE | Goal: charging station |
| "Move slowly to the office" | NAVIGATE | Goal: office, Speed: low |
| "Stop the robot" | STOP | Action: stop |
| "Avoid obstacles" | SET_CONSTRAINT | Constraint: avoid |
| "Find shortest route" | OPTIMIZE | Objective: shortest |

---

## 🔧 Hardware Extension (Future)

```
Simulation Mode ──────────────────► Hardware Mode
─────────────────                   ──────────────
Simulated LiDAR         →          Physical Distance Sensors
Virtual robot position  →          Odometry
Virtual obstacles       →          Camera observations
Simulated movement      →          Motor commands
Simulation adapter      →          Jetson Nano adapter
```

---

## 📈 Research Questions

1. Does DQN outperform A* under dynamic obstacle conditions?
2. How does obstacle density affect the RL agent's success rate?
3. Does training on randomized maps improve generalization?
4. How do different reward functions affect collision avoidance?
5. How does simulated sensor noise degrade agent performance?

---

## 🗓️ Development Timeline

| Week | Milestone |
|------|-----------|
| 1 | 2D Environment + Robot Simulation |
| 2 | Map Generation + A* Baseline |
| 3–4 | DQN Implementation + Training |
| 5 | PPO + Algorithm Comparison |
| 6 | NLP Integration |
| 7 | Speech Integration |
| 8 | Streamlit Application |
| 9–10 | Testing + Deployment |

---

## 📚 References

- [DRL Robot Navigation - IR-SIM](https://github.com/reiniscimurs/DRL-robot-navigation-IR-SIM)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [BARN Dataset](https://www.cs.utexas.edu/~xiao/BARN/BARN.html)
- [Whisper (OpenAI)](https://huggingface.co/openai/whisper-tiny)
- [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)

---

## 👩‍💻 Author

**Garima Dahuja** — Semester Project  
NavMind AI — Voice-Controlled Autonomous Robot Navigation Using Deep Reinforcement Learning

---

*Built with ❤️ using Python, PyTorch, Streamlit, and Gymnasium*

