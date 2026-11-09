"""
NavMind AI - Voice Assistant Page
Voice and text command interface for robot navigation.
"""

import sys
import warnings
warnings.filterwarnings("ignore", message=".*use_container_width.*")

import streamlit as st
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="NavMind AI — Voice Assistant", page_icon="🎤", layout="wide")

st.markdown("# 🎤 Voice Assistant")
st.markdown("*Give natural language commands to control the robot navigation*")
st.divider()

# ─── Initialize NLP Pipeline ────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    from nlp.intent_classifier import IntentClassifier
    from nlp.slot_extractor import SlotExtractor
    from navigation.planner import NavigationPlanner
    return IntentClassifier(use_bert=False), SlotExtractor(), NavigationPlanner()

@st.cache_resource
def load_tts():
    from speech.text_to_speech import TextToSpeech
    tts = TextToSpeech(rate=145)
    tts.initialize()
    return tts

classifier, extractor, planner = load_nlp()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🗣️ Command Input")

    # Input method selector
    input_method = st.radio(
        "Input Method",
        ["⌨️ Text Command", "🎤 Voice Recording"],
        horizontal=True,
    )

    command_text = ""

    if input_method == "⌨️ Text Command":
        command_text = st.text_input(
            "Type your navigation command:",
            placeholder="e.g. Navigate to the charging station",
            key="text_command",
        )
        process_btn = st.button("🚀 Process Command", type="primary", use_container_width=True)

    else:
        st.info("🎤 Voice recording requires `sounddevice` and `openai-whisper` to be installed.")
        duration = st.slider("Recording Duration (seconds)", 2, 10, 5)

        col_a, col_b = st.columns(2)
        with col_a:
            record_btn = st.button("🔴 Record Voice", use_container_width=True, type="primary")
        with col_b:
            process_btn = st.button("🚀 Process", use_container_width=True)

        if record_btn:
            with st.spinner(f"🎙️ Recording for {duration}s..."):
                try:
                    from speech.speech_to_text import SpeechToText
                    stt = SpeechToText()
                    if stt.load():
                        text = stt.record_and_transcribe(duration=duration)
                        st.session_state["voice_text"] = text
                        st.success(f"📝 Transcribed: **{text}**")
                    else:
                        st.error("Whisper model could not be loaded.")
                except Exception as e:
                    st.error(f"Recording error: {e}")

        command_text = st.session_state.get("voice_text", "")
        if command_text:
            st.write(f"**Transcribed text:** {command_text}")

    st.divider()

    # Sample commands
    st.markdown("### 💡 Sample Commands")
    sample_commands = [
        "Navigate to the charging station",
        "Move slowly to the office",
        "Stop the robot",
        "Turn left",
        "Find the shortest route",
        "Avoid obstacles and go to the kitchen",
        "Where are you?",
        "Go home",
    ]
    for cmd in sample_commands:
        if st.button(cmd, key=f"sample_{cmd}", use_container_width=True):
            st.session_state["selected_sample"] = cmd
            command_text = cmd

    if "selected_sample" in st.session_state:
        command_text = st.session_state["selected_sample"]

with col2:
    st.markdown("### 🧠 NLP Analysis")

    if command_text and (
        st.session_state.get("last_command") != command_text
        or st.button("🔄 Re-analyze", key="reanalyze")
    ):
        st.session_state["last_command"] = command_text

        with st.spinner("🔍 Analyzing command..."):
            time.sleep(0.3)  # Simulate processing

            # NLP Pipeline
            intent_result = classifier.classify(command_text)
            slots = extractor.extract(command_text)
            task = planner.plan_from_text(command_text)

        # Display results
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        method = intent_result["method"]

        # Intent
        intent_colors = {
            "NAVIGATE": "🟢", "STOP": "🔴", "SET_CONSTRAINT": "🟡",
            "OPTIMIZE": "🔵", "DIRECTION": "🟣", "QUERY": "⚪", "UNKNOWN": "⚫",
        }
        icon = intent_colors.get(intent, "⚫")
        st.success(f"{icon} **Intent Detected:** `{intent}`")
        st.progress(confidence, text=f"Confidence: {confidence:.1%} ({method})")

        st.markdown("**📦 Extracted Slots:**")
        slots_display = {
            "🎯 Destination": slots.get("destination") or "—",
            "⚡ Speed": slots.get("speed") or "normal",
            "🛡️ Constraint": slots.get("constraint") or "—",
            "🧭 Direction": slots.get("direction") or "—",
        }
        for key, val in slots_display.items():
            if val and val != "—":
                st.write(f"**{key}:** {val}")

        # Navigation task
        st.markdown("**📋 Navigation Task:**")
        task_desc = planner.describe_task(task)
        st.info(f"📍 {task_desc}")

        if task.has_goal:
            st.write(f"**Goal coordinates:** ({task.goal_x:.0f}, {task.goal_y:.0f})")
        if task.speed_factor != 1.0:
            st.write(f"**Speed factor:** {task.speed_factor}×")

        # TTS response
        st.divider()
        st.markdown("**🔊 System Response:**")
        response_text = task_desc
        st.write(f"*\"{response_text}\"*")

        speak_btn = st.button("🔊 Speak Response", use_container_width=True)
        if speak_btn:
            try:
                tts = load_tts()
                tts.speak(response_text)
                st.success("Speaking...")
            except Exception as e:
                st.warning(f"TTS unavailable: {e}")

        # Store task for navigation page
        st.session_state["current_task"] = {
            "intent": intent,
            "task": task,
            "command": command_text,
        }

st.divider()

# ─── Command History ─────────────────────────────────────────────────────────
if "command_history" not in st.session_state:
    st.session_state["command_history"] = []

if command_text and command_text not in [h["command"] for h in st.session_state["command_history"]]:
    if "last_command" in st.session_state:
        st.session_state["command_history"].insert(0, {
            "command": command_text,
            "time": time.strftime("%H:%M:%S"),
        })
        st.session_state["command_history"] = st.session_state["command_history"][:10]

if st.session_state["command_history"]:
    st.markdown("### 📜 Command History (Last 10)")
    for h in st.session_state["command_history"]:
        st.markdown(f"`{h['time']}` — {h['command']}")

# ─── NLP Model Info ──────────────────────────────────────────────────────────
with st.expander("🔧 NLP Pipeline Details"):
    st.markdown("""
    **Intent Classifier:**
    - Method: Rule-based regex patterns (fast, no model needed)
    - Upgrade path: Fine-tune DistilBERT on `nlp/commands.csv` for higher accuracy
    - Train command: `python -c "from nlp.intent_classifier import IntentClassifier; IntentClassifier().train()"`

    **Slot Extractor:**
    - Method: Rule-based keyword matching
    - Extracts: destination, speed, constraint, direction, coordinates

    **Text-to-Speech:**
    - Engine: pyttsx3 (offline, Windows SAPI5)
    - Rate: 145 wpm, Volume: 90%

    **Speech-to-Text:**
    - Model: OpenAI Whisper Tiny
    - Install: `pip install openai-whisper sounddevice soundfile`
    """)

