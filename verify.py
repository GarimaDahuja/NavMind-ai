"""
NavMind AI - Quick GPU and Environment Verification Script
Run this after installation to verify all components are working.
"""

import sys


def check_component(name: str, check_fn) -> bool:
    try:
        result = check_fn()
        print(f"  ✅ {name}: {result}")
        return True
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("  NavMind AI — System Verification")
    print("=" * 60)

    checks = []

    print("\n📦 Core Dependencies:")
    checks.append(check_component("Python", lambda: sys.version.split()[0]))
    checks.append(check_component("NumPy", lambda: __import__("numpy").__version__))
    checks.append(check_component("Gymnasium", lambda: __import__("gymnasium").__version__))
    checks.append(check_component("Pygame", lambda: __import__("pygame").ver))
    checks.append(check_component("Matplotlib", lambda: __import__("matplotlib").__version__))

    print("\n🧠 Deep Learning:")
    checks.append(check_component("PyTorch", lambda: __import__("torch").__version__))
    checks.append(check_component(
        "CUDA Available",
        lambda: "Yes — " + __import__("torch").cuda.get_device_name(0)
        if __import__("torch").cuda.is_available() else "No (CPU mode)"
    ))
    checks.append(check_component("Stable-Baselines3", lambda: __import__("stable_baselines3").__version__))
    checks.append(check_component("Transformers", lambda: __import__("transformers").__version__))

    print("\n🌐 Web App:")
    checks.append(check_component("Streamlit", lambda: __import__("streamlit").__version__))
    checks.append(check_component("Plotly", lambda: __import__("plotly").__version__))

    print("\n🎤 Speech (Optional):")
    check_component("Whisper", lambda: "installed" if __import__("whisper") else "")
    check_component("pyttsx3", lambda: "installed" if __import__("pyttsx3") else "")
    check_component("sounddevice", lambda: "installed" if __import__("sounddevice") else "")

    print("\n🤖 NavMind Modules:")
    checks.append(check_component(
        "RobotNavEnv",
        lambda: "OK — " + str(__import__("environment.robot_env", fromlist=["RobotNavEnv"]).RobotNavEnv())
    ))
    checks.append(check_component(
        "DQNAgent",
        lambda: "OK — " + str(__import__("reinforcement_learning.dqn", fromlist=["DQNAgent"]).DQNAgent())
    ))
    checks.append(check_component(
        "AStarPlanner",
        lambda: "OK — " + str(__import__("navigation.astar", fromlist=["AStarPlanner"]).AStarPlanner())
    ))
    checks.append(check_component(
        "IntentClassifier",
        lambda: "OK — " + str(__import__("nlp.intent_classifier", fromlist=["IntentClassifier"]).IntentClassifier(use_bert=False))
    ))

    total = sum(checks)
    print(f"\n{'='*60}")
    print(f"  {total}/{len(checks)} checks passed")
    if total == len(checks):
        print("  🎉 All systems ready! Run: streamlit run app.py")
    else:
        print("  ⚠️  Some components missing. Run: pip install -r requirements.txt")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

