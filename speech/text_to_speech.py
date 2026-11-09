"""
NavMind AI - Text to Speech
Converts navigation responses to speech using pyttsx3.
"""
from __future__ import annotations

import threading


class TextToSpeech:
    """
    Text-to-speech module using pyttsx3 (offline, no API key needed).
    
    Provides natural spoken feedback for navigation events.
    """

    # Pre-defined response templates
    RESPONSES = {
        "navigation_start": "Navigation started. Heading to {destination}.",
        "goal_reached": "Destination reached successfully. Navigation complete.",
        "collision": "Warning! Collision detected. Robot has stopped.",
        "timeout": "Navigation timeout. Could not reach destination in time.",
        "stop": "Robot stopped.",
        "command_received": "Command received. {description}",
        "obstacle_detected": "Obstacle detected ahead. Adjusting course.",
        "low_battery": "Low battery warning. Returning to charging station.",
        "ready": "NavMind AI is ready. Awaiting navigation commands.",
        "error": "An error occurred. Please try again.",
    }

    def __init__(
        self,
        rate: int = 150,
        volume: float = 0.9,
        voice_gender: str = "female",
    ):
        """
        Args:
            rate: Speech rate (words per minute)
            volume: Volume level [0.0, 1.0]
            voice_gender: Preferred voice gender ('male' or 'female')
        """
        self.rate = rate
        self.volume = volume
        self.voice_gender = voice_gender
        self.engine = None
        self._loaded = False
        self._speaking = False

    def initialize(self) -> bool:
        """Initialize pyttsx3 engine. Returns True if successful."""
        try:
            import pyttsx3  # type: ignore
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", self.rate)
            self.engine.setProperty("volume", self.volume)

            # Select voice
            voices = self.engine.getProperty("voices")
            if voices:
                # Try to find preferred gender voice
                selected = voices[0]  # default to first
                for voice in voices:
                    name = voice.name.lower()
                    if self.voice_gender == "female" and "female" in name:
                        selected = voice
                        break
                    elif self.voice_gender == "male" and "male" in name:
                        selected = voice
                        break
                self.engine.setProperty("voice", selected.id)

            self._loaded = True
            return True
        except ImportError:
            print("[TTS] pyttsx3 not installed. Run: pip install pyttsx3")
            return False
        except Exception as e:
            print(f"[TTS] Initialization failed: {e}")
            return False

    def speak(self, text: str, blocking: bool = False):
        """
        Convert text to speech.

        Args:
            text: Text to speak
            blocking: If True, wait for speech to finish before returning
        """
        if not self._loaded:
            self.initialize()

        if not self._loaded:
            print(f"[TTS Fallback] {text}")
            return

        if blocking:
            self._speak_now(text)
        else:
            thread = threading.Thread(target=self._speak_now, args=(text,), daemon=True)
            thread.start()

    def _speak_now(self, text: str):
        """Internal: speak synchronously."""
        try:
            self._speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"[TTS] Speech error: {e}")
        finally:
            self._speaking = False

    def announce(self, event: str, **kwargs) -> str:
        """
        Speak a predefined navigation event response.

        Args:
            event: Event key from RESPONSES dict
            **kwargs: Template substitution variables

        Returns:
            The spoken text
        """
        template = self.RESPONSES.get(event, "Event: {event}")
        try:
            text = template.format(event=event, **kwargs)
        except KeyError:
            text = template
        self.speak(text)
        return text

    def set_rate(self, rate: int):
        """Update speech rate."""
        self.rate = rate
        if self._loaded and self.engine:
            self.engine.setProperty("rate", rate)

    def set_volume(self, volume: float):
        """Update volume level [0.0, 1.0]."""
        self.volume = max(0.0, min(1.0, volume))
        if self._loaded and self.engine:
            self.engine.setProperty("volume", self.volume)

    def stop(self):
        """Stop current speech."""
        if self._loaded and self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass

    def is_speaking(self) -> bool:
        return self._speaking

    def is_ready(self) -> bool:
        return self._loaded

    def get_available_voices(self) -> list[str]:
        """Return list of available voice names."""
        if not self._loaded:
            self.initialize()
        if self._loaded and self.engine:
            voices = self.engine.getProperty("voices")
            return [v.name for v in voices] if voices else []
        return []

