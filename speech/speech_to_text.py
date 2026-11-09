"""
NavMind AI - Speech to Text
Converts voice commands to text using OpenAI Whisper Tiny.
"""
from __future__ import annotations

import os
import tempfile
import numpy as np
from typing import Optional


class SpeechToText:
    """
    Voice-to-text module using OpenAI Whisper Tiny.

    Whisper Tiny is optimized for local inference on consumer GPUs (RTX 3050).
    Falls back to file-based transcription if microphone is unavailable.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        language: str = "en",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium')
            language: Target language code ('en' for English)
            device: Compute device ('cuda', 'cpu', or None for auto)
        """
        self.model_size = model_size
        self.language = language
        self.device = device
        self.model = None
        self._loaded = False

    def load(self) -> bool:
        """Load Whisper model. Returns True if successful."""
        try:
            import whisper  # type: ignore
            self.model = whisper.load_model(
                self.model_size,
                device=self.device,
            )
            self._loaded = True
            print(f"[STT] Whisper '{self.model_size}' model loaded.")
            return True
        except ImportError:
            print("[STT] Whisper not installed. Run: pip install openai-whisper")
            return False
        except Exception as e:
            print(f"[STT] Failed to load Whisper: {e}")
            return False

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, M4A, etc.)

        Returns:
            Transcribed text string
        """
        if not self._loaded:
            if not self.load():
                return ""
        try:
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                fp16=False,
            )
            return result["text"].strip()
        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            return ""

    def transcribe_numpy(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio from a numpy array.

        Args:
            audio_array: Float32 numpy array of audio samples
            sample_rate: Sample rate in Hz (Whisper expects 16000)

        Returns:
            Transcribed text
        """
        if not self._loaded:
            if not self.load():
                return ""
        try:
            # Save to temp WAV file
            import soundfile as sf  # type: ignore
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_array, sample_rate)
                result = self.transcribe_file(tmp.name)
            os.unlink(tmp.name)
            return result
        except Exception as e:
            print(f"[STT] Numpy transcription error: {e}")
            return ""

    def record_and_transcribe(
        self,
        duration: float = 5.0,
        sample_rate: int = 16000,
    ) -> str:
        """
        Record from microphone and transcribe.

        Args:
            duration: Recording duration in seconds
            sample_rate: Audio sample rate

        Returns:
            Transcribed text
        """
        try:
            import sounddevice as sd  # type: ignore
            print(f"[STT] 🎤 Recording for {duration}s...")
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = audio.flatten()
            print("[STT] ✅ Recording complete. Transcribing...")
            return self.transcribe_numpy(audio, sample_rate)
        except ImportError:
            print("[STT] sounddevice not installed. Run: pip install sounddevice")
            return ""
        except Exception as e:
            print(f"[STT] Recording error: {e}")
            return ""

    def is_ready(self) -> bool:
        """True if model is loaded and ready."""
        return self._loaded

    def get_model_info(self) -> dict:
        """Return model metadata."""
        return {
            "model_size": self.model_size,
            "language": self.language,
            "device": self.device,
            "loaded": self._loaded,
        }

