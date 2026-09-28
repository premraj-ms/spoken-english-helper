import os
import io
import wave
import tempfile
from typing import Optional, Tuple
import numpy as np
import soundfile as sf

# STT: faster-whisper
_whisper_model = None

def get_whisper_model(model_name: str = "base.en"):
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print(f"[STT] Loading lightweight faster-whisper ({model_name})...")
        _whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)
        print("[STT] Faster-whisper ready.")
    return _whisper_model

def transcribe_audio_file(audio_path_or_bytes) -> str:
    """Transcribes an audio file or bytes to text with minimal CPU overhead."""
    model = get_whisper_model()
    
    if isinstance(audio_path_or_bytes, bytes):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_path_or_bytes)
            tmp_path = tmp.name
        try:
            segments, _ = model.transcribe(
                tmp_path,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400)
            )
            text = " ".join([segment.text for segment in segments]).strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        segments, _ = model.transcribe(
            audio_path_or_bytes,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400)
        )
        text = " ".join([segment.text for segment in segments]).strip()

    return text

# TTS: Kokoro ONNX
_kokoro_pipeline = None

def get_kokoro():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        try:
            # Patch np.load to support pickled voice embeddings in numpy >= 2.0
            orig_np_load = np.load
            def patched_np_load(*args, **kwargs):
                if "allow_pickle" not in kwargs:
                    kwargs["allow_pickle"] = True
                return orig_np_load(*args, **kwargs)
            np.load = patched_np_load

            from kokoro_onnx import Kokoro
            model_path = "models/kokoro-v0_19.onnx"
            voices_path = "models/voices-v1.0.bin" if os.path.exists("models/voices-v1.0.bin") else "models/voices.json"
            
            if os.path.exists(model_path) and os.path.exists(voices_path):
                print(f"[TTS] Loading Kokoro-ONNX from {voices_path}...")
                _kokoro_pipeline = Kokoro(model_path, voices_path)
                print("[TTS] Kokoro-ONNX ready.")
            else:
                print("[TTS] Kokoro model files not found locally, will use audio generation fallback.")
                return None
        except Exception as e:
            print(f"[TTS] Warning: Could not initialize Kokoro: {e}")
            return None
    return _kokoro_pipeline

def generate_speech_audio(text: str, voice: str = "af_bella", speed: float = 1.0) -> Optional[bytes]:
    """Generates WAV audio bytes from text using Kokoro TTS."""
    kokoro = get_kokoro()
    if kokoro is not None:
        try:
            # Strip any accidental brackets, quotes, or JSON syntax
            clean_text = text.replace('{', '').replace('}', '').replace('\\"', '"').replace('\\n', ' ').strip()
            if not clean_text or len(clean_text) < 2:
                clean_text = "That sounds great! Tell me more about that."

            samples, sample_rate = kokoro.create(clean_text, voice=voice, speed=speed, lang="en-us")
            buffer = io.BytesIO()
            sf.write(buffer, samples, sample_rate, format="WAV")
            return buffer.getvalue()
        except Exception as e:
            print(f"[TTS] Kokoro generation error: {e}")
            
    return None
