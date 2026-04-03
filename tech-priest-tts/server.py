from io import BytesIO
from pathlib import Path
import inspect
import traceback
import wave
from collections.abc import Iterable

import numpy as np
from scipy.io import wavfile

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from piper import PiperVoice

app = FastAPI(title="Tech Priest TTS (Offline Piper)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VOICE_MODEL_PATH = Path("en_GB-alan-medium.onnx")

if not VOICE_MODEL_PATH.exists():
    raise RuntimeError(
        f"Piper voice model not found: {VOICE_MODEL_PATH}. "
        "Download it first with the correct Piper voice model."
    )

try:
    voice = PiperVoice.load(str(VOICE_MODEL_PATH))
except Exception as exc:
    raise RuntimeError(f"Failed to load Piper voice model: {exc}") from exc


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


def get_piper_sample_rate() -> int:
    config = getattr(voice, "config", None)
    sample_rate = getattr(config, "sample_rate", None)

    if isinstance(sample_rate, int) and sample_rate > 0:
        return sample_rate

    # Safe fallback used by common Piper voices
    return 22050


def ensure_wav_bytes(value: bytes) -> bytes:
    if not value:
        raise RuntimeError("Piper returned empty audio bytes.")

    try:
        with wave.open(BytesIO(value), "rb") as wav_reader:
            wav_reader.getnchannels()
            wav_reader.getframerate()
            wav_reader.getnframes()
    except Exception as exc:
        raise RuntimeError(f"Piper did not return valid WAV data: {exc}") from exc

    return value


def exhaust_if_iterable(result: object) -> None:
    if isinstance(result, Iterable) and not isinstance(result, (bytes, bytearray, str)):
        for _ in result:
            pass

def run_piper_synthesize(text: str, wav_file: wave.Wave_write) -> None:
    """
    Piper Python bindings in this install return iterable audio chunks
    from voice.synthesize(text, ...), rather than writing directly to
    a wav_file argument.
    """
    errors: list[str] = []

    attempts: list[tuple[str, callable]] = [
        ("voice.synthesize(text)", lambda: voice.synthesize(text)),
        (
            "voice.synthesize(text=text)",
            lambda: voice.synthesize(text=text),
        ),
        (
            "voice.synthesize(text, None, False)",
            lambda: voice.synthesize(text, None, False),
        ),
        (
            "voice.synthesize(text=text, syn_config=None, include_alignments=False)",
            lambda: voice.synthesize(
                text=text,
                syn_config=None,
                include_alignments=False,
            ),
        ),
    ]

    for label, fn in attempts:
        try:
            result = fn()

            wrote_audio = False

            for chunk in result:
                audio_int16_bytes = getattr(chunk, "audio_int16_bytes", None)
                if audio_int16_bytes:
                    wav_file.writeframes(audio_int16_bytes)
                    wrote_audio = True
                    continue

                audio_bytes = getattr(chunk, "audio_bytes", None)
                if audio_bytes:
                    wav_file.writeframes(audio_bytes)
                    wrote_audio = True
                    continue

                audio = getattr(chunk, "audio", None)
                if audio is not None:
                    if isinstance(audio, bytes):
                        wav_file.writeframes(audio)
                        wrote_audio = True
                        continue

                    if hasattr(audio, "tobytes"):
                        wav_file.writeframes(audio.tobytes())
                        wrote_audio = True
                        continue

            if wrote_audio:
                return

            errors.append(f"{label} -> synthesize returned no writable audio chunks")
        except TypeError as exc:
            errors.append(f"{label} -> {exc}")
            continue
        except AttributeError as exc:
            errors.append(f"{label} -> {exc}")
            continue
        except Exception as exc:
            raise RuntimeError(f"{label} failed: {exc}") from exc

    signature_text = "unavailable"
    try:
        signature_text = str(inspect.signature(voice.synthesize))
    except Exception:
        pass

    raise RuntimeError(
        "No compatible Piper synthesize() call signature worked. "
        f"Detected signature: {signature_text}. "
        f"Attempts: {' | '.join(errors)}"
    )

def synthesize_piper_wav_bytes(text: str) -> bytes:
    output_buffer = BytesIO()
    sample_rate = get_piper_sample_rate()

    try:
        with wave.open(output_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            run_piper_synthesize(text, wav_file)

    except Exception as exc:
        raise RuntimeError(f"Piper synth call failed: {exc}") from exc

    wav_bytes = output_buffer.getvalue()
    return ensure_wav_bytes(wav_bytes)


def apply_tech_priest_effect(wav_bytes: bytes) -> bytes:
    input_buffer = BytesIO(wav_bytes)

    try:
        sample_rate, audio_data = wavfile.read(input_buffer)
    except Exception as exc:
        raise RuntimeError(f"Could not read Piper WAV output: {exc}") from exc

    if getattr(audio_data, "size", 0) == 0:
        raise RuntimeError("Decoded WAV audio was empty.")

    if audio_data.dtype == np.int16:
        x = audio_data.astype(np.float32) / 32768.0
    elif np.issubdtype(audio_data.dtype, np.integer):
        max_int = np.iinfo(audio_data.dtype).max
        x = audio_data.astype(np.float32) / float(max_int)
    else:
        x = audio_data.astype(np.float32)

    if x.ndim > 1:
        x = x.mean(axis=1)

    hp = x - np.concatenate(([0.0], x[:-1] * 0.97))
    x = (x * 0.9) + (hp * 0.35)

    presence = x - np.concatenate(([0.0], x[:-1] * 0.985))
    x = (x * 0.85) + (presence * 0.4)

    delay_ms = 6
    delay_samples = int(sample_rate * delay_ms / 1000)
    dbl = np.zeros_like(x)
    if 0 < delay_samples < len(x):
        dbl[delay_samples:] = x[:-delay_samples] * 0.08
    x = x + dbl

    slap_ms = 12
    slap_samples = int(sample_rate * slap_ms / 1000)
    slap = np.zeros_like(x)
    if 0 < slap_samples < len(x):
        slap[slap_samples:] = x[:-slap_samples] * 0.06
    x = x + slap

    x = np.tanh(x * 1.15)

    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak * 0.95

    x = np.clip(x, -1.0, 1.0)
    out = (x * 32767).astype(np.int16)

    output_buffer = BytesIO()
    wavfile.write(output_buffer, sample_rate, out)
    processed_bytes = output_buffer.getvalue()

    if not processed_bytes:
        raise RuntimeError("Processed WAV output was empty.")

    return processed_bytes


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "piper",
        "voice": VOICE_MODEL_PATH.name,
    }


@app.post("/tts")
async def tts(payload: TtsRequest):
    text = payload.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        raw_wav_bytes = synthesize_piper_wav_bytes(text)
        processed_wav_bytes = apply_tech_priest_effect(raw_wav_bytes)

        return Response(
            content=processed_wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="speech.wav"',
                "Content-Length": str(len(processed_wav_bytes)),
                "Cache-Control": "no-store",
            },
        )
    except Exception as exc:
        print("\n=== TTS BACKEND ERROR START ===")
        traceback.print_exc()
        print("=== TTS BACKEND ERROR END ===\n")

        raise HTTPException(
            status_code=500,
            detail=f"Piper synthesis failed: {exc}",
        ) from exc