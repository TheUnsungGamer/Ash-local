from io import BytesIO
from pathlib import Path
import inspect
import traceback
import wave
from collections.abc import Iterable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from piper import PiperVoice
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range

from pydub.effects import compress_dynamic_range, high_pass_filter

def process_voice(audio: AudioSegment) -> AudioSegment:
    # 1. High-pass filter (remove low-end mud)
    audio = high_pass_filter(audio, cutoff=100)

    # 2. Compression
    audio = compress_dynamic_range(
        audio,
        threshold=-20.0,
        ratio=4.0,
        attack=5,
        release=50
    )

    # 3. Presence boost
    boosted = audio.high_pass_filter(2500).apply_gain(3)
    audio = audio.overlay(boosted)

    # 4. Light reverb
    delay = audio - 12
    delay = delay.fade_in(10).fade_out(100)
    audio = audio.overlay(delay, position=15)

    return audio

app = FastAPI(title="Tech Priest TTS (Offline Piper)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Recommended for a Verity-ish direction:
# VOICE_MODEL_PATH = Path("en_GB-southern_english_female-medium.onnx")
# VOICE_MODEL_PATH = Path("en_GB-southern_english_female-low.onnx")
VOICE_MODEL_PATH = Path("en_GB-jenny_dioco-medium.onnx")

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
        ("voice.synthesize(text=text)", lambda: voice.synthesize(text=text)),
        ("voice.synthesize(text, None, False)", lambda: voice.synthesize(text, None, False)),
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
    """
    Verity / ship-AI style processing:
    - subtle compression for a more constant, authoritative level
    - 100 Hz high-pass to remove mud
    - presence lift in the 2.5kHz–4kHz region
    - very short light cockpit-style reflections
    - gentle normalization
    """
    try:
        audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
    except Exception as exc:
        raise RuntimeError(f"Could not load WAV into pydub: {exc}") from exc

    # Keep the voice focused and centered
    audio = audio.set_channels(1)

    # 1) Subtle compression
    audio = compress_dynamic_range(
        audio,
        threshold=-20.0,
        ratio=2.5,
        attack=5.0,
        release=80.0,
    )

    # 2) High-pass filter around 100 Hz
    audio = audio.high_pass_filter(100)

    # 3) Presence boost approximation for 2.5kHz–4kHz
    # Pydub doesn't do true parametric EQ, so isolate this region and blend it back louder.
    presence_band = audio.high_pass_filter(2500).low_pass_filter(4000) + 3
    audio = audio.overlay(presence_band)

    # Extra light upper sheen for digital clarity
    air_band = audio.high_pass_filter(4500) - 7
    audio = audio.overlay(air_band)

    # 4) Light "cockpit" reverb using a few short reflections
    def delayed(seg: AudioSegment, delay_ms: int, gain_db: float) -> AudioSegment:
        return AudioSegment.silent(duration=delay_ms, frame_rate=seg.frame_rate) + (seg + gain_db)

    reflections = AudioSegment.silent(duration=len(audio), frame_rate=audio.frame_rate)
    reflections = reflections.overlay(delayed(audio, 15, -20))
    reflections = reflections.overlay(delayed(audio, 28, -23))
    reflections = reflections.overlay(delayed(audio, 42, -27))

    # Keep it subtle; this should feel like space around the voice, not audible echo
    audio = audio.overlay(reflections - 2)

    # 5) Gentle normalization with headroom
    audio = audio.normalize(headroom=1.0)

    out_buffer = BytesIO()
    try:
        audio.export(out_buffer, format="wav")
    except Exception as exc:
        raise RuntimeError(f"Could not export processed WAV: {exc}") from exc

    processed_bytes = out_buffer.getvalue()

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