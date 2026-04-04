from io import BytesIO
from pathlib import Path
import traceback
import wave
import requests
import tempfile
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from piper import PiperVoice
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range


app = FastAPI(title="Tech Priest TTS (Piper + RVC)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VOICE_MODEL_PATH = Path("en_GB-jenny_dioco-medium.onnx")

if not VOICE_MODEL_PATH.exists():
    raise RuntimeError(f"Piper model not found: {VOICE_MODEL_PATH}")

voice = PiperVoice.load(str(VOICE_MODEL_PATH))


# =========================
# 🔊 RVC CONFIG
# =========================
RVC_URL = "http://127.0.0.1:7897"

RVC_MODEL_PATH = r"C:\Users\richa\Desktop\RVC-beta0717\assets\weights\verity.pth"
RVC_INDEX_PATH = r""


def apply_rvc_conversion(input_wav_bytes: bytes) -> bytes:
    if not input_wav_bytes:
        raise RuntimeError("Empty WAV passed to RVC")

    if not os.path.exists(RVC_MODEL_PATH):
        raise RuntimeError(f"RVC model not found: {RVC_MODEL_PATH}")

    if RVC_INDEX_PATH and not os.path.exists(RVC_INDEX_PATH):
        raise RuntimeError(f"RVC index not found: {RVC_INDEX_PATH}")

    input_tmp_path = None
    output_tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
            tmp_in.write(input_wav_bytes)
            input_tmp_path = tmp_in.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            output_tmp_path = tmp_out.name

        payload = {
            "fn_index": 0,
            "data": [
                0,                  # speaker/sid
                input_tmp_path,     # input audio path
                0,                  # f0_up_key
                None,               # auto f0 file path / unused
                "rmvpe",            # f0 method
                RVC_INDEX_PATH,     # feature index path textbox
                RVC_INDEX_PATH,     # auto-detected index dropdown/value
                0.75,               # index_rate
                3,                  # filter_radius
                0,                  # resample_sr
                0.25,               # rms_mix_rate
                0.33,               # protect
            ],
        }

        resp = requests.post(
            f"{RVC_URL}/api/predict",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()

        result = resp.json()

        def walk(obj):
            if isinstance(obj, str):
                yield obj
            elif isinstance(obj, dict):
                for v in obj.values():
                    yield from walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    yield from walk(v)

        candidate_paths = []
        for item in walk(result):
            if isinstance(item, str) and item.lower().endswith(".wav"):
                candidate_paths.append(item)

        for path in candidate_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return f.read()

        raise RuntimeError(f"RVC returned no readable WAV path. Response: {result}")

    except requests.RequestException as exc:
        raise RuntimeError(f"RVC request failed: {exc}") from exc

    finally:
        for tmp_path in (input_tmp_path, output_tmp_path):
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass


# =========================
# 🔊 VERITY FX
# =========================
def apply_verity_effect(wav_bytes: bytes) -> bytes:
    try:
        audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
    except Exception as exc:
        raise RuntimeError(f"Could not load WAV into pydub: {exc}") from exc

    audio = audio.set_channels(1)
    audio = audio.high_pass_filter(150)

    audio = compress_dynamic_range(
        audio,
        threshold=-20.0,
        ratio=4.0,
        attack=5.0,
        release=50.0
    )

    presence = audio.high_pass_filter(2500).low_pass_filter(4500) + 5
    audio = audio.overlay(presence)

    reflection = audio - 22
    audio = audio.overlay(reflection, position=12)

    original_rate = audio.frame_rate
    audio = audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(original_rate * 1.012)}
    )
    audio = audio.set_frame_rate(original_rate)

    audio = audio.normalize(headroom=1.0)

    out = BytesIO()
    audio.export(out, format="wav")
    return out.getvalue()


# =========================
# 🔊 PIPER
# =========================
class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


def get_piper_sample_rate() -> int:
    config = getattr(voice, "config", None)
    sample_rate = getattr(config, "sample_rate", None)
    if isinstance(sample_rate, int) and sample_rate > 0:
        return sample_rate
    return 22050


def run_piper_synthesize(text: str, wav_file) -> None:
    result = voice.synthesize(text)

    wrote_audio = False

    for chunk in result:
        audio_bytes = (
            getattr(chunk, "audio_int16_bytes", None)
            or getattr(chunk, "audio_bytes", None)
        )

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

    if not wrote_audio:
        raise RuntimeError("Piper synth returned no writable audio chunks.")


def synthesize_piper_wav_bytes(text: str) -> bytes:
    buffer = BytesIO()

    try:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(get_piper_sample_rate())
            run_piper_synthesize(text, wav_file)
    except Exception as exc:
        raise RuntimeError(f"Piper synthesis failed: {exc}") from exc

    wav_bytes = buffer.getvalue()
    if not wav_bytes:
        raise RuntimeError("Piper returned empty WAV bytes.")

    return wav_bytes


# =========================
# 🚀 ROUTES
# =========================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "piper",
        "voice": VOICE_MODEL_PATH.name,
    }


@app.post("/tts")
async def tts(payload: TtsRequest):
    try:
        raw = synthesize_piper_wav_bytes(payload.text)
        rvc_audio = apply_rvc_conversion(raw)
        final_audio = apply_verity_effect(rvc_audio)

        return Response(
            content=final_audio,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="speech.wav"',
                "Cache-Control": "no-store",
            },
        )

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))