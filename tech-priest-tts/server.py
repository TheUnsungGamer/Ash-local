from io import BytesIO
from pathlib import Path
import os
import subprocess
import tempfile
import traceback
import wave

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from piper import PiperVoice
from pydantic import BaseModel, Field
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range


# =========================
# APP INIT
# =========================
app = FastAPI(title="Tech Priest TTS (Piper -> RVC CLI -> Verity FX)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PIPER CONFIG
# =========================
VOICE_MODEL_PATH = Path("en_GB-jenny_dioco-medium.onnx")

if not VOICE_MODEL_PATH.exists():
    raise RuntimeError(f"Piper model not found: {VOICE_MODEL_PATH}")

voice = PiperVoice.load(str(VOICE_MODEL_PATH))


# =========================
# RVC CONFIG
# =========================
RVC_DIR = r"C:\Users\richa\Desktop\RVC-beta0717"
RVC_PYTHON = os.path.join(RVC_DIR, "runtime", "python.exe")
RVC_MODEL = os.path.join(RVC_DIR, "assets", "weights", "verity.pth")

# Leave blank if you do not have an index file
RVC_INDEX = ""

# RVC tuning
RVC_PITCH = 0
RVC_F0_METHOD = "rmvpe"
RVC_INDEX_RATE = 0.75
RVC_FILTER_RADIUS = 3
RVC_RESAMPLE_SR = 0
RVC_RMS_MIX_RATE = 0.25
RVC_PROTECT = 0.33
RVC_DEVICE = "cuda:0"
RVC_IS_HALF = "True"


# =========================
# REQUEST MODEL
# =========================
class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


# =========================
# HELPERS
# =========================
def find_rvc_infer_cli(rvc_dir: str) -> str:
    candidates = [
        os.path.join(rvc_dir, "infer_cli.py"),
        os.path.join(rvc_dir, "tools", "infer_cli.py"),
        os.path.join(rvc_dir, "infer", "infer_cli.py"),
        os.path.join(rvc_dir, "tools", "infer", "infer_cli.py"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    for root, _, files in os.walk(rvc_dir):
        if "infer_cli.py" in files:
            return os.path.join(root, "infer_cli.py")

    raise RuntimeError(
        "infer_cli.py not found anywhere under RVC_DIR. "
        f"Checked under: {rvc_dir}"
    )


RVC_INFER_CLI = find_rvc_infer_cli(RVC_DIR)


# =========================
# PIPER SYNTHESIS
# =========================
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
# RVC CONVERSION (SUBPROCESS)
# =========================
def apply_rvc_conversion(input_wav_bytes: bytes) -> bytes:
    if not input_wav_bytes:
        raise RuntimeError("Empty WAV passed to RVC.")

    if not os.path.exists(RVC_DIR):
        raise RuntimeError(f"RVC_DIR not found: {RVC_DIR}")

    if not os.path.exists(RVC_PYTHON):
        raise RuntimeError(f"RVC runtime python not found: {RVC_PYTHON}")

    if not os.path.exists(RVC_INFER_CLI):
        raise RuntimeError(f"infer_cli.py not found: {RVC_INFER_CLI}")

    if not os.path.exists(RVC_MODEL):
        raise RuntimeError(f"RVC model not found: {RVC_MODEL}")

    if RVC_INDEX and not os.path.exists(RVC_INDEX):
        raise RuntimeError(f"RVC index not found: {RVC_INDEX}")

    input_path = None
    output_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
            tmp_in.write(input_wav_bytes)
            input_path = tmp_in.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            output_path = tmp_out.name

        cmd = [
            RVC_PYTHON,
            RVC_INFER_CLI,
            "--input_path", input_path,
            "--opt_path", output_path,
            "--model_name", RVC_MODEL,
            "--f0up_key", str(RVC_PITCH),
            "--f0method", RVC_F0_METHOD,
            "--index_rate", str(RVC_INDEX_RATE),
            "--filter_radius", str(RVC_FILTER_RADIUS),
            "--resample_sr", str(RVC_RESAMPLE_SR),
            "--rms_mix_rate", str(RVC_RMS_MIX_RATE),
            "--protect", str(RVC_PROTECT),
            "--device", RVC_DEVICE,
            "--is_half", RVC_IS_HALF,
        ]

        if RVC_INDEX:
            cmd.extend(["--index_path", RVC_INDEX])

        result = subprocess.run(
            cmd,
            cwd=RVC_DIR,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "RVC subprocess failed.\n"
                f"Command: {' '.join(cmd)}\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )

        if not output_path or not os.path.exists(output_path):
            raise RuntimeError(
                "RVC completed but output file was not created.\n"
                f"Command: {' '.join(cmd)}\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )

        with open(output_path, "rb") as f:
            converted = f.read()

        if not converted:
            raise RuntimeError("RVC output WAV was empty.")

        return converted

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("RVC subprocess timed out.") from exc
    except Exception as exc:
        raise RuntimeError(f"RVC conversion failed: {exc}") from exc
    finally:
        for path in (input_path, output_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


# =========================
# VERITY FX
# =========================
def apply_verity_effect(wav_bytes: bytes) -> bytes:
    try:
        audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
    except Exception as exc:
        raise RuntimeError(f"Could not load WAV: {exc}") from exc

    audio = audio.set_channels(1)
    audio = audio.high_pass_filter(150)

    audio = compress_dynamic_range(
        audio,
        threshold=-20.0,
        ratio=4.0,
        attack=5.0,
        release=50.0,
    )

    presence = audio.high_pass_filter(2500).low_pass_filter(4500) + 5
    audio = audio.overlay(presence)

    reflection = audio - 22
    audio = audio.overlay(reflection, position=12)

    original_rate = audio.frame_rate
    audio = audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(original_rate * 1.012)},
    ).set_frame_rate(original_rate)

    audio = audio.normalize(headroom=1.0)

    out = BytesIO()
    audio.export(out, format="wav")
    return out.getvalue()


# =========================
# ROUTES
# =========================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline": "Piper -> RVC CLI -> Verity FX",
        "voice": VOICE_MODEL_PATH.name,
        "rvc_dir": RVC_DIR,
        "rvc_python": RVC_PYTHON,
        "rvc_infer_cli": RVC_INFER_CLI,
        "rvc_model": RVC_MODEL,
        "rvc_index": RVC_INDEX,
    }


@app.post("/tts")
async def tts(payload: TtsRequest):
    try:
        piper_audio = synthesize_piper_wav_bytes(payload.text)
        rvc_audio = apply_rvc_conversion(piper_audio)
        final_audio = apply_verity_effect(rvc_audio)

        return Response(
            content=final_audio,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="verity.wav"',
                "Cache-Control": "no-store",
            },
        )

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))