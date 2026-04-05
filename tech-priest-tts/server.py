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

# Start with "cpu" to rule out CUDA issues. Switch to "cuda:0" once working.
RVC_DEVICE = "cpu"
RVC_IS_HALF = "False"  # Must be False when using CPU


# =========================
# REQUEST MODEL
# =========================
class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


# =========================
# HELPERS
# =========================

RVC_INFER_CLI = r"C:\Users\richa\Desktop\RVC-beta0717\infer_cli.py"



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
    """
    Run RVC voice conversion via infer_cli.py subprocess.

    FIX 1 — Positional args, not --named-flags:
        RVC-beta0717's infer_cli.py reads sys.argv positionally.
        Using --named-flags causes silent failure (rc=0, no output written),
        which is the root cause of the NoneType / empty output errors.

        Positional order:
            f0up_key  f0method  input_path  opt_path  model_name
            index_path  index_rate  filter_radius  resample_sr
            rms_mix_rate  protect  device  is_half

    FIX 2 — Don't pre-create the output temp file:
        Some RVC builds won't overwrite an existing file, even an empty one.
        We get the path from mkstemp, close + delete the file, then let RVC create it fresh.
    """
    if not input_wav_bytes:
        raise RuntimeError("Empty WAV passed to RVC.")

    for label, path in [
        ("RVC_DIR",       RVC_DIR),
        ("RVC_PYTHON",    RVC_PYTHON),
        ("RVC_INFER_CLI", RVC_INFER_CLI),
        ("RVC_MODEL",     RVC_MODEL),
    ]:
        if not os.path.exists(path):
            raise RuntimeError(f"{label} not found: {path}")

    if RVC_INDEX and not os.path.exists(RVC_INDEX):
        raise RuntimeError(f"RVC index not found: {RVC_INDEX}")

    input_path = None
    output_path = None

    try:
        # Write Piper output to temp input file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
            tmp_in.write(input_wav_bytes)
            input_path = tmp_in.name

        # Reserve a temp path for RVC output, then DELETE the empty file
        # so RVC can create it fresh (some builds won't overwrite existing files)
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        os.remove(output_path)

        index_path_arg = RVC_INDEX if RVC_INDEX else ""

        # Positional argument order for RVC-beta0717
        cmd = [
    RVC_PYTHON,
    RVC_INFER_CLI,
    "--input_path", input_path,
    "--opt_path", output_path,
    "--model_name", RVC_MODEL,
    "--f0up_key", str(RVC_PITCH),
    "--f0method", RVC_F0_METHOD,
    "--index_path", index_path_arg,
    "--index_rate", str(RVC_INDEX_RATE),
    "--filter_radius", str(RVC_FILTER_RADIUS),
    "--resample_sr", str(RVC_RESAMPLE_SR),
    "--rms_mix_rate", str(RVC_RMS_MIX_RATE),
    "--protect", str(RVC_PROTECT),
    "--device", RVC_DEVICE,
    "--is_half", RVC_IS_HALF,
]

        print(f"[RVC] CMD: {' '.join(cmd)}", flush=True)

        result = subprocess.run(
            cmd,
            cwd=RVC_DIR,
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Always surface RVC output so you can see what's happening
        if result.stdout:
            print(f"[RVC STDOUT]\n{result.stdout}", flush=True)
        if result.stderr:
            print(f"[RVC STDERR]\n{result.stderr}", flush=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"RVC subprocess exited with code {result.returncode}.\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(
                "RVC exited cleanly (rc=0) but produced no output file.\n"
                "This usually means the positional argument order doesn't match your build.\n"
                f"Open {RVC_INFER_CLI} and check how it reads sys.argv, then adjust the cmd list.\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        with open(output_path, "rb") as f:
            converted = f.read()

        if not converted:
            raise RuntimeError("RVC output WAV was empty after reading.")

        return converted

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("RVC subprocess timed out (300s). Consider switching to GPU.") from exc
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"RVC conversion failed unexpectedly: {exc}") from exc
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
        "rvc_index": RVC_INDEX or "(none)",
        "rvc_device": RVC_DEVICE,
        "rvc_is_half": RVC_IS_HALF,
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
