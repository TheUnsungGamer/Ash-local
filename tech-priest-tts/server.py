from pydub import AudioSegment
from pydub.effects import compress_dynamic_range
from io import BytesIO
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


def apply_verity_effect(wav_bytes: bytes) -> bytes:
    from io import BytesIO
    from pydub import AudioSegment
    from pydub.effects import compress_dynamic_range

    audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")

    # 1. Mono, but keep body
    audio = audio.set_channels(1)

    # Less aggressive cleanup
    audio = audio.high_pass_filter(100)
    audio = audio.low_pass_filter(9500)

    # 2. Smoother compression (less aggressive)
    audio = compress_dynamic_range(
        audio,
        threshold=-18.0,
        ratio=2.5,
        attack=8.0,
        release=90.0
    )

    # 3. Gentle presence (not sharp overlay)
    presence = audio.high_pass_filter(2800).low_pass_filter(4200) + 2
    audio = audio.overlay(presence)

    # 4. Add a bit of mid clarity (this is new)
    mid = audio.high_pass_filter(900).low_pass_filter(1500) + 1
    audio = audio.overlay(mid)

    # 5. VERY subtle air
    air = audio.high_pass_filter(6000) - 12
    audio = audio.overlay(air)

    # 6. Remove obvious "effect" reverb (THIS is key)
    # No slapback reflections here

    # 7. Normalize
    audio = audio.normalize(headroom=1.0)

    out = BytesIO()
    audio.export(out, format="wav")
    return out.getvalue()   
    from io import BytesIO
    from pydub import AudioSegment
    from pydub.effects import compress_dynamic_range

    try:
        audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
    except Exception as exc:
        raise RuntimeError(f"Could not load WAV: {exc}") from exc

    # Verity works best as clean, centered, controlled speech
    audio = audio.set_channels(1)

    # Keep some body; don't over-thin it
    audio = audio.high_pass_filter(110)
    audio = audio.low_pass_filter(9000)

    # Firm but not smashed
    audio = compress_dynamic_range(
        audio,
        threshold=-21.0,
        ratio=3.0,
        attack=6.0,
        release=70.0,
    )

    # Main clarity band
    presence = audio.high_pass_filter(2600).low_pass_filter(4200) + 2
    audio = audio.overlay(presence)

    # Tiny bit of upper sheen
    air = audio.high_pass_filter(6000) - 10
    audio = audio.overlay(air)

    # Very subtle cockpit space — almost imperceptible
    early_reflection = (audio.high_pass_filter(1800) - 28)
    audio = audio.overlay(early_reflection, position=14)

    # No pitch shift
    # No speed trick

    audio = audio.normalize(headroom=1.0)

    out_buffer = BytesIO()
    audio.export(out_buffer, format="wav")
    return out_buffer.getvalue() 
    from io import BytesIO
    from pydub import AudioSegment
    from pydub.effects import compress_dynamic_range

    audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")

    # 1. Mono + keep a little more body
    audio = audio.set_channels(1)
    audio = audio.high_pass_filter(120)
    audio = audio.low_pass_filter(8500)

    # 2. Compression
    audio = compress_dynamic_range(
        audio,
        threshold=-20.0,
        ratio=4.0,
        attack=5.0,
        release=50.0
    )

    # 3. Presence, but less sharp
    presence = audio.high_pass_filter(2500).low_pass_filter(4200) + 3
    audio = audio.overlay(presence)

    # 4. Light cockpit reflection
    reflection = audio - 24
    audio = audio.overlay(reflection, position=12)

    # 5. Smaller pitch shift
    original_rate = audio.frame_rate
    audio = audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(original_rate * 1.06)}
    )
    audio = audio.set_frame_rate(original_rate)

    # 6. Normalize
    audio = audio.normalize(headroom=1.0)

    out = BytesIO()
    audio.export(out, format="wav")
    return out.getvalue()
    from io import BytesIO
    from pydub import AudioSegment
    from pydub.effects import compress_dynamic_range

    audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")

    # 1. Mono + clean low end
    audio = audio.set_channels(1)
    audio = audio.high_pass_filter(150)

    # 2. Compression (consistent AI tone)
    audio = compress_dynamic_range(
        audio,
        threshold=-20.0,
        ratio=4.0,
        attack=5.0,
        release=50.0
    )

    # 3. Presence boost (THE key part)
    presence = audio.high_pass_filter(2500).low_pass_filter(4500) + 5
    audio = audio.overlay(presence)

    # 4. Light "cockpit" reflections (keep subtle)
    reflection = audio - 22
    audio = audio.overlay(reflection, position=12)

    # 5. Proper pitch shift (~+2 semitones)
    original_rate = audio.frame_rate
    audio = audio._spawn(
        audio.raw_data,
        overrides={"frame_rate": int(original_rate * 1.12)}
    )
    audio = audio.set_frame_rate(original_rate)

    # 6. Normalize
    audio = audio.normalize(headroom=1.0)

    out = BytesIO()
    audio.export(out, format="wav")

    return out.getvalue()   
    try:
        audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
    except Exception as exc:
        raise RuntimeError(f"Could not load WAV: {exc}") from exc

    # 1. Clean the 'Human' Resonance (Tech-Priest sounds high-fidelity)
    audio = audio.set_channels(1)
    audio = audio.high_pass_filter(200)  # Removes that 'bassy' human chest sound
    audio = audio.low_pass_filter(8500)  # Roll off the digital 'fizz'

    # 2. Command Compression (Consistent ship computer volume)
    # This keeps the output authoritative and clear even during intense moments.
    audio = compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)

    # 3. The 'Tech-Priest' Crispness (Presence Boost)
    # Targets the 3.5kHz range where the 'computerized' clarity lives
    presence = audio.high_pass_filter(3000).low_pass_filter(5000) + 5
    audio = audio.overlay(presence)

    # 4. Metallic Cockpit Reflections (The secret sauce)
    # Instead of deep reverb, we use two very short 'slapback' reflections
    # to simulate the acoustics of a small, metal ship cabin.
    reflection1 = audio - 22  # 22dB quieter
    reflection2 = audio - 26
    audio = audio.overlay(reflection1, position=12)  # 12ms delay
    audio = audio.overlay(reflection2, position=28)  # 28ms delay

    # 5. Synthetic Speed Edge (Subtle pitch shift)
    # A tiny speed increase (1.2%) makes the voice feel 'generated' rather than recorded.
    original_rate = audio.frame_rate
    audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(original_rate * 1.012)})
    audio = audio.set_frame_rate(original_rate)

    # 6. Final Normalize
    audio = audio.normalize(headroom=1.0)

    out_buffer = BytesIO()
    audio.export(out_buffer, format="wav")
    return out_buffer.getvalue()   
    try:
        audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
    except Exception as exc:
        raise RuntimeError(f"Could not load WAV: {exc}") from exc

    # 1. Clean the 'Human' Resonance (Verity sounds high-fidelity)
    audio = audio.set_channels(1)
    audio = audio.high_pass_filter(200) # Removes that 'bassy' human chest sound
    audio = audio.low_pass_filter(8500)  # Roll off the digital 'fizz'
    
    # 2. Command Compression (Consistent ship computer volume)
    # This keeps the output authoritative and clear even during intense moments.
    audio = compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)

    # 3. The 'Verity' Crispness (Presence Boost)
    # Targets the 3.5kHz range where the 'computerized' clarity lives
    presence = audio.high_pass_filter(3000).low_pass_filter(5000) + 5 
    audio = audio.overlay(presence)

    # 4. Metallic Cockpit Reflections (The secret sauce)
    # Instead of deep reverb, we use two very short 'slapback' reflections 
    # to simulate the acoustics of a small, metal ship cabin.
    reflection1 = audio - 22 # 22dB quieter
    reflection2 = audio - 26
    audio = audio.overlay(reflection1, position=12) # 12ms delay
    audio = audio.overlay(reflection2, position=28) # 28ms delay

    # 5. Synthetic Speed Edge (Subtle pitch shift)
    # A tiny speed increase (1.2%) makes the voice feel 'generated' rather than recorded.
    original_rate = audio.frame_rate
    audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(original_rate * 1.012)})
    audio = audio.set_frame_rate(original_rate)

    # 6. Final Normalize
    audio = audio.normalize(headroom=1.0)

    out_buffer = BytesIO()
    audio.export(out_buffer, format="wav")
    return out_buffer.getvalue()
    try:
        audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
    except Exception as exc:
        raise RuntimeError(f"Could not load WAV: {exc}") from exc

    # 1. The "Clean" Cut (Verity is never 'boomy' or 'bassy')
    audio = audio.set_channels(1)
    audio = audio.high_pass_filter(180) # Cut the human chest resonance
    audio = audio.low_pass_filter(7500)  # Roll off the digital 'fizz'
    
    # 2. Authoritative Compression (Ship computers have zero volume drift)
    audio = compress_dynamic_range(audio, threshold=-24.0, ratio=5.0, attack=3.0, release=40.0)

    # 3. Comms Clarity (The "Verity Sheen")
    # This targets the 'nasal' and 'presence' frequencies (800Hz - 4kHz)
    mid_boost = audio.high_pass_filter(800).low_pass_filter(3500) + 3
    presence = audio.high_pass_filter(2500).low_pass_filter(5000) + 5
    audio = audio.overlay(mid_boost).overlay(presence)

    # 4. Metallic Cockpit Reflections (The secret sauce)
    # Verity sounds like she is in a small metal box. We use two very short delays.
    reflection1 = audio.high_pass_filter(1500) - 24
    reflection2 = audio.high_pass_filter(2000) - 28
    audio = audio.overlay(reflection1, position=10) # 10ms delay
    audio = audio.overlay(reflection2, position=22) # 22ms delay

    # 5. Synthetic Edge (Slightly 'off' speed)
    original_rate = audio.frame_rate
    audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(original_rate * 1.012)})
    audio = audio.set_frame_rate(original_rate)

    # 6. Final Normalization
    audio = audio.normalize(headroom=1.0)

    out_buffer = BytesIO()
    audio.export(out_buffer, format="wav")
    return out_buffer.getvalue()
    try:
        audio = AudioSegment.from_file(BytesIO(wav_bytes), format="wav")
    except Exception as exc:
        raise RuntimeError(f"Could not load WAV: {exc}") from exc

    # 1. Clean the 'Human' Mud (Verity is very crisp)
    audio = audio.set_channels(1)
    audio = audio.high_pass_filter(150) # Increased from 130 to remove more 'chest' resonance
    
    # 2. Tight Compression (Authoritative Ship Voice)
    audio = compress_dynamic_range(audio, threshold=-22.0, ratio=4.0, attack=5.0, release=50.0)

    # 3. The 'Verity' Sheen (Clarity Boost)
    # This targets the 2.5kHz - 4.5kHz range where the 'computer' clarity lives
    presence = audio.high_pass_filter(2500).low_pass_filter(4500) + 6 
    audio = audio.overlay(presence)

    # 4. Subtle Cockpit Space (Short 'slap' reflections)
    # We want it to sound like it's in a small metal cabin, not a hallway
    reflection = audio - 22
    audio = audio.overlay(reflection, position=12) # 12ms delay for that 'metallic' feel

    # 5. The Frame-Rate 'Sync' Trick
    # This gives it that slight synthetic 'edge'
    original_rate = audio.frame_rate
    audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(original_rate * 1.015)})
    audio = audio.set_frame_rate(original_rate)

    # 6. Normalize so she's always at the same volume
    audio = audio.normalize(headroom=1.0)

    out_buffer = BytesIO()
    audio.export(out_buffer, format="wav")
    return out_buffer.getvalue()
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
        processed_wav_bytes = apply_verity_effect(raw_wav_bytes)

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