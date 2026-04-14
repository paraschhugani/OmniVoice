#!/usr/bin/env python3
"""
Flask TTS server for OmniVoice.

POST /synthesize
    Body (JSON):
        {
            "voice":    "rahul" | "monika",   (required)
            "text":     "Hello world",        (required)
            "language": "English",            (optional, default auto-detect)
            "speed":    1.0,                  (optional)
            "stream":   false                 (optional) if true, returns raw 16-bit
                                              signed PCM chunks sentence-by-sentence
                                              (audio/pcm). TTFB = first-sentence latency.
                                              Headers: X-Sample-Rate, X-Audio-Channels,
                                              X-Audio-Bit-Depth describe the format.
        }
    Response: WAV audio (audio/wav) or streaming audio/pcm when stream=true

Usage:
    python server.py --model k2-fsa/OmniVoice --port 5000
"""

import argparse
import io
import logging
import os
import subprocess
import time

import numpy as np
import soundfile as sf
import torch
from flask import Flask, Response, jsonify, request, send_file, stream_with_context

from omnivoice import OmniVoice
from omnivoice.utils.text import chunk_text_punctuation

# ---------------------------------------------------------------------------
# Reference voices — add more entries here as needed
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VOICE_FILES = {
    "rahul": os.path.join(_BASE_DIR, "rahul.mp3"),
    "monika": os.path.join(_BASE_DIR, "monika.mp3"),
}

# ---------------------------------------------------------------------------
# Globals (populated at startup)
# ---------------------------------------------------------------------------
model: OmniVoice = None
voice_prompts: dict = {}   # name -> VoiceClonePrompt (pre-computed)

# ---------------------------------------------------------------------------
# Gunicorn entry point — load model once when worker starts
# Called automatically by Gunicorn via --preload or post_fork hook.
# ---------------------------------------------------------------------------
def _load_model_for_gunicorn():
    """Load model into globals. Called at module import when using Gunicorn."""
    import os as _os
    checkpoint = _os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
    device = _os.environ.get("OMNIVOICE_DEVICE") or get_best_device()
    load_asr = _os.environ.get("OMNIVOICE_LOAD_ASR", "0") == "1"

    global model
    if model is not None:
        return  # already loaded

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = OmniVoice.from_pretrained(
        checkpoint, device_map=device, dtype=torch.float16, load_asr=load_asr
    )
    for name, path in VOICE_FILES.items():
        if not _os.path.exists(path):
            continue
        voice_prompts[name] = model.create_voice_clone_prompt(ref_audio=path)

    if voice_prompts:
        first_prompt = next(iter(voice_prompts.values()))
        try:
            model.generate(text="Hello.", voice_clone_prompt=first_prompt, num_step=16)
        except Exception:
            pass


# Auto-load when imported by Gunicorn (GUNICORN_CMD_ARGS is set by gunicorn)
if "GUNICORN_CMD_ARGS" in __import__("os").environ or \
        __import__("os").environ.get("OMNIVOICE_AUTOLOAD") == "1":
    _load_model_for_gunicorn()


def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Audio encoding helper
# ---------------------------------------------------------------------------
_FORMAT_MIME = {
    "mp3":  "audio/mpeg",
    "opus": "audio/ogg; codecs=opus",
    "aac":  "audio/aac",
    "flac": "audio/flac",
    "wav":  "audio/wav",
    "pcm":  "audio/pcm",
}

def _encode_audio(waveform: np.ndarray, sampling_rate: int, fmt: str) -> tuple:
    """Encode waveform to the requested format. Returns (bytes, mimetype)."""
    fmt = fmt.lower()

    if fmt == "pcm":
        pcm = (waveform * 32767).astype(np.int16).tobytes()
        return pcm, "audio/pcm"

    # Build WAV in memory — base for all other formats
    wav_buf = io.BytesIO()
    sf.write(wav_buf, waveform, sampling_rate, format="WAV", subtype="PCM_16")
    wav_buf.seek(0)

    if fmt == "wav":
        return wav_buf.read(), "audio/wav"

    # Use ffmpeg for mp3 / opus / aac / flac
    ffmpeg_fmt = {"mp3": "mp3", "opus": "opus", "aac": "adts", "flac": "flac"}.get(fmt)
    if ffmpeg_fmt:
        try:
            proc = subprocess.run(
                ["ffmpeg", "-f", "wav", "-i", "pipe:0",
                 "-f", ffmpeg_fmt, "-loglevel", "error", "pipe:1"],
                input=wav_buf.read(),
                capture_output=True,
                timeout=30,
            )
            if proc.returncode == 0:
                return proc.stdout, _FORMAT_MIME.get(fmt, "audio/mpeg")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logging.warning(f"ffmpeg not available or timed out — falling back to WAV")

    # Fallback to WAV
    wav_buf.seek(0)
    return wav_buf.read(), "audio/wav"


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

# Target ~2-3 s of audio per chunk; tune up for better quality, down for lower latency.
_STREAM_CHUNK_CHARS = 120


def _iter_stream_pcm(
    text: str,
    voice_name: str,
    language,
    speed: float,
    num_step: int,
    postprocess_output: bool,
    denoise: bool,
):
    """Yield raw signed-16-bit PCM bytes for each sentence chunk.

    Splits *text* at sentence boundaries, generates audio for each chunk
    independently, and yields bytes immediately — so the caller (client)
    receives the first chunk without waiting for the full audio to be ready.

    PCM format: 16-bit signed little-endian, mono, model.sampling_rate Hz.
    """
    chunks = chunk_text_punctuation(text, chunk_len=_STREAM_CHUNK_CHARS, min_chunk_len=10)
    if not chunks:
        chunks = [text]

    voice_prompt = voice_prompts[voice_name]

    for chunk in chunks:
        try:
            audios = model.generate(
                text=chunk,
                language=language,
                voice_clone_prompt=voice_prompt,
                speed=speed if speed != 1.0 else None,
                num_step=num_step,
                postprocess_output=postprocess_output,
                denoise=denoise,
            )
            waveform = audios[0]  # float32 (T,)
            pcm = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            yield pcm
        except Exception:
            logging.exception("Streaming chunk generation failed — stopping stream")
            return


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/synthesize", methods=["POST"])
@app.route("/audio/speech", methods=["POST"])
@app.route("/v1/audio/speech", methods=["POST"])
def synthesize():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    # OpenAI TTS uses "input"; our own API uses "text" — support both
    text = (data.get("input") or data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Field 'input' (or 'text') is required and must not be empty."}), 400

    voice_name = data.get("voice", "").strip().lower()
    if not voice_name:
        return jsonify({"error": "Field 'voice' is required (rahul | monika)."}), 400
    if voice_name not in voice_prompts:
        available = list(voice_prompts.keys())
        return jsonify({"error": f"Unknown voice '{voice_name}'. Available: {available}"}), 400

    language = data.get("language") or None
    speed = float(data.get("speed", 1.0))
    num_step = int(data.get("num_step", 32))
    postprocess_output = bool(data.get("postprocess_output", True))
    denoise = bool(data.get("denoise", True))
    # OpenAI TTS response_format: mp3 | opus | aac | flac | wav | pcm
    response_format = data.get("response_format", "mp3").lower()
    stream = bool(data.get("stream", True))

    # --- Streaming path ---
    # Returns raw 16-bit signed PCM chunks as they are generated sentence by
    # sentence. TTFB drops from full-generation time to first-sentence time.
    if stream:
        logging.info(
            f"[stream] voice={voice_name} | text_len={len(text)} | num_step={num_step}"
        )
        gen = stream_with_context(
            _iter_stream_pcm(
                text=text,
                voice_name=voice_name,
                language=language,
                speed=speed,
                num_step=num_step,
                postprocess_output=postprocess_output,
                denoise=denoise,
            )
        )
        return Response(
            gen,
            mimetype="audio/pcm",
            headers={
                "X-Sample-Rate": str(model.sampling_rate),
                "X-Audio-Channels": "1",
                "X-Audio-Bit-Depth": "16",
                "Transfer-Encoding": "chunked",
            },
        )

    # --- Non-streaming path (original behaviour) ---
    try:
        t0 = time.perf_counter()
        audios = model.generate(
            text=text,
            language=language,
            voice_clone_prompt=voice_prompts[voice_name],
            speed=speed if speed != 1.0 else None,
            num_step=num_step,
            postprocess_output=postprocess_output,
            denoise=denoise,
        )
        elapsed = time.perf_counter() - t0
    except Exception as e:
        logging.exception("Generation failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    waveform = audios[0]
    audio_duration = len(waveform) / model.sampling_rate
    rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")
    logging.info(
        f"voice={voice_name} | fmt={response_format} | text_len={len(text)} | "
        f"gen={elapsed:.2f}s | audio={audio_duration:.2f}s | RTF={rtf:.3f}"
    )

    audio_bytes, mimetype = _encode_audio(waveform, model.sampling_rate, response_format)
    buf = io.BytesIO(audio_bytes)

    response = send_file(
        buf,
        mimetype=mimetype,
        as_attachment=False,
        download_name=f"output.{response_format}",
    )
    response.headers["X-RTF"] = f"{rtf:.4f}"
    response.headers["X-Generation-Time"] = f"{elapsed:.3f}s"
    response.headers["X-Audio-Duration"] = f"{audio_duration:.3f}s"
    return response


@app.route("/voices", methods=["GET"])
def list_voices():
    """List available voice names."""
    return jsonify({"voices": list(voice_prompts.keys())})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        prog="server",
        description="OmniVoice Flask TTS server",
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR. Reference text won't be auto-transcribed.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="torch.compile the LLM backbone. First request is slow (compilation), "
        "subsequent requests are faster.",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        default=False,
        help="Use Flash Attention 2 (requires flash-attn package, CUDA only).",
    )
    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    args = build_parser().parse_args()

    device = args.device or get_best_device()

    global model
    load_kwargs = dict(
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
    )
    if args.flash_attn:
        load_kwargs["attn_implementation"] = "flash_attention_2"
        logging.info("Flash Attention 2 enabled.")

    logging.info(f"Loading OmniVoice from '{args.model}' on {device} ...")
    model = OmniVoice.from_pretrained(args.model, **load_kwargs)
    logging.info("Model loaded.")

    if args.compile:
        logging.info("Compiling LLM backbone with torch.compile ...")
        model.llm = torch.compile(model.llm, mode="reduce-overhead")
        logging.info("Compilation done (first request will trigger tracing).")

    # Pre-compute voice clone prompts so inference is fast per request
    for name, path in VOICE_FILES.items():
        if not os.path.exists(path):
            logging.warning(f"Reference audio not found for '{name}': {path}")
            continue
        logging.info(f"Building voice prompt for '{name}' from {path} ...")
        voice_prompts[name] = model.create_voice_clone_prompt(ref_audio=path)
        logging.info(f"  '{name}' ready.")

    if not voice_prompts:
        logging.error("No voice prompts loaded — check that .mp3 files exist.")

    # Warmup: run a dummy inference to pre-compile CUDA kernels.
    # Without this, the first real request is 2–3x slower.
    if voice_prompts:
        logging.info("Warming up model (dummy inference) ...")
        first_prompt = next(iter(voice_prompts.values()))
        try:
            model.generate(
                text="Hello.",
                voice_clone_prompt=first_prompt,
                num_step=16,
            )
            logging.info("Warmup done.")
        except Exception as e:
            logging.warning(f"Warmup failed (non-fatal): {e}")

    # Enable TF32 on Ampere+ NVIDIA GPUs — free ~10-20% speedup, no quality loss.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("TF32 enabled.")

    logging.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
