#!/usr/bin/env python3
"""
Flask TTS server for OmniVoice.

POST /synthesize
    Body (JSON):
        {
            "voice":    "rahul" | "monika",   (required)
            "text":     "Hello world",        (required)
            "language": "English",            (optional, default auto-detect)
            "speed":    1.0                   (optional)
        }
    Response: WAV audio (audio/wav)

Usage:
    python server.py --model k2-fsa/OmniVoice --port 5000
"""

import argparse
import io
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
from flask import Flask, jsonify, request, send_file

from omnivoice import OmniVoice

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
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Field 'text' is required and must not be empty."}), 400

    voice_name = data.get("voice", "").strip().lower()
    if not voice_name:
        return jsonify({"error": "Field 'voice' is required (rahul | monika)."}), 400
    if voice_name not in voice_prompts:
        available = list(voice_prompts.keys())
        return jsonify({"error": f"Unknown voice '{voice_name}'. Available: {available}"}), 400

    language = data.get("language") or None
    speed = float(data.get("speed", 1.0))
    # Lower = faster, higher = better quality. 8–16 is a good speed/quality tradeoff.
    num_step = int(data.get("num_step", 16))
    # Set False to skip silence removal — saves ~10-20ms per request.
    postprocess_output = bool(data.get("postprocess_output", True))
    # Set False to skip denoising token — slight speedup.
    denoise = bool(data.get("denoise", True))

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
        f"voice={voice_name} | text_len={len(text)} | "
        f"gen={elapsed:.2f}s | audio={audio_duration:.2f}s | RTF={rtf:.3f}"
    )

    # Encode as WAV in memory and stream back
    buf = io.BytesIO()
    sf.write(buf, waveform, model.sampling_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)

    response = send_file(
        buf,
        mimetype="audio/wav",
        as_attachment=False,
        download_name="output.wav",
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
