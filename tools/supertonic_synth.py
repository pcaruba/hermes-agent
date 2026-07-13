#!/usr/bin/env python3
"""Supertonic 3 TTS wrapper for Hermes command TTS provider.

Reads text from --input_path, generates audio via Supertonic 3 (ONNX),
converts to the format requested by --output_path extension.

Hermes template variables: {input_path}, {output_path}, {voice}, {speed}

Usage:
    supertonic_synth.py --input_path /tmp/text.txt --output_path /tmp/out.ogg \
                        --voice M4 --speed 1.05

Requires: pip install supertonic numpy soundfile
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description="Supertonic 3 TTS for Hermes")
    parser.add_argument("--input_path", required=True, help="Path to text file")
    parser.add_argument("--output_path", required=True, help="Path for output audio")
    parser.add_argument("--voice", default="M4", help="Voice name (M1-M5, F1-F5)")
    parser.add_argument("--speed", type=float, default=1.05,
                        help="Speech speed multiplier (default: 1.05)")
    args = parser.parse_args()

    text = Path(args.input_path).read_text(encoding="utf-8").strip()
    if not text:
        print("Error: empty input text", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from supertonic import TTS
    except ImportError:
        print("Error: supertonic not installed. Run: pip install supertonic", file=sys.stderr)
        sys.exit(1)

    tts = TTS(auto_download=True)
    style = tts.get_voice_style(args.voice)
    wav, dur = tts.synthesize(text, voice_style=style, speed=args.speed, lang="en")
    sr = tts.sample_rate  # 44100 Hz

    # Supertonic returns (1, N) stereo — squeeze to mono
    wav = wav.squeeze()

    # Trim to exact duration
    audio_len = int(sr * float(dur[0]))
    wav = wav[:audio_len]

    # Determine target format from output_path extension
    ext = output_path.suffix.lower()

    if ext == ".ogg":
        # Write temp WAV, then convert to OGG Opus (Telegram native voice msg)
        tmp_wav = output_path.with_suffix(".wav")
        sf.write(str(tmp_wav), wav, sr)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_wav),
             "-c:a", "libopus", "-b:a", "64k", str(output_path)],
            capture_output=True, check=True, timeout=30,
        )
        tmp_wav.unlink(missing_ok=True)
    elif ext == ".mp3":
        tmp_wav = output_path.with_suffix(".wav")
        sf.write(str(tmp_wav), wav, 24000)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_wav),
             "-c:a", "libmp3lame", "-b:a", "128k",
             str(output_path)],
            capture_output=True, check=True, timeout=30,
        )
        tmp_wav.unlink(missing_ok=True)
    else:
        # WAV or other — write directly
        sf.write(str(output_path), wav, 24000)

    if not output_path.exists():
        print(f"Error: output not created at {output_path}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
