from __future__ import annotations

import argparse
import asyncio
import importlib
import logging

from app.audio_capture import list_input_devices
from app.pipeline import PipelineConfig, run_pipeline


def configure_runtime_logging() -> None:
    """Suppress noisy library logs so terminal output stays readable."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

    try:
        transformers_logging = importlib.import_module("transformers.utils.logging")
        transformers_logging.set_verbosity_error()
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Terminal proof of concept for German sermon translation."
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available microphone input devices and exit.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Microphone input device index from --list-devices.",
    )
    parser.add_argument(
        "--asr-model",
        default="Qwen/Qwen3-ASR-0.6B",
        help="Qwen ASR model name or local path.",
    )
    parser.add_argument(
        "--translation-model",
        default="translategemma",
        help="Ollama model name to use for translation.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Base URL for the local Ollama server.",
    )
    parser.add_argument(
        "--min-speech-seconds",
        type=float,
        default=2.0,
        help="Minimum utterance length before sending audio to ASR.",
    )
    parser.add_argument(
        "--max-speech-seconds",
        type=float,
        default=15.0,
        help="Maximum utterance length before forcing a flush to ASR.",
    )
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=0.8,
        help="Silence duration that ends an utterance.",
    )
    return parser


async def _run_from_args(args: argparse.Namespace) -> None:
    config = PipelineConfig(
        device_index=args.device_index,
        asr_model_name=args.asr_model,
        translation_model_name=args.translation_model,
        ollama_url=args.ollama_url,
        min_speech_seconds=args.min_speech_seconds,
        max_speech_seconds=args.max_speech_seconds,
        silence_duration_seconds=args.silence_seconds,
    )
    await run_pipeline(config)


def cli() -> None:
    configure_runtime_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.list_devices:
        devices = list_input_devices()
        if not devices:
            print("No input devices found.")
            return

        print("Available input devices:")
        for device in devices:
            print(f"  {device}")
        return

    try:
        asyncio.run(_run_from_args(args))
    except KeyboardInterrupt:
        print("\n[shutdown] Stopped by user.")


if __name__ == "__main__":
    cli()
