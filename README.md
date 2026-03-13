# Live Sermon Translator

Terminal proof of concept for live German sermon transcription with Qwen3-ASR and English translation through `translategemma` on Ollama.

## Requirements

- Python 3.11+
- `uv`
- Ollama with `translategemma`
- `portaudio` installed on macOS for `pyaudio`

## Setup

```bash
brew install portaudio
ollama serve #or just install or run ollama if you already have it installed
ollama pull translategemma
uv sync
```

## Run

```bash
uv run church-translator --list-devices
uv run church-translator --device-index 0
```

Or:

```bash
uv run python -m app.main --device-index 0
```

Press `Ctrl+C` to stop the live pipeline.

## What It Does

The terminal pipeline works like this:

1. Capture microphone audio from the selected input device.
2. Use Silero VAD to split the audio into speech-sized utterances.
3. Transcribe each utterance with Qwen3-ASR.
4. Translate the German transcript to English with `translategemma` via Ollama.
5. Print the English output to the terminal.

## CLI Arguments

You can inspect the full CLI help with:

```bash
uv run church-translator --help
```

Supported arguments:

- `--list-devices`: List available microphone input devices and exit.
- `--device-index <int>`: Select the microphone input device index to use.
- `--asr-model <name-or-path>`: Override the ASR model. Default: `Qwen/Qwen3-ASR-0.6B`.
- `--translation-model <name>`: Override the Ollama translation model. Default: `translategemma`.
- `--ollama-url <url>`: Set the Ollama base URL. Default: `http://127.0.0.1:11434`.
- `--min-speech-seconds <float>`: Minimum utterance length before sending audio to ASR. Default: `2.0`.
- `--max-speech-seconds <float>`: Maximum utterance length before forcing an utterance flush. Default: `15.0`.
- `--silence-seconds <float>`: Silence duration that ends the current utterance. Default: `0.8`.

## Common Commands

List available input devices:

```bash
uv run church-translator --list-devices
```

Run with a specific microphone:

```bash
uv run church-translator --device-index 0
```

Run with a custom Ollama endpoint:

```bash
uv run church-translator --device-index 0 --ollama-url http://192.168.1.20:11434
```

Tune utterance segmentation for faster flushing:

```bash
uv run church-translator --device-index 0 --min-speech-seconds 1.5 --silence-seconds 0.5
```

Force shorter maximum utterances:

```bash
uv run church-translator --device-index 0 --max-speech-seconds 8
```

## Notes

- `Qwen/Qwen3-ASR-0.6B` is the default because it is the most practical local choice for Apple Silicon.
- The first startup can take longer because models and runtime components may need to initialize.
- If Ollama is not running or `translategemma` is not pulled locally, startup will fail early with a clear error.
