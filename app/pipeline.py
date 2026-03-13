from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass

from app.audio_capture import AudioChunk, MicrophoneSegmenter
from app.transcriber import QwenTranscriber, TranscriptSegment
from app.translator import OllamaTranslator


@dataclass(slots=True)
class PipelineConfig:
    sample_rate: int = 16_000
    chunk_size: int = 512
    silence_duration_seconds: float = 0.8
    min_speech_seconds: float = 2.0
    max_speech_seconds: float = 15.0
    device_index: int | None = None
    asr_model_name: str = "Qwen/Qwen3-ASR-0.6B"
    translation_model_name: str = "translategemma"
    ollama_url: str = "http://127.0.0.1:11434"
    source_language: str = "German"


class TranslatorPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.audio_queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        self.text_queue: asyncio.Queue[TranscriptSegment | None] = asyncio.Queue()
        self.stop_event = threading.Event()

        self.segmenter = MicrophoneSegmenter(
            sample_rate=config.sample_rate,
            chunk_size=config.chunk_size,
            silence_duration_seconds=config.silence_duration_seconds,
            min_speech_seconds=config.min_speech_seconds,
            max_speech_seconds=config.max_speech_seconds,
            device_index=config.device_index,
        )
        self.transcriber = QwenTranscriber(
            model_name=config.asr_model_name,
            language=config.source_language,
        )
        self.translator = OllamaTranslator(
            base_url=config.ollama_url,
            model_name=config.translation_model_name,
        )

    async def run(self) -> None:
        print("[startup] Checking Ollama connectivity...")
        await self.translator.healthcheck()

        print(
            "[startup] Loading Qwen ASR model. This can take a while the first time..."
        )
        await self.transcriber.load()
        print(
            f"[startup] Qwen ASR ready on {self.transcriber.device_name}"
            f" ({self.transcriber.dtype_name})."
        )
        print("[recording] Press Ctrl+C to stop.\n")

        loop = asyncio.get_running_loop()
        transcriber_task = asyncio.create_task(self._transcriber_loop())
        translator_task = asyncio.create_task(self._translator_loop())
        capture_task = asyncio.create_task(
            asyncio.to_thread(
                self.segmenter.run,
                loop=loop,
                output_queue=self.audio_queue,
                stop_event=self.stop_event,
            )
        )

        try:
            await capture_task
        except asyncio.CancelledError:
            await asyncio.shield(
                self._shutdown(
                    capture_task=capture_task,
                    transcriber_task=transcriber_task,
                    translator_task=translator_task,
                )
            )
            raise
        except Exception:
            await asyncio.shield(
                self._shutdown(
                    capture_task=capture_task,
                    transcriber_task=transcriber_task,
                    translator_task=translator_task,
                )
            )
            raise
        else:
            await self._shutdown(
                capture_task=capture_task,
                transcriber_task=transcriber_task,
                translator_task=translator_task,
            )

    async def _shutdown(
        self,
        *,
        capture_task: asyncio.Task[None],
        transcriber_task: asyncio.Task[None],
        translator_task: asyncio.Task[None],
    ) -> None:
        self.stop_event.set()

        with contextlib.suppress(asyncio.CancelledError, Exception):
            await capture_task

        await self.audio_queue.put(None)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await transcriber_task

        await self.text_queue.put(None)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await translator_task

        await self.translator.close()

    async def _transcriber_loop(self) -> None:
        while True:
            chunk = await self.audio_queue.get()
            try:
                if chunk is None:
                    return

                segment = await self.transcriber.transcribe(chunk)
                if segment is not None:
                    await self.text_queue.put(segment)
            except Exception as exc:  # pragma: no cover - runtime dependency failures
                print(f"[asr:error] {exc}")
            finally:
                self.audio_queue.task_done()

    async def _translator_loop(self) -> None:
        while True:
            segment = await self.text_queue.get()
            try:
                if segment is None:
                    return

                print(f"[DE] {segment.text}")
                translated = await self.translator.translate(
                    segment.text,
                    stream_output=True,
                )

                if not translated:
                    print("[no translation returned]", end="")

                print("\n")
            except Exception as exc:  # pragma: no cover - runtime dependency failures
                print(f"\n[translate:error] {exc}\n")
            finally:
                self.text_queue.task_done()


async def run_pipeline(config: PipelineConfig) -> None:
    pipeline = TranslatorPipeline(config)
    await pipeline.run()
