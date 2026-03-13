from __future__ import annotations

import asyncio
from dataclasses import dataclass

import torch

from app.audio_capture import AudioChunk


@dataclass(slots=True)
class TranscriptSegment:
    text: str
    language: str
    started_at: float
    ended_at: float

    @property
    def duration_seconds(self) -> float:
        return self.ended_at - self.started_at


class QwenTranscriber:
    """Thin wrapper around Qwen3-ASR with Apple Silicon fallback logic."""

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-ASR-0.6B",
        language: str = "German",
        max_new_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.max_new_tokens = max_new_tokens

        self.device_name = "uninitialized"
        self.dtype_name = "unknown"
        self._model = None

    async def load(self) -> None:
        await asyncio.to_thread(self._load_sync)

    async def transcribe(self, chunk: AudioChunk) -> TranscriptSegment | None:
        return await asyncio.to_thread(self._transcribe_sync, chunk)

    def _load_sync(self) -> None:
        from qwen_asr import Qwen3ASRModel

        attempts: list[tuple[str, torch.dtype]] = []
        if torch.backends.mps.is_available():
            attempts.append(("mps", torch.float16))
        attempts.append(("cpu", torch.float32))

        failures: list[str] = []
        for device_name, dtype in attempts:
            try:
                self._model = Qwen3ASRModel.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                    device_map=device_name,
                    max_inference_batch_size=1,
                    max_new_tokens=self.max_new_tokens,
                )
                self.device_name = device_name
                self.dtype_name = str(dtype).replace("torch.", "")
                return
            except Exception as exc:  # pragma: no cover - depends on local runtime
                failures.append(f"{device_name}: {exc}")

        failure_text = "\n".join(failures)
        raise RuntimeError(
            f"Unable to load {self.model_name} on MPS or CPU.\n{failure_text}"
        )

    def _transcribe_sync(self, chunk: AudioChunk) -> TranscriptSegment | None:
        if self._model is None:
            raise RuntimeError("QwenTranscriber.load() must be called before transcribe().")

        results = self._model.transcribe(
            audio=(chunk.samples, chunk.sample_rate),
            language=self.language,
        )
        if not results:
            return None

        result = results[0]
        text = " ".join((getattr(result, "text", "") or "").split())
        if not text:
            return None

        detected_language = getattr(result, "language", self.language)
        return TranscriptSegment(
            text=text,
            language=str(detected_language),
            started_at=chunk.started_at,
            ended_at=chunk.ended_at,
        )
