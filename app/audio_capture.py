from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyaudio
import torch


@dataclass(slots=True)
class AudioChunk:
    samples: np.ndarray
    sample_rate: int
    started_at: float
    ended_at: float

    @property
    def duration_seconds(self) -> float:
        return float(len(self.samples) / self.sample_rate)


def list_input_devices() -> list[str]:
    """Return human-readable descriptions of available microphone devices."""
    audio = pyaudio.PyAudio()
    devices: list[str] = []
    try:
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            max_input_channels = int(info.get("maxInputChannels", 0))
            if max_input_channels <= 0:
                continue

            name = str(info.get("name", f"Device {index}"))
            default_rate = int(info.get("defaultSampleRate", 0))
            devices.append(
                f"[{index}] {name} | inputs={max_input_channels} | default_rate={default_rate}"
            )
    finally:
        audio.terminate()

    return devices


class MicrophoneSegmenter:
    """Capture microphone audio and emit utterance-sized chunks to an asyncio queue."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        chunk_size: int = 512,
        silence_duration_seconds: float = 0.8,
        min_speech_seconds: float = 2.0,
        max_speech_seconds: float = 15.0,
        device_index: int | None = None,
        vad_threshold: float = 0.5,
        speech_pad_ms: int = 200,
        pre_roll_seconds: float = 0.25,
    ) -> None:
        if chunk_size != 512:
            raise ValueError("Silero VADIterator expects a 512-sample chunk at 16kHz.")

        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_duration_seconds = silence_duration_seconds
        self.min_speech_seconds = min_speech_seconds
        self.max_speech_seconds = max_speech_seconds
        self.device_index = device_index
        self.vad_threshold = vad_threshold
        self.speech_pad_ms = speech_pad_ms
        self.pre_roll_chunks = max(1, round(pre_roll_seconds * sample_rate / chunk_size))

        self._vad_model: Any | None = None
        self._vad_iterator_cls: type[Any] | None = None

    def _load_vad(self) -> None:
        if self._vad_model is not None and self._vad_iterator_cls is not None:
            return

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            force_reload=False,
        )
        *_, vad_iterator_cls, _ = utils
        self._vad_model = model
        self._vad_iterator_cls = vad_iterator_cls

    def run(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        output_queue: asyncio.Queue[AudioChunk],
        stop_event: threading.Event,
    ) -> None:
        """Blocking capture loop intended to run in a worker thread."""
        self._load_vad()

        assert self._vad_model is not None
        assert self._vad_iterator_cls is not None

        vad_iterator = self._vad_iterator_cls(
            self._vad_model,
            threshold=self.vad_threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=int(self.silence_duration_seconds * 1000),
            speech_pad_ms=self.speech_pad_ms,
        )

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
        )

        pre_roll: deque[np.ndarray] = deque(maxlen=self.pre_roll_chunks)
        utterance_frames: list[np.ndarray] = []
        utterance_start: float | None = None
        speaking = False

        try:
            while not stop_event.is_set():
                raw = stream.read(self.chunk_size, exception_on_overflow=False)
                chunk_pcm = np.frombuffer(raw, dtype=np.int16).copy()
                chunk_float = torch.from_numpy(chunk_pcm.astype(np.float32) / 32768.0)
                event = vad_iterator(chunk_float, return_seconds=True)
                now = time.time()

                if event and "start" in event and not speaking:
                    speaking = True
                    utterance_frames = list(pre_roll)
                    utterance_frames.append(chunk_pcm)
                    utterance_start = now - (len(utterance_frames) * self.chunk_size / self.sample_rate)
                elif speaking:
                    utterance_frames.append(chunk_pcm)
                else:
                    pre_roll.append(chunk_pcm)

                if speaking and utterance_frames:
                    duration = len(utterance_frames) * self.chunk_size / self.sample_rate
                    if event and "end" in event:
                        self._emit_if_valid(
                            loop=loop,
                            output_queue=output_queue,
                            utterance_frames=utterance_frames,
                            started_at=utterance_start or now - duration,
                            ended_at=now,
                        )
                        speaking = False
                        utterance_frames = []
                        utterance_start = None
                        pre_roll.clear()
                    elif duration >= self.max_speech_seconds:
                        self._emit_if_valid(
                            loop=loop,
                            output_queue=output_queue,
                            utterance_frames=utterance_frames,
                            started_at=utterance_start or now - duration,
                            ended_at=now,
                        )
                        speaking = False
                        utterance_frames = []
                        utterance_start = None
                        pre_roll.clear()
                        vad_iterator.reset_states()
        finally:
            if utterance_frames:
                now = time.time()
                duration = len(utterance_frames) * self.chunk_size / self.sample_rate
                if duration >= self.min_speech_seconds:
                    self._emit_if_valid(
                        loop=loop,
                        output_queue=output_queue,
                        utterance_frames=utterance_frames,
                        started_at=utterance_start or now - duration,
                        ended_at=now,
                    )

            stream.stop_stream()
            stream.close()
            audio.terminate()

    def _emit_if_valid(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        output_queue: asyncio.Queue[AudioChunk],
        utterance_frames: list[np.ndarray],
        started_at: float,
        ended_at: float,
    ) -> None:
        concatenated = np.concatenate(utterance_frames)
        duration = len(concatenated) / self.sample_rate
        if duration < self.min_speech_seconds:
            return

        normalized = concatenated.astype(np.float32) / 32768.0
        chunk = AudioChunk(
            samples=normalized,
            sample_rate=self.sample_rate,
            started_at=started_at,
            ended_at=ended_at,
        )
        future = asyncio.run_coroutine_threadsafe(output_queue.put(chunk), loop)
        future.result()
