import asyncio
import time


def blocking_microphone():
    for i in range(3):
        print(f"[mic] reading chunk {i}")
        time.sleep(1)  # blocking work
    print("[mic] done")


async def transcriber():
    for i in range(3):
        print(f"[asr] waiting for audio {i}")
        await asyncio.sleep(0.5)  # non-blocking async wait
    print("[asr] done")


async def translator():
    for i in range(3):
        print(f"[translate] waiting for text {i}")
        await asyncio.sleep(0.7)  # non-blocking async wait
    print("[translate] done")


async def main():
    asr_task = asyncio.create_task(transcriber())
    translate_task = asyncio.create_task(translator())
    mic_task = asyncio.create_task(asyncio.to_thread(blocking_microphone))

    await asyncio.gather(asr_task, translate_task, mic_task)


asyncio.run(main())
