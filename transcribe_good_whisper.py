import speech_recognition as sr
import pyaudio
import wave
import threading
import time
from datetime import datetime

def record_audio(stream, frames, audio_format, channels, rate, chunk):
    while True:
        data = stream.read(chunk)
        frames.append(data)

def transcribe_audio(recognizer, audio):
    try:
        print("transcribe start: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        text = recognizer.recognize_whisper(audio, language="english", model="turbo")
        print("Transcription: " + text)
        print("transcribe end: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

def main():
    # Audio recording parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 6

    audio = pyaudio.PyAudio()

    # Start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    # Start a new thread to record audio
    recording_thread = threading.Thread(target=record_audio, args=(stream, frames, FORMAT, CHANNELS, RATE, CHUNK))
    recording_thread.start()

    recognizer = sr.Recognizer()

    try:
        while True:
            time.sleep(RECORD_SECONDS)

            # Create an AudioData instance
            audio_data = b''.join(frames)
            audio_segment = sr.AudioData(audio_data, RATE, 2)

            # Start a new thread for transcription
            transcription_thread = threading.Thread(target=transcribe_audio, args=(recognizer, audio_segment))
            transcription_thread.start()

            # Clear frames for the next recording
            frames.clear()

    except KeyboardInterrupt:
        print("Stopping recording...")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    main()