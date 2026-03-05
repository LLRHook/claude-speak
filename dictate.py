"""
Voice dictation using Faster Whisper — replaces Win+H.

Press Ctrl+Space to start recording, press Ctrl+Space again to stop and transcribe.
Press Ctrl+Q to quit.

Requires admin privileges for global hotkey capture.
"""

import re
import time
import numpy as np
import sounddevice as sd
import keyboard
import pyautogui
import pyperclip

TOGGLE_KEY = "ctrl+space"
QUIT_KEY = "ctrl+q"
MODEL_SIZE = "large-v3"
SAMPLE_RATE = 16000

print(f"Loading Faster Whisper model '{MODEL_SIZE}' on CUDA (first run downloads ~3GB)...")
from faster_whisper import WhisperModel
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
print(f"Model '{MODEL_SIZE}' loaded on CUDA.\n")

recording = False
audio_frames = []
stream = None


def clean_text(text):
    """Clean up common Whisper transcription artifacts and fix punctuation."""
    text = re.sub(r'\b(uh|um|uh huh)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r' {2,}', ' ', text)
    def capitalize_after(match):
        return match.group(1) + ' ' + match.group(2).upper()
    text = re.sub(r'([.!?])\s+([a-z])', capitalize_after, text)
    if text:
        text = text[0].upper() + text[1:]
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text.strip()


def start_recording():
    global recording, audio_frames, stream
    recording = True
    audio_frames = []
    print("Recording... (press Ctrl+Space to stop)")

    def callback(indata, frames, time_info, status):
        if recording:
            audio_frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=callback,
    )
    stream.start()


def stop_and_transcribe():
    global recording, stream
    recording = False
    if stream:
        stream.stop()
        stream.close()
        stream = None

    if not audio_frames:
        print("No audio captured.")
        return

    audio = np.concatenate(audio_frames, axis=0).flatten()
    duration = len(audio) / SAMPLE_RATE
    if duration < 0.3:
        print("Too short, skipping.")
        return

    print(f"Transcribing {duration:.1f}s of audio...")
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        language="en",
        condition_on_previous_text=True,
        no_speech_threshold=0.6,
    )
    raw_text = " ".join(seg.text for seg in segments).strip()
    text = clean_text(raw_text)

    if text:
        print(f"-> {text}\n")
        time.sleep(0.05)
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
    else:
        print("(no speech detected)\n")


def toggle_recording():
    if recording:
        stop_and_transcribe()
    else:
        start_recording()


keyboard.add_hotkey(TOGGLE_KEY, toggle_recording)

print(f"Ready! Press [{TOGGLE_KEY}] to start/stop dictation, [{QUIT_KEY}] to quit.\n")

keyboard.wait(QUIT_KEY)
print("Exiting.")
