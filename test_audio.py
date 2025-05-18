#!/usr/bin/env python3
import os
import argparse
from faster_whisper import WhisperModel
import torch
import audio_utils
import config
import pyttsx3


def test_stt():
    """Test Speech-to-Text functionality"""
    print("Testing Speech-to-Text...")

    # Initialize STT model
    print(f"Loading Whisper model: {config.STT_MODEL}")
    stt_model = WhisperModel(
        config.STT_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    # Record audio
    print("Please speak something...")
    audio_file = audio_utils.record_until_silence(
        threshold=config.SILENCE_THRESHOLD,
        silence_duration=config.SILENCE_DURATION,
        max_duration=config.MAX_RECORDING_DURATION
    )

    if audio_file is None:
        print("No speech detected.")
        return

    # Transcribe
    print("Transcribing...")
    segments, _ = stt_model.transcribe(audio_file)
    text = " ".join([segment.text for segment in segments])

    # Clean up
    os.unlink(audio_file)

    # Print result
    print(f"Transcription: {text}")


def test_tts(text=None):
    """Test Text-to-Speech functionality"""
    print("Testing Text-to-Speech...")

    # Initialize TTS engine
    print("Initializing pyttsx3...")
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', config.TTS_RATE)
    tts_engine.setProperty('volume', config.TTS_VOLUME)

    # Get available voices and set a voice
    voices = tts_engine.getProperty('voices')
    if voices:
        # Use the first voice by default
        tts_engine.setProperty('voice', voices[0].id)

    # Get text to speak
    if text is None:
        text = input("Enter text to speak: ")

    # Generate speech
    print(f"Speaking: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()


def main():
    parser = argparse.ArgumentParser(description="Test audio components")
    parser.add_argument("--stt", action="store_true",
                        help="Test Speech-to-Text")
    parser.add_argument("--tts", action="store_true",
                        help="Test Text-to-Speech")
    parser.add_argument("--text", type=str,
                        help="Text to speak (for TTS test)")

    args = parser.parse_args()

    if args.stt:
        test_stt()

    if args.tts:
        test_tts(args.text)

    if not args.stt and not args.tts:
        print("Please specify --stt or --tts to test components")
        print("Example: python test_audio.py --stt")
        print("Example: python test_audio.py --tts --text 'Hello, world!'")


if __name__ == "__main__":
    main()
