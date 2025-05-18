#!/usr/bin/env python3
import os
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np
import time


def record_audio(duration=5, sample_rate=16000, channels=1):
    """
    Record audio from the microphone for a specified duration

    Args:
        duration (int): Recording duration in seconds
        sample_rate (int): Audio sample rate
        channels (int): Number of audio channels

    Returns:
        str: Path to the temporary audio file
    """
    print(f"Recording for {duration} seconds...")

    # Record audio
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype='float32'
    )
    sd.wait()

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio_data, sample_rate)

    return temp_file.name


def record_until_silence(threshold=0.03, silence_duration=1.0, max_duration=10, sample_rate=16000):
    """
    Record audio until silence is detected or max duration is reached

    Args:
        threshold (float): Volume threshold for silence detection
        silence_duration (float): Duration of silence to stop recording
        max_duration (int): Maximum recording duration in seconds
        sample_rate (int): Audio sample rate

    Returns:
        str: Path to the temporary audio file
    """
    print("Listening... (speak now)")

    # Initialize recording parameters
    chunk_duration = 0.1  # Process audio in 100ms chunks
    chunk_samples = int(chunk_duration * sample_rate)
    silence_samples = int(silence_duration / chunk_duration)

    # Keep track of recording
    recording = []
    silence_counter = 0
    is_speaking = False

    # Start time tracking
    start_time = time.time()

    # Record until silence or max duration
    with sd.InputStream(samplerate=sample_rate, channels=1, callback=None) as stream:
        while True:
            # Read audio chunk
            audio_chunk, _ = stream.read(chunk_samples)
            recording.append(audio_chunk.copy())

            # Calculate volume
            volume = np.abs(audio_chunk).mean()

            # Check if speaking
            if volume > threshold:
                is_speaking = True
                silence_counter = 0
            elif is_speaking:
                silence_counter += 1

            # Check if we should stop
            elapsed_time = time.time() - start_time
            if (is_speaking and silence_counter >= silence_samples) or elapsed_time >= max_duration:
                break

            # If nothing detected for 2 seconds, start over
            if not is_speaking and elapsed_time > 2:
                recording = []
                start_time = time.time()

    # If no speech detected, return None
    if not is_speaking:
        print("No speech detected")
        return None

    # Combine all audio chunks
    audio_data = np.concatenate(recording, axis=0)

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio_data, sample_rate)

    return temp_file.name


def play_audio(file_path):
    """
    Play audio from a file

    Args:
        file_path (str): Path to the audio file
    """
    data, samplerate = sf.read(file_path)
    sd.play(data, samplerate)
    sd.wait()
