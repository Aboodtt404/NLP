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


def record_until_silence(threshold=0.03, silence_duration=1.0, max_duration=30.0, sample_rate=16000):
    """Record audio until silence is detected or max duration is reached."""
    
    # Initialize variables
    audio_data = []
    silence_samples = 0
    silence_threshold = int(silence_duration * sample_rate)
    max_samples = int(max_duration * sample_rate)
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Error in audio recording: {status}")
        audio_data.extend(indata[:, 0])
        
        # Check for silence
        if np.abs(indata[:, 0]).mean() < threshold:
            nonlocal silence_samples
            silence_samples += len(indata)
        else:
            silence_samples = 0
            
    # Start recording
    with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback):
        while len(audio_data) < max_samples and silence_samples < silence_threshold:
            sd.sleep(100)
    
    if not audio_data:
        return None
        
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_array = np.array(audio_data)
    sf.write(temp_file.name, audio_array, sample_rate)
    
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
