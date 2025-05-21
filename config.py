#!/usr/bin/env python3
import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# Model configurations
STT_MODEL = "tiny"  # Whisper model size (tiny, base, small, medium, large)
LLM_MODEL = "HuggingFaceTB/SmolLM-360M-Instruct"  # Small but capable model

# Audio recording configurations
SILENCE_THRESHOLD = 0.03
SILENCE_DURATION = 1.0  # seconds
MAX_RECORDING_DURATION = 30.0  # seconds

# Text generation configurations
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2  # Recommended for this model
TOP_P = 0.9  # Recommended for this model

# TTS configurations
TTS_RATE = 175  # Speed of speech
TTS_VOLUME = 1.0  # Volume (0.0 to 1.0)

# System prompt for the assistant
SYSTEM_PROMPT = """You are a helpful AI assistant. Be concise, friendly, and provide accurate information."""
