#!/usr/bin/env python3
import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# Models
STT_MODEL = os.getenv("STT_MODEL", "base")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# Audio Settings
MAX_RECORDING_DURATION = int(os.getenv("MAX_RECORDING_DURATION", "15"))
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.03"))
SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", "1.0"))

# TTS Settings
TTS_RATE = int(os.getenv("TTS_RATE", "150"))  # Speech rate
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "0.9"))  # Volume (0.0 to 1.0)

# LLM Settings
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.95"))

# System prompts
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, friendly AI assistant. Answer questions concisely and accurately."
)
