#!/usr/bin/env python3
import os
import time
import tempfile
import numpy as np
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import audio_utils
import config
import pyttsx3


class SmartAssistant:
    def __init__(self):
        print("Initializing Smart Assistant...")

        # Initialize Speech-to-Text (Whisper)
        print("Loading STT model...")
        self.stt_model = WhisperModel(
            config.STT_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Language Model (SmolLM)
        print("Loading LLM model...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        
        # Load model with efficient settings
        print("Loading model with efficient settings...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # Initialize Text-to-Speech
        print("Initializing TTS...")
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', config.TTS_RATE)
        self.tts_engine.setProperty('volume', config.TTS_VOLUME)

        # Get available voices and set a voice
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Use the first voice by default
            self.tts_engine.setProperty('voice', voices[0].id)

        # Conversation history
        self.conversation_history = [f"System: {config.SYSTEM_PROMPT}"]
        print("Smart Assistant initialized and ready!")

    def listen(self):
        """Record audio from microphone using smart silence detection"""
        return audio_utils.record_until_silence(
            threshold=config.SILENCE_THRESHOLD,
            silence_duration=config.SILENCE_DURATION,
            max_duration=config.MAX_RECORDING_DURATION
        )

    def transcribe(self, audio_file):
        """Convert speech to text using Whisper"""
        segments, _ = self.stt_model.transcribe(audio_file)
        return " ".join([segment.text for segment in segments])

    def generate_response(self, query):
        """Generate response using SmolLM"""
        messages = [{"role": "user", "content": query}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        response = response.split("Assistant: ")[-1].strip()
        return response

    def speak(self, text):
        """Convert text to speech using pyttsx3"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def run(self):
        """Main loop for the assistant"""
        print("\nSmart Assistant is running. Say 'exit' or 'quit' to stop.")
        print("-" * 50)

        while True:
            # Listen for command
            print("\nListening...")
            audio_file = self.listen()

            # Transcribe audio to text
            query = self.transcribe(audio_file)
            print("\nYou said:", query)

            # Check for exit command
            if query.lower() in ["exit", "quit", "stop", "bye"]:
                print("\nExiting Smart Assistant. Goodbye!")
                self.speak("Goodbye!")
                break

            # Generate response
            print("\nThinking...")
            response = self.generate_response(query)
            print("\nAssistant:", response)
            print("-" * 50)

            # Speak response
            self.speak(response)


if __name__ == "__main__":
    assistant = SmartAssistant()
    assistant.run()
