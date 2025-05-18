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

        # Initialize Language Model (Mistral)
        print("Loading LLM model...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        self.llm = pipeline(
            "text-generation",
            model=config.LLM_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        # Initialize Text-to-Speech
        print("Loading TTS engine...")
        self.tts_engine = pyttsx3.init()
        # Configure TTS properties
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
        if audio_file is None:
            return ""

        segments, _ = self.stt_model.transcribe(audio_file)
        text = " ".join([segment.text for segment in segments])
        os.unlink(audio_file)  # Clean up temporary file

        print(f"You said: {text}")
        return text

    def generate_response(self, query):
        """Generate response using LLM"""
        if not query.strip():
            return "I didn't catch that. Could you please repeat?"

        # Add to conversation history
        self.conversation_history.append(f"User: {query}")

        # Format prompt with conversation history
        # Keep last 5 exchanges for context
        prompt = "\n".join(self.conversation_history[-5:])
        prompt += f"\nAssistant:"

        # Generate response
        response = self.llm(
            prompt,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=True,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
        )[0]["generated_text"]

        # Extract just the assistant's response
        assistant_response = response.split("Assistant:")[-1].strip()

        # Add to conversation history
        self.conversation_history.append(f"Assistant: {assistant_response}")

        print(f"Assistant: {assistant_response}")
        return assistant_response

    def speak(self, text):
        """Convert text to speech and play it"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def run(self):
        """Main loop for the assistant"""
        print("Smart Assistant is running. Say 'exit' or 'quit' to stop.")

        while True:
            # Listen for command
            audio_file = self.listen()

            # Transcribe audio to text
            query = self.transcribe(audio_file)

            # Check for exit command
            if query.lower() in ["exit", "quit", "stop", "bye"]:
                print("Exiting Smart Assistant. Goodbye!")
                self.speak("Goodbye!")
                break

            # Generate response
            response = self.generate_response(query)

            # Speak response
            self.speak(response)


if __name__ == "__main__":
    assistant = SmartAssistant()
    assistant.run()
