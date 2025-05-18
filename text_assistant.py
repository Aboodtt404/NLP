#!/usr/bin/env python3
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import config


class TextAssistant:
    def __init__(self):
        print("Initializing Text-based Assistant...")

        # Initialize Language Model (Mistral)
        print("Loading LLM model...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        self.llm = pipeline(
            "text-generation",
            model=config.LLM_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        # Conversation history
        self.conversation_history = [f"System: {config.SYSTEM_PROMPT}"]
        print("Text Assistant initialized and ready!")

    def generate_response(self, query):
        """Generate response using LLM"""
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

        return assistant_response

    def run(self):
        """Main loop for the assistant"""
        print("Text Assistant is running. Type 'exit' or 'quit' to stop.")

        while True:
            # Get user input
            query = input("You: ")

            # Check for exit command
            if query.lower() in ["exit", "quit", "stop", "bye"]:
                print("Exiting Text Assistant. Goodbye!")
                break

            # Generate response
            response = self.generate_response(query)

            # Print response
            print(f"Assistant: {response}")


if __name__ == "__main__":
    assistant = TextAssistant()
    assistant.run()
