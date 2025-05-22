import json
import torch
from model import load_model
from voice_recognition import VoiceAssistant
from classical_nlp import ClassicalNLPProcessor, combine_predictions
import datetime
import random
import webbrowser
import urllib.parse
import re


class SmartAssistant:
    def __init__(self, model_path='../models/best_model.pth', mapping_path='../models/intent_mapping.json'):
        # Load transformer model and mappings
        with open(mapping_path, 'r') as f:
            self.intent_to_idx = json.load(f)
            self.idx_to_intent = {idx: intent for intent,
                                  idx in self.intent_to_idx.items()}

        self.model = load_model(
            model_path, num_classes=len(self.intent_to_idx))
        self.voice_assistant = VoiceAssistant()

        # Initialize classical NLP processor
        self.nlp_processor = ClassicalNLPProcessor()

        # Train classical model with existing data
        self.train_classical_model()

        self.conversation_history = []

        self.search_urls = {
            'web': "https://www.google.com/search?q={query}",
            'images': "https://www.google.com/search?q={query}&tbm=isch",
            'news': "https://news.google.com/search?q={query}",
            'videos': "https://www.youtube.com/results?search_query={query}",
            'maps': "https://www.google.com/maps/search/{query}",
            'shopping': "https://www.google.com/search?q={query}&tbm=shop",
            'scholar': "https://scholar.google.com/scholar?q={query}"
        }

        self.response_templates = {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Greetings! How may I assist you?"
            ],
            "goodbye": [
                "Goodbye! Have a great day!",
                "See you later!",
                "Bye! Let me know if you need anything else."
            ],
            "weather": [
                "I can help you check the weather. Which city would you like to know about?",
                "I'll look up the weather for you. Could you specify the location?"
            ],
            "time": [
                f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}",
                f"It's {datetime.datetime.now().strftime('%I:%M %p')} right now"
            ],
            "search": [
                "I'll search that for you right away.",
                "Let me look that up for you.",
                "I'll open a web search for that information."
            ],
            "unknown": [
                "I'm not sure I understood that. Could you please rephrase?",
                "I'm still learning. Could you try asking in a different way?",
                "I didn't quite catch that. Could you explain more?"
            ]
        }

    def train_classical_model(self):
        """Train the classical NLP model using existing training data."""
        try:
            with open('../data/data_full.json', 'r') as f:
                data = json.load(f)

            train_texts = []
            train_labels = []

            for item in data['train']:
                if isinstance(item, dict):
                    text, intent = item['text'], item['intent']
                else:
                    text, intent = item[0], item[1]

                if intent != 'oos':
                    train_texts.append(text)
                    train_labels.append(intent)

            self.nlp_processor.train_classical_model(train_texts, train_labels)
            print("Classical NLP model trained successfully")
        except Exception as e:
            print(f"Error training classical model: {str(e)}")

    def extract_search_query(self, text: str, intent: str) -> str:
        """Extract the actual search query from the text using classical NLP."""
        entities = self.nlp_processor.extract_entities(text, intent)
        if 'target' in entities:
            return entities['target']

        # Fallback to basic extraction if pattern matching fails
        words = text.lower().split()
        for i, word in enumerate(words):
            if word in ['show', 'find', 'search', 'get', 'about', 'for', 'of']:
                return ' '.join(words[i+1:])

        return text

    def web_search(self, query: str, search_type: str = 'web') -> str:
        """Perform web search with improved query extraction."""
        # Clean and extract the actual search query
        clean_query = self.extract_search_query(query, f'search_{search_type}')

        # Get search URL
        encoded_query = urllib.parse.quote(clean_query)
        search_url = self.search_urls.get(
            search_type, self.search_urls['web']).format(query=encoded_query)

        # Open in browser
        webbrowser.open(search_url)

        # Add to history
        self.add_to_history("user", query)

        # Generate response
        response = f"I've opened a {search_type} search for '{clean_query}'. You can find the results in your browser."
        self.add_to_history("assistant", response)
        return response

    def process_input(self, text: str) -> dict:
        """Process input using both classical and transformer-based approaches."""
        # Get transformer model prediction
        predicted_class, transformer_conf = self.model.predict(text)
        transformer_intent = self.idx_to_intent[predicted_class]

        # Get classical NLP analysis
        classical_result = self.nlp_processor.analyze_text(text)

        # Combine predictions
        final_intent, confidence = combine_predictions(
            classical_result, transformer_intent, transformer_conf)

        # Generate response based on final intent
        response = self.get_response(final_intent, confidence, text)

        return {
            "text": text,
            "intent": final_intent,
            "confidence": confidence,
            "response": response,
            "analysis": classical_result
        }

    def add_to_history(self, speaker: str, text: str):
        """Add conversation entry to history."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conversation_history.append({
            "timestamp": timestamp,
            "speaker": speaker,
            "text": text
        })

    def get_conversation_summary(self) -> str:
        """Get summary of recent conversations."""
        if not self.conversation_history:
            return "No previous conversations found."

        summary = "Recent conversations:\n"
        for entry in self.conversation_history[-5:]:
            summary += f"[{entry['timestamp']}] {entry['speaker'].capitalize()}: {entry['text']}\n"
        return summary

    def get_response(self, intent: str, confidence: float, text: str) -> str:
        """Generate response based on intent and confidence."""
        if confidence < 0.6:  # Increased threshold for more conservative responses
            return random.choice(self.response_templates["unknown"])

        self.add_to_history("user", text)
        response = ""

        # Handle different intents
        if "search" in intent:
            search_type = intent.split('_')[1] if '_' in intent else 'web'
            response = self.web_search(text, search_type)
        elif "greeting" in intent:
            response = random.choice(self.response_templates["greeting"])
        elif "goodbye" in intent:
            response = random.choice(self.response_templates["goodbye"])
        elif "weather" in intent:
            response = random.choice(self.response_templates["weather"])
        elif "time" in intent:
            response = random.choice(self.response_templates["time"])
        elif "history" in intent:
            response = self.get_conversation_summary()
        else:
            response = f"I understand you want to {intent.replace('_', ' ')}. Let me help you with that."

        self.add_to_history("assistant", response)
        return response

    def run_voice_interaction(self):
        """Run voice interaction mode."""
        print("Smart Assistant is ready! Say something...")
        self.print_help()

        while True:
            text = self.voice_assistant.listen()

            if text.lower() in ["exit", "quit", "goodbye", "bye"]:
                response = random.choice(self.response_templates["goodbye"])
                print("Assistant:", response)
                self.voice_assistant.speak(response)
                break

            result = self.process_input(text)

            print("\nYou said:", result["text"])
            print("Detected Intent:", result["intent"])
            print("Confidence:", f"{result['confidence']:.2f}")
            print("Assistant:", result["response"])

            self.voice_assistant.speak(result["response"])

    def print_help(self):
        """Print help information."""
        print("\nTry these commands:")
        print("Search Categories:")
        print("- 'Show me images of cats'")
        print("- 'Find news about technology'")
        print("- 'Search for videos about cooking'")
        print("- 'Look up directions to nearest coffee shop'")
        print("- 'Find scientific papers about AI'")
        print("\nOther Features:")
        print("- 'What time is it?'")
        print("- 'Show me our conversation history'")
        print("- Say 'exit' or 'quit' to end")


def main():
    try:
        assistant = SmartAssistant()
        print("Initializing Smart Assistant...")
        print("Using model with", len(assistant.intent_to_idx), "intents")
        print("\nType 'voice' to start voice interaction")
        print("Type 'text' to start text interaction")
        print("Type 'exit' to quit")

        assistant.print_help()

        while True:
            mode = input("\nEnter mode: ").lower()

            if mode == "exit":
                print("Goodbye!")
                break
            elif mode == "voice":
                assistant.run_voice_interaction()
            elif mode == "text":
                while True:
                    text = input("\nYou: ")
                    if text.lower() in ["exit", "quit", "back"]:
                        break

                    result = assistant.process_input(text)
                    print("Intent:", result["intent"])
                    print("Confidence:", f"{result['confidence']:.2f}")
                    print("Assistant:", result["response"])

                    # Print detailed analysis in text mode
                    if result['confidence'] < 0.8:
                        print("\nDetailed Analysis:")
                        print("Pattern Match:",
                              result['analysis']['pattern_match'])
                        print("ML Classification:",
                              result['analysis']['ml_classification'])
                        print("Text Stats:", result['analysis']['text_stats'])
            else:
                print("Invalid mode. Please try again.")
    except KeyboardInterrupt:
        print("\nExiting Smart Assistant...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Exiting Smart Assistant...")


if __name__ == "__main__":
    main()
