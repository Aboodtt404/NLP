#!/usr/bin/env python3
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run Smart Assistant")
    parser.add_argument("--text", action="store_true",
                        help="Run in text-only mode")
    parser.add_argument("--voice", action="store_true",
                        help="Run in voice mode")

    args = parser.parse_args()

    if args.text:
        from text_assistant import TextAssistant
        assistant = TextAssistant()
    elif args.voice:
        from assistant import SmartAssistant
        assistant = SmartAssistant()
    else:
        print("Please specify --text or --voice mode")
        print("Example: python run_assistant.py --text")
        print("Example: python run_assistant.py --voice")
        return

    assistant.run()


if __name__ == "__main__":
    main()
