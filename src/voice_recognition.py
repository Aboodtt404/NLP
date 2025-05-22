import speech_recognition as sr
from gtts import gTTS
import os
import playsound
import tempfile


class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def listen(self):
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=5)

                print("Processing...")
                text = self.recognizer.recognize_google(audio)
                return text.lower()
        except sr.WaitTimeoutError:
            return "Timeout occurred while listening"
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results; {e}"

    def speak(self, text):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_filename = fp.name

            tts = gTTS(text=text, lang='en')
            tts.save(temp_filename)

            playsound.playsound(temp_filename)

            os.remove(temp_filename)

        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")


if __name__ == "__main__":
    assistant = VoiceAssistant()

    print("Say something...")
    text = assistant.listen()
    print(f"You said: {text}")

    assistant.speak("I heard what you said!")
