import speech_recognition as sr
import win32com.client as wincom
import os

speaker = wincom.Dispatch("SAPI.SpVoice")

def say(text):
    # os.system(f'echo {text}')
    speaker.Speak(text)

def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 1
        audio = r.listen(source)
        try:
            query = r.recognize_google(audio, language="en-in")
            print(f"User said {query}")
            return query
        except Exception as e:
            return "Some error occurred"

if __name__ == "__main__":
    print("Ramsha")
    say("Hello ramsha, I am JARVIS")
    while True:
        print("listening")
        text = takecommand()
        say(text)
