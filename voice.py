import pyttsx3
engine = pyttsx3.init()
engine.say('I am confirming 01 Beta Artificial Intelligence is now deployed and I am scanning ')
voices = engine.getProperty('voices')
for voice in voices:
    engine.setProperty('voice', voice.id)
engine.runAndWait()



