"""Speak a sentence out loud, offline, with every voice installed on this machine.

Not machine learning, just a handy utility kept from the original repository.
pyttsx3 drives the operating system's own speech engine (NSSpeechSynthesizer
on macOS, SAPI5 on Windows, eSpeak on Linux), so it needs no internet.

Run:  python extras/text_to_speech.py "Model training finished"
      python extras/text_to_speech.py --list-voices
Needs: pip install pyttsx3
"""

import argparse

import pyttsx3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text", nargs="?", default="Model training finished.")
    parser.add_argument("--voice", help="Use only voices whose name contains this text, e.g. 'Samantha'.")
    parser.add_argument("--rate", type=int, default=180, help="Words per minute.")
    parser.add_argument("--list-voices", action="store_true")
    args = parser.parse_args()

    engine = pyttsx3.init()
    engine.setProperty("rate", args.rate)
    voices = engine.getProperty("voices")

    if args.list_voices:
        for voice in voices:
            print(f"{voice.name:<30} {voice.id}")
        return

    # The original queued the sentence once, then switched through every voice,
    # so it spoke once in whichever voice was set last. Here each chosen voice
    # says the line in turn.
    chosen = [v for v in voices if not args.voice or args.voice.lower() in v.name.lower()] or voices[:1]
    for voice in chosen:
        engine.setProperty("voice", voice.id)
        engine.say(args.text)
        engine.runAndWait()


if __name__ == "__main__":
    main()
