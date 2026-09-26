import speech_recognition as sr


MICROPHONE_NAME = "Headset (realme Buds Air8)"


def find_microphone():
    microphones = sr.Microphone.list_microphone_names()

    print()
    print("[MICROPHONES] Available devices:")

    for index, name in enumerate(microphones):
        print(f"  {index}: {name}")

    for index, name in enumerate(microphones):
        if name == MICROPHONE_NAME:
            print()
            print(f"[MICROPHONE] Selected: {name}")
            print(f"[MICROPHONE] Device index: {index}")
            return index

    print()
    print("[MICROPHONE ERROR]")
    print(f"Could not find: {MICROPHONE_NAME}")

    return None


class VoiceInput:

    def __init__(self):

        self.recognizer = sr.Recognizer()

        # More forgiving for normal speech.
        self.recognizer.pause_threshold = 1.0
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.5

        self.microphone_index = find_microphone()

        self.microphone = None

        if self.microphone_index is not None:
            self.microphone = sr.Microphone(
                device_index=self.microphone_index
            )

            # Calibrate only ONCE.
            print()
            print("[MICROPHONE] Initial calibration...")
            print("Please remain silent for 2 seconds.")

            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(
                        source,
                        duration=2
                    )

                print(
                    "[MICROPHONE] Calibration complete."
                )

                print(
                    f"[MICROPHONE] Energy threshold: "
                    f"{self.recognizer.energy_threshold:.0f}"
                )

            except OSError as error:

                print()
                print("[MICROPHONE ERROR]")
                print(error)

                self.microphone = None

    def listen_once(self):

        if self.microphone is None:
            return ""

        print()
        print("=" * 40)
        print("          VOICE INPUT")
        print("=" * 40)

        print()
        print("Speak now...")
        print("(You have 10 seconds)")

        try:

            with self.microphone as source:

                try:

                    audio = self.recognizer.listen(
                        source,
                        timeout=10,
                        phrase_time_limit=10
                    )

                except sr.WaitTimeoutError:

                    print()
                    print("[VOICE] No speech detected.")

                    return ""

        except OSError as error:

            print()
            print("[MICROPHONE ERROR]")
            print(error)

            return ""

        print()
        print("[VOICE] Processing...")

        try:

            text = self.recognizer.recognize_google(
                audio,
                language="en-IN"
            )

            text = text.strip()

            if text:

                print()
                print("[VOICE TEXT]", text)

                return text

            print()
            print("[VOICE] Empty result.")

            return ""

        except sr.UnknownValueError:

            print()
            print(
                "[VOICE] Speech was detected, "
                "but could not be understood."
            )

            return ""

        except sr.RequestError as error:

            print()
            print(
                "[VOICE ERROR] Speech recognition "
                "service unavailable."
            )

            print(error)

            return ""


# ---------------------------------------------------------
# Backward-compatible function
# ---------------------------------------------------------

_voice_input = None


def listen_once():

    global _voice_input

    if _voice_input is None:
        _voice_input = VoiceInput()

    return _voice_input.listen_once()


# ---------------------------------------------------------
# Standalone test
# ---------------------------------------------------------

def main():

    print()
    print("=" * 50)
    print("        SIgnFlow VOICE INPUT TEST")
    print("=" * 50)

    result = listen_once()

    print()
    print("=" * 50)
    print("                 RESULT")
    print("=" * 50)

    if result:
        print(result)
    else:
        print("No text recognized.")

    print("=" * 50)


if __name__ == "__main__":
    main()