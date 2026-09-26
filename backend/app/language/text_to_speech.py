import subprocess
import os
import winsound


LANGUAGE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL = os.path.join(
    LANGUAGE_DIR,
    "hi_IN-pratham-medium.onnx"
)

OUTPUT_FILE = os.path.join(
    LANGUAGE_DIR,
    "speech_output.wav"
)

INPUT_FILE = os.path.join(
    LANGUAGE_DIR,
    "tts_input.txt"
)


# ============================================================
# TEXT TO SPEECH
# ============================================================

def text_to_speech(text):

    if not text:
        print("[TTS] No text received.")
        return False

    text = str(text).strip()

    print()
    print("========================================")
    print("[PIPER TTS] Starting...")
    print("========================================")
    print("[TTS TEXT]:", repr(text))
    print("[TTS MODEL]:", MODEL)
    print("[TTS OUTPUT]:", OUTPUT_FILE)


    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL):

        print("[TTS ERROR] Model not found:")
        print(MODEL)

        return False


    # --------------------------------------------------------
    # Remove old output
    # --------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):

        try:
            os.remove(OUTPUT_FILE)
        except Exception as error:
            print(
                "[TTS WARNING] Could not remove old WAV:"
            )
            print(repr(error))


    # --------------------------------------------------------
    # Write Hindi text as UTF-8
    # --------------------------------------------------------

    try:

        with open(
            INPUT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(text)

        print(
            "[TTS] Input file created:"
        )
        print(INPUT_FILE)

    except Exception as error:

        print(
            "[TTS ERROR] Could not write input file:"
        )
        print(repr(error))

        return False


    # --------------------------------------------------------
    # Piper
    #
    # IMPORTANT:
    # Use the same file-input approach as your old
    # working implementation.
    # --------------------------------------------------------

    command = [
        "py",
        "-3.10",
        "-m",
        "piper",

        "-m",
        MODEL,

        "-i",
        INPUT_FILE,

        "-f",
        OUTPUT_FILE
    ]


    print()
    print("[PIPER] Running...")
    print("[PIPER COMMAND]:")
    print(command)


    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

    except Exception as error:

        print()
        print(
            "[TTS ERROR] Could not start Piper:"
        )
        print(repr(error))

        return False


    # --------------------------------------------------------
    # Piper output
    # --------------------------------------------------------

    if result.stdout:

        print()
        print("[PIPER STDOUT]")
        print(result.stdout)


    if result.stderr:

        print()
        print("[PIPER STDERR]")
        print(result.stderr)


    # --------------------------------------------------------
    # Check Piper result
    # --------------------------------------------------------

    if result.returncode != 0:

        print()
        print(
            "[TTS ERROR] Piper failed."
        )

        print(
            "[EXIT CODE]:",
            result.returncode
        )

        return False


    # --------------------------------------------------------
    # Check WAV
    # --------------------------------------------------------

    if not os.path.exists(OUTPUT_FILE):

        print()
        print(
            "[TTS ERROR] Piper did not create WAV."
        )

        return False


    file_size = os.path.getsize(
        OUTPUT_FILE
    )

    print()
    print(
        "[TTS] WAV created:",
        file_size,
        "bytes"
    )


    if file_size <= 0:

        print(
            "[TTS ERROR] WAV is empty."
        )

        return False


    # --------------------------------------------------------
    # Play speech
    # --------------------------------------------------------

    try:

        print()
        print(
            "[TTS] Playing Hindi speech..."
        )

        winsound.PlaySound(
            OUTPUT_FILE,
            winsound.SND_FILENAME
        )

        print(
            "[TTS] Playback finished."
        )

    except Exception as error:

        print()
        print(
            "[TTS ERROR] Playback failed:"
        )

        print(repr(error))

        return False


    print()
    print("========================================")
    print("[PIPER TTS] SUCCESS")
    print("========================================")

    return True
