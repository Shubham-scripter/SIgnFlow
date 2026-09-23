import subprocess
import os
import winsound


MODEL = os.path.join(
    os.path.dirname(__file__),
    "hi_IN-pratham-medium.onnx"
)


def text_to_speech(text):
    output_file = os.path.join(
        os.path.dirname(__file__),
        "speech_output.wav"
    )

    input_file = os.path.join(
        os.path.dirname(__file__),
        "tts_input.txt"
    )

    try:
        # Save text to a file
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(text)

        # Run Piper
        command = [
            "py",
            "-m",
            "piper",
            "-m",
            MODEL,
            "-i",
            input_file,
            "-f",
            output_file
        ]

        subprocess.run(command, check=True)

        # Play generated speech
        if os.path.exists(output_file):
            winsound.PlaySound(
                output_file,
                winsound.SND_FILENAME
            )

    except Exception as e:
        print("TTS error:", e)


if __name__ == "__main__":
    text_to_speech("नमस्कार, आज आप कैसे हैं?")