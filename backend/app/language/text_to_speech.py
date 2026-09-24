import os
import base64
import winsound
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_TTS_MODEL = os.getenv(
    "GEMINI_TTS_MODEL",
    "gemini-3.8-flash-lite-tts"
)

client = genai.Client(api_key=GEMINI_API_KEY)

OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__),
    "speech_output.wav"
)


def text_to_speech(text):
    print("\nGenerating natural speech...")
    print("Text:", text)

    try:
        response = client.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": text,
                            "speech_metadata": {
                                "style": (
                                    "Natural, clear and friendly "
                                    "conversational speech. "
                                    "Speak at a comfortable pace "
                                    "with natural pronunciation."
                                )
                            }
                        }
                    ]
                }
            ],
            config={
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "voice": "Kore"
                    }
                }
            }
        )

        audio_data = response.candidates[0].content.parts[0].inline_data.data

        with open(OUTPUT_FILE, "wb") as f:
            f.write(audio_data)

        print("Speech generated:", OUTPUT_FILE)

        winsound.PlaySound(
            OUTPUT_FILE,
            winsound.SND_FILENAME
        )

    except Exception as e:
        print("Gemini TTS error:", e)


if __name__ == "__main__":
    text_to_speech(
        "Hello, how are you today?"
    )