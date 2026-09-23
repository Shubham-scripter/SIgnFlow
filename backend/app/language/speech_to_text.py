import sounddevice as sd
from scipy.io.wavfile import write
import whisper


# Load Whisper model
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model loaded!")


def record_audio(filename="recording.wav", duration=5, sample_rate=16000):
    print("\nRecording... Speak now!")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        device=53
    )

    sd.wait()

    write(filename, sample_rate, audio)

    print("Recording saved:", filename)


def speech_to_text(audio_file="recording.wav"):
    print("Converting speech to text...")

    result = model.transcribe(
        audio_file,
        language="en"
    )

    return result["text"].strip()


if __name__ == "__main__":
    record_audio()

    text = speech_to_text()

    print("\nRecognized:")
    print(text)