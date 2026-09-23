from translation import translate
from text_to_speech import text_to_speech


def create_sentence(words):
    sentence = " ".join(words)
    return sentence.capitalize()


if __name__ == "__main__":
    # Simulated speech-to-text result
    words = ["hello", "how", "are", "you", "today"]

    # Create sentence
    sentence = create_sentence(words)
    print("Original:", sentence)

    # Translate English -> Hindi
    translated = translate(sentence, "Hindi")
    print("Translated:", translated)

    # Speak Hindi
    text_to_speech(translated)