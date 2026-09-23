from translation import translate
from text_to_speech import text_to_speech


def create_sentence(words):
    """
    Convert a list of words into a proper sentence.
    """
    sentence = " ".join(words).strip()

    if not sentence:
        return ""

    return sentence.capitalize()


def process_text(words, target_language="Hindi"):
    """
    Complete language pipeline:

    Words
      ↓
    Sentence
      ↓
    Translation
      ↓
    Text-to-Speech
    """

    # Step 1: Create sentence
    sentence = create_sentence(words)

    if not sentence:
        print("No words received.")
        return

    print("Original:", sentence)

    # Step 2: Translate
    translated = translate(sentence, target_language)

    print("Translated:", translated)

    # Step 3: Speak translated sentence
    text_to_speech(translated)

    return translated


if __name__ == "__main__":

    # Simulated output from Speech-to-Text
    words = [
        "hello",
        "how",
        "are",
        "you",
        "today"
    ]

    process_text(words, "Hindi")