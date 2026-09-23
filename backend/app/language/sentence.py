from translation import translate
from text_to_speech import text_to_speech


def create_sentence(words):
    sentence = " ".join(words)
    return sentence.capitalize()


if __name__ == "__main__":
    words = ["hello"]

    sentence = create_sentence(words)
    print("Sentence:", sentence)

    translated = translate(sentence, "Hindi")
    print("Translated:", translated)

    text_to_speech(translated)