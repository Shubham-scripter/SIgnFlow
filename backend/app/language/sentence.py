try:
    from .translation import generate_language_result
    from .text_to_speech import text_to_speech
except ImportError:
    from translation import generate_language_result
    from text_to_speech import text_to_speech


def process_text(words, target_language="Hindi"):

    if not words:
        print("No words received.")
        return ""

    print()
    print("=" * 50)
    print("SIGNFLOW LANGUAGE PIPELINE")
    print("=" * 50)

    # --------------------------------------------------------
    # STEP 1 — RECOGNIZED SIGNS
    # --------------------------------------------------------

    print(
        "Recognized signs:",
        " ".join(words)
    )

    # --------------------------------------------------------
    # STEP 2 — LOCAL QWEN / PHRASE CACHE
    # --------------------------------------------------------

    english, translated = generate_language_result(
        words,
        target_language
    )

    if not english:

        print("Could not generate sentence.")
        return ""

    print(
        "Natural sentence:",
        english
    )

    print(
        "Translated:",
        translated
    )

    # --------------------------------------------------------
    # STEP 3 — TEXT TO SPEECH
    # --------------------------------------------------------

    if translated:

        print("Speaking...")

        try:

            text_to_speech(
                translated
            )

        except Exception as e:

            print(
                "TTS error:",
                e
            )

    print("=" * 50)

    return translated


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_words = [
    "tomorrow",
    "you",
    "come",
    "college"
]

    result = process_text(
        test_words,
        "Hindi"
    )

    print()
    print("Final result:", result)