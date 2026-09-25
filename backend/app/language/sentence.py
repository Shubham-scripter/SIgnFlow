import traceback


# ============================================================
# IMPORT LANGUAGE MODULES
# ============================================================

try:
    from .translation import generate_language_result
    from .text_to_speech import text_to_speech

except ImportError:
    from translation import generate_language_result
    from text_to_speech import text_to_speech


# ============================================================
# SIGNFLOW LANGUAGE PIPELINE
# ============================================================

def process_text(words, target_language="Hindi"):

    # --------------------------------------------------------
    # No words
    # --------------------------------------------------------

    if not words:
        print("[SENTENCE] No words received.")
        return "", ""

    print()
    print("=" * 70)
    print("SIGNFLOW LANGUAGE PIPELINE STARTED")
    print("=" * 70)

    # ========================================================
    # STEP 1 — RECOGNIZED SIGNS
    # ========================================================

    sign_text = " ".join(
        str(word).strip()
        for word in words
        if word
    )

    print("[SENTENCE] Recognized signs:")
    print("[SENTENCE]", sign_text)

    print()
    print("[SENTENCE] Target language:")
    print("[SENTENCE]", target_language)

    # ========================================================
    # STEP 2 — QWEN + ARGOS
    # ========================================================

    print()
    print("[SENTENCE] Calling generate_language_result()...")
    print()

    try:

        english, translated = generate_language_result(
            words,
            target_language
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("[SENTENCE] LANGUAGE GENERATION ERROR")
        print("=" * 70)

        print(
            "[SENTENCE] Error type:",
            type(e).__name__
        )

        print(
            "[SENTENCE] Error:",
            str(e)
        )

        print()
        print("[SENTENCE] Full traceback:")

        traceback.print_exc()

        print("=" * 70)
        print()

        return "", ""

    # ========================================================
    # DEBUG — CHECK RETURN VALUES
    # ========================================================

    print()
    print("=" * 70)
    print("[SENTENCE] LANGUAGE RESULT RECEIVED")
    print("=" * 70)

    print(
        "[SENTENCE] English returned:",
        repr(english)
    )

    print(
        "[SENTENCE] Translation returned:",
        repr(translated)
    )

    print("=" * 70)

    # ========================================================
    # CHECK ENGLISH
    # ========================================================

    if not english:

        print()
        print(
            "[SENTENCE] ERROR: English sentence is empty."
        )

        print(
            "[SENTENCE] Cannot continue to TTS."
        )

        return "", ""

    print()
    print(
        "[SENTENCE] Natural English sentence:"
    )

    print(
        "[SENTENCE]",
        english
    )

    # ========================================================
    # CHECK TRANSLATION
    # ========================================================

    if translated:

        print()
        print(
            "[SENTENCE] Translation:"
        )

        print(
            "[SENTENCE]",
            translated
        )

    else:

        print()
        print(
            "[SENTENCE] WARNING: Translation is empty."
        )

        print(
            "[SENTENCE] TTS will NOT start."
        )

        print(
            "[SENTENCE] Returning English sentence only."
        )

        return english, ""

    # ========================================================
    # STEP 3 — TEXT TO SPEECH
    # ========================================================

    print()
    print("=" * 70)
    print("[SENTENCE] STARTING TTS")
    print("=" * 70)

    print(
        "[SENTENCE] TTS function:",
        text_to_speech
    )

    print(
        "[SENTENCE] Text being sent to TTS:"
    )

    print(
        "[SENTENCE]",
        repr(translated)
    )

    print("=" * 70)

    # ========================================================
    # CALL TTS
    # ========================================================

    try:

        print()
        print(
            "[SENTENCE] Calling text_to_speech()..."
        )

        text_to_speech(
            translated
        )

        print()
        print("=" * 70)
        print("[SENTENCE] TTS COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:

        print()
        print("=" * 70)
        print("[SENTENCE] TTS ERROR")
        print("=" * 70)

        print(
            "[SENTENCE] Error type:",
            type(e).__name__
        )

        print(
            "[SENTENCE] Error:",
            str(e)
        )

        print()
        print("[SENTENCE] TTS traceback:")

        traceback.print_exc()

        print("=" * 70)

        print()
        print(
            "[SENTENCE] Translation is still available."
        )

    # ========================================================
    # STEP 4 — FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("SIGNFLOW LANGUAGE PIPELINE FINISHED")
    print("=" * 70)

    print(
        "[SENTENCE] Final English:",
        repr(english)
    )

    print(
        "[SENTENCE] Final Translation:",
        repr(translated)
    )

    print("=" * 70)
    print()

    # IMPORTANT:
    # Return BOTH values.
    #
    # english     -> natural English sentence
    # translated  -> target language
    #
    return english, translated


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SIGNFLOW SENTENCE.PY STANDALONE TEST")
    print("=" * 70)
    print()

    test_words = [
        "tomorrow",
        "you",
        "come",
        "college"
    ]

    english, translated = process_text(
        test_words,
        "Hindi"
    )

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        "English:",
        repr(english)
    )

    print(
        "Hindi:",
        repr(translated)
    )

    print("=" * 70)