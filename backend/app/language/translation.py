import json
import urllib.request
import urllib.error

import argostranslate.translate


# ============================================================
# QWEN CONFIGURATION
# ============================================================

QWEN_URL = "http://127.0.0.1:8080/v1/chat/completions"

QWEN_MODEL = (
    "DhruvalLabs/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"
)


# ============================================================
# COMMON PHRASE CACHE
# ============================================================

COMMON_PHRASES = {
    "you name what": {
        "english": "What is your name?"
    },

    "you where live": {
        "english": "Where do you live?"
    },

    "i want water": {
        "english": "I want water."
    },

    "i not understand": {
        "english": "I don't understand."
    },

    "you help me": {
        "english": "Can you help me?"
    },

    "hello how are you": {
        "english": "Hello, how are you?"
    },

    "thank you": {
        "english": "Thank you."
    },

    "what time": {
        "english": "What time is it?"
    }
}


# ============================================================
# CLEAN QWEN OUTPUT
# ============================================================

def clean_text(text):
    """
    Cleans the text returned by Qwen.

    Removes:
    - <think>...</think> sections
    - markdown code fences
    - surrounding quotation marks
    """

    if not text:
        return ""

    text = str(text).strip()

    # Remove Qwen thinking section if present
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    # Remove markdown code fences
    text = text.replace("```text", "")
    text = text.replace("```", "")

    text = text.strip()

    # Remove surrounding quotes
    if (
        len(text) >= 2
        and text[0] == '"'
        and text[-1] == '"'
    ):
        text = text[1:-1].strip()

    return text


# ============================================================
# ASK LOCAL QWEN
# ============================================================

def ask_qwen(prompt):
    """
    Sends a prompt to the local Qwen3 server.

    Qwen is running locally through llama.cpp.
    """

    payload = {
        "model": QWEN_MODEL,

        "messages": [
            {
                "role": "system",
                "content": (
                    "You convert recognized sign-language "
                    "keywords into natural English sentences."
                )
            },

            {
                "role": "user",
                "content": prompt
            }
        ],

        "temperature": 0.2,

        "max_tokens": 100
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        QWEN_URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:

        print()
        print("================================")
        print("[QWEN]")
        print("Sending request to local Qwen...")
        print("URL:", QWEN_URL)
        print("Model:", QWEN_MODEL)
        print("================================")

        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            response_data = response.read().decode(
                "utf-8"
            )

        result = json.loads(response_data)

        # OpenAI-compatible response
        choices = result.get("choices", [])

        if not choices:
            print("[QWEN] No choices returned.")
            print("[QWEN] Full response:")
            print(result)
            return ""

        message = choices[0].get(
            "message",
            {}
        )

        content = message.get(
            "content",
            ""
        )

        content = clean_text(content)

        print()
        print("[QWEN] Raw/cleaned result:")
        print(repr(content))
        print("================================")

        return content

    except urllib.error.HTTPError as e:

        print()
        print("================================")
        print("[QWEN HTTP ERROR]")
        print("Status:", e.code)

        try:
            error_body = e.read().decode(
                "utf-8",
                errors="replace"
            )

            print("Response:")
            print(error_body)

        except Exception:
            pass

        print("================================")

        return ""

    except urllib.error.URLError as e:

        print()
        print("================================")
        print("[QWEN URL ERROR]")
        print(str(e))
        print("Make sure llama.cpp/Qwen is running.")
        print("================================")

        return ""

    except TimeoutError:

        print()
        print("================================")
        print("[QWEN TIMEOUT]")
        print("Qwen took too long to respond.")
        print("================================")

        return ""

    except Exception as e:

        print()
        print("================================")
        print("[QWEN ERROR]")
        print("Error type:", type(e).__name__)
        print("Error:", str(e))
        print("================================")

        return ""


# ============================================================
# ARGOS LOCAL TRANSLATION
# ============================================================

def translate_local(
    text,
    target_language="Hindi"
):
    """
    Translates English text using local Argos Translate.

    Currently configured:
        English -> Hindi
    """

    if not text:
        print(
            "[ARGOS] Translation skipped: "
            "empty input."
        )

        return ""

    target_language = (
        str(target_language)
        .lower()
        .strip()
    )

    print()
    print("================================")
    print("[ARGOS]")
    print("Input text:")
    print(repr(text))

    print("Target language:")
    print(repr(target_language))
    print("================================")

    # --------------------------------------------------------
    # HINDI
    # --------------------------------------------------------

    if target_language == "hindi":

        print(
            "[ARGOS] Using "
            "English -> Hindi"
        )

        try:

            translated = (
                argostranslate.translate.translate(
                    text,
                    "en",
                    "hi"
                )
            )

            if translated:
                translated = translated.strip()

            print()
            print("================================")
            print("[ARGOS RESULT]")
            print("Input:")
            print(repr(text))

            print("Output:")
            print(repr(translated))

            print(
                "Output type:",
                type(translated).__name__
            )

            print("================================")

            return translated

        except Exception as e:

            print()
            print("================================")
            print("[ARGOS TRANSLATION ERROR]")
            print(
                "Error type:",
                type(e).__name__
            )
            print(
                "Error:",
                str(e)
            )

            print("Input was:")
            print(repr(text))

            print("Target language:")
            print(repr(target_language))

            print("================================")

            return ""

    # --------------------------------------------------------
    # OTHER LANGUAGES
    # --------------------------------------------------------

    print(
        f"[ARGOS] Translation for "
        f"{target_language} is not configured yet."
    )

    return ""


# ============================================================
# GENERATE NATURAL LANGUAGE RESULT
# ============================================================

def generate_language_result(
    words,
    target_language="Hindi"
):
    """
    Converts recognized sign words into:

        1. Natural English sentence
        2. Target-language translation

    Example:

        ["tomorrow", "you", "come", "college"]

        ↓

        English:
        "Tomorrow you will come to college."

        ↓

        Hindi:
        "कल तुम कॉलेज में आएंगे।"
    """

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not words:

        print(
            "[LANGUAGE] No words received."
        )

        return "", ""

    # --------------------------------------------------------
    # CLEAN WORDS
    # --------------------------------------------------------

    cleaned_words = [
        str(word).strip().lower()
        for word in words
        if word
        and str(word).strip()
    ]

    if not cleaned_words:

        print(
            "[LANGUAGE] No valid words "
            "after cleaning."
        )

        return "", ""

    # --------------------------------------------------------
    # BUILD SIGN TEXT
    # --------------------------------------------------------

    sign_text = " ".join(
        cleaned_words
    )

    print()
    print("================================")
    print("[LANGUAGE PIPELINE]")
    print("Recognized sign keywords:")
    print(sign_text)
    print("Target language:")
    print(target_language)
    print("================================")

    # --------------------------------------------------------
    # CHECK COMMON PHRASE CACHE
    # --------------------------------------------------------

    if sign_text in COMMON_PHRASES:

        print(
            "[LANGUAGE] Using fast "
            "local phrase cache."
        )

        english = COMMON_PHRASES[
            sign_text
        ]["english"]

    # --------------------------------------------------------
    # OTHERWISE USE QWEN
    # --------------------------------------------------------

    else:

        print(
            "[LANGUAGE] Using local "
            "Qwen3-4B..."
        )

        prompt = f"""
Convert the following recognized
sign-language keywords into ONE natural
English sentence.

The input consists of keywords recognized
from Indian Sign Language.

Use the intended meaning and normal
English grammar.

Do not explain the conversion.

Do not list the words.

Do not give multiple alternatives.

Return ONLY one natural English sentence.

Recognized sign keywords:

{sign_text}

Return only the natural English sentence.
"""

        result = ask_qwen(prompt)

        if not result:

            print(
                "[LANGUAGE] Qwen did not "
                "return a result."
            )

            return "", ""

        english = clean_text(result)

    # --------------------------------------------------------
    # CHECK ENGLISH RESULT
    # --------------------------------------------------------

    if not english:

        print(
            "[LANGUAGE] Natural English "
            "sentence is empty."
        )

        return "", ""

    print()
    print("================================")
    print("[LANGUAGE DEBUG]")
    print("Natural English generated:")
    print(repr(english))
    print(
        "English type:",
        type(english).__name__
    )
    print("================================")

    # ========================================================
    # CRITICAL DEBUG POINT
    # ========================================================
    #
    # This is the exact value being sent
    # from Qwen -> Argos.
    #
    # ========================================================

    print()
    print("================================")
    print("[LANGUAGE DEBUG]")
    print("English being sent to Argos:")
    print(repr(english))

    print("Target language:")
    print(repr(target_language))

    print("================================")

    # --------------------------------------------------------
    # TRANSLATE ENGLISH -> TARGET LANGUAGE
    # --------------------------------------------------------

    translated = translate_local(
        english,
        target_language
    )

    # ========================================================
    # CRITICAL DEBUG POINT #2
    # ========================================================

    print()
    print("================================")
    print("[LANGUAGE DEBUG]")
    print("Argos returned:")
    print(repr(translated))

    print(
        "Translation type:",
        type(translated).__name__
    )

    print("================================")

    # --------------------------------------------------------
    # TRANSLATION FAILURE
    # --------------------------------------------------------

    if not translated:

        print(
            "[LANGUAGE] Translation failed."
        )

        print(
            "[LANGUAGE] Returning English "
            "sentence with empty translation."
        )

        return english, ""

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print()
    print("================================")
    print("[LANGUAGE FINAL RESULT]")
    print("English:")
    print(english)

    print()
    print("Translated:")
    print(translated)

    print("================================")

    return english, translated


# ============================================================
# SIMPLE TRANSLATE FUNCTION
# ============================================================

def translate(
    text,
    target_language="Hindi"
):
    """
    Direct translation helper.

    Example:

        translate(
            "Please pay attention.",
            "Hindi"
        )
    """

    if not text:
        return ""

    return translate_local(
        text,
        target_language
    )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("================================")
    print("SIGNFLOW TRANSLATION TEST")
    print("================================")

    test_words = [
        "tomorrow",
        "you",
        "come",
        "college"
    ]

    english, translated = (
        generate_language_result(
            test_words,
            "Hindi"
        )
    )

    print()
    print("================================")
    print("[TEST RESULT]")
    print("English:")
    print(repr(english))

    print()
    print("Hindi:")
    print(repr(translated))

    print("================================")