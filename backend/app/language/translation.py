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

    if not text:
        return ""

    text = text.strip()

    # Remove Qwen thinking section if present.
    if "</think>" in text:

        text = text.split(
            "</think>",
            1
        )[1].strip()

    # Remove accidental markdown fences.
    text = text.replace(
        "```text",
        ""
    )

    text = text.replace(
        "```",
        ""
    )

    return text.strip()


# ============================================================
# ASK LOCAL QWEN
# ============================================================

def ask_qwen(prompt):

    payload = {

        "model": QWEN_MODEL,

        "messages": [

            {
                "role": "system",

                "content": (
                    "You are the language engine "
                    "of SignFlow, a sign-language "
                    "translation system."
                )
            },

            {
                "role": "user",

                "content": prompt
            }

        ],

        "temperature": 0.2,

        "max_tokens": 100,

        "stream": False

    }


    data = json.dumps(
        payload,
        ensure_ascii=False
    ).encode(
        "utf-8"
    )


    request = urllib.request.Request(

        QWEN_URL,

        data=data,

        headers={
            "Content-Type": "application/json"
        },

        method="POST"

    )


    try:

        with urllib.request.urlopen(
            request,
            timeout=30
        ) as response:

            result = json.loads(
                response.read().decode(
                    "utf-8"
                )
            )


        return result[
            "choices"
        ][
            0
        ][
            "message"
        ][
            "content"
        ]


    except urllib.error.URLError as e:

        print(
            "Qwen connection error:",
            e
        )

        return ""


    except Exception as e:

        print(
            "Qwen error:",
            e
        )

        return ""


# ============================================================
# LOCAL TRANSLATION
# ============================================================

def translate_local(
    text,
    target_language="Hindi"
):

    if not text:

        return ""


    target_language = (
        target_language.lower()
    )


    # --------------------------------------------------------
    # English -> Hindi
    # --------------------------------------------------------

    if target_language == "hindi":

        try:

            translated = (
                argostranslate.translate.translate(
                    text,
                    "en",
                    "hi"
                )
            )

            return translated.strip()


        except Exception as e:

            print(
                "Argos translation error:",
                e
            )

            return ""


    # --------------------------------------------------------
    # Other languages
    # --------------------------------------------------------

    print(
        "Argos translation for "
        f"{target_language} is not configured yet."
    )

    return ""


# ============================================================
# GENERATE NATURAL SENTENCE + TRANSLATION
# ============================================================

def generate_language_result(
    words,
    target_language="Hindi"
):

    if not words:

        return "", ""


    # --------------------------------------------------------
    # Clean recognized words
    # --------------------------------------------------------

    cleaned_words = [

        word.strip().lower()

        for word in words

        if word and word.strip()

    ]


    if not cleaned_words:

        return "", ""


    sign_text = " ".join(
        cleaned_words
    )


    # ========================================================
    # FAST COMMON PHRASE CACHE
    # ========================================================

    if sign_text in COMMON_PHRASES:

        print(
            "Using fast local phrase cache."
        )

        english = (
            COMMON_PHRASES[
                sign_text
            ][
                "english"
            ]
        )


    # ========================================================
    # QWEN SENTENCE GENERATION
    # ========================================================

    else:

        print(
            "Using local Qwen3-4B..."
        )


        prompt = f"""
Convert the following recognized
sign-language keywords into ONE natural
English sentence.

The input may use sign-language word order.

Rules:
- Preserve the intended meaning.
- Correct the grammar.
- Correct unnatural word order.
- Add only necessary grammatical words.
- Do not invent information.
- Do not remove important information.
- Preserve time expressions exactly.
- Preserve names, numbers and places.
- Return ONLY the English sentence.

Examples:

YOU NAME WHAT
-> What is your name?

YOU WHERE LIVE
-> Where do you live?

I WANT WATER
-> I want water.

I NOT UNDERSTAND
-> I don't understand.

YOU HELP ME
-> Can you help me?

WHAT TIME
-> What time is it?

Recognized sign keywords:

{sign_text}

Return only the natural English sentence.
"""


        result = ask_qwen(
            prompt
        )


        if not result:

            print(
                "Qwen did not return a result."
            )

            return "", ""


        english = clean_text(
            result
        )


    # ========================================================
    # LOCAL TRANSLATION
    # ========================================================

    print(
        "Using local Argos Translate ->",
        target_language
    )


    translated = translate_local(
        english,
        target_language
    )


    if not translated:

        print(
            "Translation failed."
        )

        # Return English even if translation
        # fails so the rest of SignFlow can
        # still see the generated sentence.

        return english, ""


    return english, translated


# ============================================================
# SIMPLE TRANSLATE FUNCTION
# ============================================================

def translate(
    text,
    target_language="Hindi"
):

    if not text:

        return ""


    return translate_local(
        text,
        target_language
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

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

    print(
        "English:",
        english
    )

    print(
        "Translated:",
        translated
    )