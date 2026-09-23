import argostranslate.translate


def translate(text, target_language="Hindi"):
    language_codes = {
        "English": "en",
        "Hindi": "hi"
    }

    target_code = language_codes.get(target_language)

    if not target_code:
        return text

    try:
        installed_languages = argostranslate.translate.get_installed_languages()

        from_language = next(
            lang for lang in installed_languages
            if lang.code == "en"
        )

        to_language = next(
            lang for lang in installed_languages
            if lang.code == target_code
        )

        translation = from_language.get_translation(to_language)

        return translation.translate(text)

    except Exception as e:
        print("Translation error:", e)
        return text


if __name__ == "__main__":
    result = translate("Hello, how are you today?", "Hindi")

    print("Original:", "Hello, how are you today?")
    print("Translated:", result)