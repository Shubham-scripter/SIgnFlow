def translate(text, target_language="Hindi"):
    translations = {
        "hello": {
            "Hindi": "नमस्ते"
        },
        "how are you": {
            "Hindi": "आप कैसे हैं?"
        },
        "thank you": {
            "Hindi": "धन्यवाद"
        }
    }

    text = text.lower().strip()

    if text in translations:
        return translations[text].get(target_language, text)

    return text


if __name__ == "__main__":
    result = translate("hello", "Hindi")
    print("Translated:", result)