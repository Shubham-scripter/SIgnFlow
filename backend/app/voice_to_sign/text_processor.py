import re


def clean_text(text):
    """
    Clean and normalize spoken English text.

    Examples:
        "What is your name?"
            -> "what is your name"

        "  HELLO!!! "
            -> "hello"

    Args:
        text (str): Raw speech-to-text result.

    Returns:
        str: Cleaned lowercase text.
    """

    if not text:
        return ""

    # Convert to string and remove surrounding whitespace
    text = str(text).strip()

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Replace multiple spaces with one
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def process_text(text):
    """
    Process raw speech recognition output.

    Args:
        text (str): Raw speech-to-text result.

    Returns:
        str: Normalized text.
    """

    cleaned = clean_text(text)

    if not cleaned:
        return ""

    print("[TEXT PROCESSOR]")
    print("[RAW TEXT]    ", text)
    print("[CLEAN TEXT]  ", cleaned)

    return cleaned


if __name__ == "__main__":

    print("=" * 40)
    print("       TEXT PROCESSOR TEST")
    print("=" * 40)

    test_text = "What is your name?"

    result = process_text(test_text)

    print()
    print("[RESULT]", result)