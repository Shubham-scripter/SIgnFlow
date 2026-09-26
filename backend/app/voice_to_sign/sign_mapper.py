"""
English text -> ISL sign sequence mapper.

This module converts cleaned English sentences into
sequences of signs supported by the current ISL vocabulary.
"""


# ============================================================
# SUPPORTED ISL VOCABULARY
# ============================================================

SUPPORTED_SIGNS = {
    "a",
    "b",
    "c",
    "correct",
    "d",
    "e",
    "hello",
    "help",
    "hug",
    "i love you",
    "namaste",
    "name",
    "no",
    "pay attention",
    "please",
    "sorry",
    "what",
    "where",
    "yes",
    "you",
}


# ============================================================
# PHRASE MAPPINGS
# ============================================================

PHRASE_MAPPINGS = {
    # Greetings
    "hello": ["hello"],
    "hi": ["hello"],
    "namaste": ["namaste"],

    # Questions
    "what is your name": ["you", "name", "what"],
    "what your name": ["you", "name", "what"],
    "your name what": ["you", "name", "what"],

    "where are you": ["where", "you"],
    "where you": ["where", "you"],

    # Help
    "can you help": ["you", "help"],
    "can you help me": ["you", "help"],
    "please help": ["please", "help"],
    "help me": ["help"],

    # Politeness
    "please": ["please"],
    "sorry": ["sorry"],

    # Yes / No
    "yes": ["yes"],
    "yeah": ["yes"],
    "no": ["no"],
    "nope": ["no"],

    # Attention
    "pay attention": ["pay attention"],
    "please pay attention": ["please", "pay attention"],

    # Love
    "i love you": ["i love you"],

    # Hug
    "hug": ["hug"],
}


# ============================================================
# MAP TEXT TO SIGNS
# ============================================================

def map_text_to_signs(text):
    """
    Convert cleaned English text into an ISL sign sequence.

    Args:
        text (str): Cleaned English text.

    Returns:
        list[str]: Sequence of supported ISL signs.
    """

    if not text:
        return []

    text = text.strip().lower()

    # --------------------------------------------------------
    # Direct phrase match
    # --------------------------------------------------------

    if text in PHRASE_MAPPINGS:
        signs = PHRASE_MAPPINGS[text]

        print("[SIGN MAPPER]")
        print("[INPUT] ", text)
        print("[SIGNS] ", signs)

        return signs.copy()

    # --------------------------------------------------------
    # Basic word-level fallback
    # --------------------------------------------------------

    words = text.split()
    signs = []

    for word in words:

        if word in SUPPORTED_SIGNS:
            signs.append(word)

    # --------------------------------------------------------
    # Remove duplicate consecutive signs
    # --------------------------------------------------------

    cleaned_signs = []

    for sign in signs:
        if not cleaned_signs or cleaned_signs[-1] != sign:
            cleaned_signs.append(sign)

    print("[SIGN MAPPER]")
    print("[INPUT] ", text)
    print("[SIGNS] ", cleaned_signs)

    return cleaned_signs


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 40)
    print("          SIGN MAPPER TEST")
    print("=" * 40)

    test_phrases = [
        "hello",
        "what is your name",
        "where are you",
        "can you help me",
        "please",
        "sorry",
        "yes",
        "no",
        "i love you",
        "please pay attention",
    ]

    for phrase in test_phrases:

        print()
        print("Input:", phrase)

        signs = map_text_to_signs(phrase)

        print("Output:", signs)

    print()
    print("=" * 40)