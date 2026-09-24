class WordStream:

    def __init__(self, stable_frames=5):

        self.stable_frames = stable_frames

        self.current_word = None
        self.frame_count = 0

        self.last_added_word = None

        self.words = []

        # True when the user has released
        # the previous sign.
        self.ready_for_new_word = True


    # ========================================================
    # PROCESS RECOGNIZED WORD
    # ========================================================

    def process(self, word):

        # ----------------------------------------------------
        # No hand / unknown prediction
        # ----------------------------------------------------

        if (
            word is None
            or word == "Unknown"
            or word == "No hand detected"
        ):

            self.current_word = None
            self.frame_count = 0

            # User has released the previous sign.
            self.ready_for_new_word = True

            return None


        # ----------------------------------------------------
        # Same word as previous frame
        # ----------------------------------------------------

        if word == self.current_word:

            self.frame_count += 1

        else:

            # A different sign has appeared.
            self.current_word = word
            self.frame_count = 1


        # ----------------------------------------------------
        # Wait until the sign is stable
        # ----------------------------------------------------

        if self.frame_count < self.stable_frames:

            return None


        # ----------------------------------------------------
        # Prevent repeated words while holding a sign
        # ----------------------------------------------------

        if (
            word == self.last_added_word
            and
            not self.ready_for_new_word
        ):

            return None


        # ----------------------------------------------------
        # Add the new word
        # ----------------------------------------------------

        self.words.append(word)

        self.last_added_word = word

        self.ready_for_new_word = False

        # Reset current detection
        self.current_word = None
        self.frame_count = 0

        return word


    # ========================================================
    # REMOVE LAST WORD
    # ========================================================

    def remove_last_word(self):

        if not self.words:

            return None


        removed_word = self.words.pop()


        # Update the last word
        if self.words:

            self.last_added_word = self.words[-1]

        else:

            self.last_added_word = None


        # Allow a new word to be detected.
        self.ready_for_new_word = True


        return removed_word


    # ========================================================
    # GET ALL WORDS
    # ========================================================

    def get_words(self):

        return self.words.copy()


    # ========================================================
    # GET CURRENT SENTENCE
    # ========================================================

    def get_sentence(self):

        return " ".join(
            self.words
        ).strip()


    # ========================================================
    # CLEAR EVERYTHING
    # ========================================================

    def clear(self):

        self.words.clear()

        self.current_word = None
        self.frame_count = 0

        self.last_added_word = None

        self.ready_for_new_word = True


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    stream = WordStream(
        stable_frames=3
    )


    print()
    print("Testing WordStream...")
    print()


    # --------------------------------------------------------
    # Simulate holding "hello"
    # --------------------------------------------------------

    test_sequence = [

        "hello",
        "hello",
        "hello",
        "hello",
        "hello",

        None,

        "help",
        "help",
        "help",
        "help",

        None,

        "yes",
        "yes",
        "yes",
        "yes"

    ]


    for word in test_sequence:

        new_word = stream.process(
            word
        )


        if new_word:

            print(
                "New word:",
                new_word
            )


    print()

    print(
        "Words:",
        stream.get_words()
    )

    print(
        "Sentence:",
        stream.get_sentence()
    )


    # --------------------------------------------------------
    # Test backspace
    # --------------------------------------------------------

    removed = stream.remove_last_word()


    print()

    print(
        "Removed:",
        removed
    )

    print(
        "Words after removing:",
        stream.get_words()
    )

    print(
        "Sentence after removing:",
        stream.get_sentence()
    )