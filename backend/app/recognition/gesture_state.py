from collections import Counter, deque


class GestureState:
    """
    Converts continuous sign predictions into one committed word at a time.

    Important difference from the old realtime loop:
    - The hand does NOT need to leave the camera frame.
    - A sign is committed once it is stable for STABLE_REQUIRED frames.
    - Holding the same sign does not repeatedly add the same word.
    - A different stable sign can be committed immediately after it appears.
    """

    def __init__(self, history_size=8, stable_required=5):
        self.history_size = history_size
        self.stable_required = stable_required

        self.history = deque(maxlen=history_size)

        # Last sign that was actually added to the sentence.
        self.last_committed = None

        # Current stable sign shown by the recognizer.
        self.current_stable = None

    def update(self, prediction, valid_prediction=True):
        """
        Feed one recognition result per camera frame.

        Returns:
            committed_word: str | None

        Example:
            YOU -> None -> None -> YOU committed
            YOU held -> None
            NAME -> None -> NAME committed
        """

        if not valid_prediction or not prediction:
            self.history.clear()
            self.current_stable = None
            return None

        prediction = str(prediction).strip().lower()

        if not prediction:
            return None

        self.history.append(prediction)

        if len(self.history) < self.stable_required:
            return None

        candidate, count = Counter(self.history).most_common(1)[0]

        if count < self.stable_required:
            return None

        self.current_stable = candidate

        # Same sign is still being held.
        # Do not add it again.
        if candidate == self.last_committed:
            return None

        # A NEW stable sign has appeared.
        self.last_committed = candidate

        # Clear old history so the next gesture gets its own
        # stability window.
        self.history.clear()

        return candidate

    def reset(self):
        """Reset the state completely."""
        self.history.clear()
        self.last_committed = None
        self.current_stable = None

    def undo_commit(self, new_last_word=None):
        """
        Used after BACKSPACE.

        Example:
            sentence = [you, name, what]
            BACKSPACE removes what
            undo_commit("name")
        """
        self.history.clear()
        self.current_stable = None
        self.last_committed = new_last_word

    def force_rearm(self):
        """
        Allows the next occurrence of the same sign to be accepted.

        This is useful if the UI later adds a dedicated word-boundary
        gesture/button.
        """
        self.history.clear()
        self.current_stable = None
        self.last_committed = None
