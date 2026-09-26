import threading
import queue
import traceback

from translation import generate_language_result
from text_to_speech import text_to_speech


class LanguageWorker:
    """
    Runs Qwen -> Argos -> TTS away from the camera/UI loop.

    The camera thread only submits a sentence and immediately continues.
    """

    def __init__(self, target_language="Hindi"):
        self.target_language = target_language

        self._queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.english = ""
        self.translation = ""
        self.tts_status = ""
        self.status = "IDLE"
        self.error = ""

        self._worker = threading.Thread(
            target=self._run,
            name="SignFlow-LanguageWorker",
            daemon=True,
        )
        self._worker.start()

    def submit(self, words):
        """
        Submit a copy of the current sentence.

        Returns True if accepted.

        If a previous language job is already waiting in the queue,
        the newest sentence replaces that waiting job.
        """
        if not words:
            return False

        sentence = [
            str(word).strip()
            for word in words
            if word and str(word).strip()
        ]

        if not sentence:
            return False

        # Only one waiting job is needed. Never allow the camera
        # loop to block behind old language requests.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

        self._queue.put_nowait(sentence)

        with self._lock:
            self.status = "PROCESSING"
            self.error = ""

        print()
        print("[LANGUAGE WORKER] Submitted:")
        print(" ".join(sentence))

        return True

    def get_state(self):
        """Return a thread-safe snapshot for the UI."""
        with self._lock:
            return {
                "english": self.english,
                "translation": self.translation,
                "tts_status": self.tts_status,
                "status": self.status,
                "error": self.error,
            }

    def _run(self):
        while not self._stop_event.is_set():
            try:
                words = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._process(words)
            except Exception as error:
                print("[LANGUAGE WORKER ERROR]", repr(error))
                traceback.print_exc()

                with self._lock:
                    self.status = "ERROR"
                    self.error = repr(error)
                    self.tts_status = "TTS ERROR"

    def _process(self, words):
        print()
        print("========================================")
        print("       BACKGROUND LANGUAGE PROCESSING")
        print("========================================")
        print("[RAW SIGNS]", words)

        try:
            english, translated = generate_language_result(
                words,
                self.target_language,
            )
        except Exception as error:
            with self._lock:
                self.status = "ERROR"
                self.error = repr(error)
                self.tts_status = "LANGUAGE ERROR"

            print("[LANGUAGE ERROR]", repr(error))
            return

        with self._lock:
            self.english = english or ""
            self.translation = translated or ""
            self.tts_status = ""
            self.error = ""

        if english:
            print("[ENGLISH]", english)
        else:
            print("[LANGUAGE] Qwen returned empty English.")

        if translated:
            print("[HINDI]", translated)
        else:
            print("[LANGUAGE] Argos returned empty translation.")

        # TTS is also performed in this worker, so the camera
        # never waits for speech generation or playback.
        if translated:
            with self._lock:
                self.tts_status = "TTS..."

            try:
                print("[TTS] Starting background TTS...")
                result = text_to_speech(translated)
                print("[TTS] text_to_speech returned:", repr(result))

                with self._lock:
                    self.tts_status = "TTS ✓"
            except Exception as error:
                print("[TTS ERROR]", repr(error))

                with self._lock:
                    self.tts_status = "TTS ERROR"
                    self.error = repr(error)
        else:
            with self._lock:
                self.tts_status = ""

        with self._lock:
            self.status = "DONE"

        print("========================================")
        print("[LANGUAGE WORKER] Finished")
        print("========================================")

    def stop(self):
        """Stop the background worker."""
        self._stop_event.set()
        self._worker.join(timeout=1.0)
