import os
import sys
import time
import json
import queue
import threading
import traceback

import cv2
import joblib
import numpy as np
import mediapipe as mp

try:
    import websocket
except ImportError:
    websocket = None

try:
    import pyvirtualcam
except ImportError:
    pyvirtualcam = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RECOGNITION_DIR = CURRENT_DIR

BACKEND_DIR = os.path.abspath(
    os.path.join(RECOGNITION_DIR, "..", "..")
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(BACKEND_DIR, "..")
)

MODELS_DIR = os.path.join(
    BACKEND_DIR,
    "models",
    "isl"
)

HAND_LANDMARKER_PATH = os.path.join(
    RECOGNITION_DIR,
    "hand_landmarker.task"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "isl_static_model.pkl"
)


# ============================================================
# IMPORT WORD STREAM
# ============================================================

try:
    from .word_stream import WordStream
except ImportError:
    from word_stream import WordStream


# ============================================================
# IMPORT LANGUAGE SYSTEM
# ============================================================

try:
    from ..language.translation import generate_language_result
    from ..language.text_to_speech import text_to_speech
except ImportError:
    try:
        from backend.app.language.translation import generate_language_result
        from backend.app.language.text_to_speech import text_to_speech
    except ImportError:
        from translation import generate_language_result
        from text_to_speech import text_to_speech


# ============================================================
# SETTINGS
# ============================================================

TARGET_LANGUAGE = "Hindi"

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360

CAMERA_FPS = 30

# How frequently frames are sent to backend/frontend
WEBSOCKET_SEND_INTERVAL = 0.10

# JPEG quality for websocket preview
WEBSOCKET_JPEG_QUALITY = 70


# ============================================================
# AUTOMATIC LANGUAGE SETTINGS
# ============================================================

# After a new word is detected, wait this long before asking
# Qwen to regenerate the sentence.
#
# This prevents Qwen/Argos from running on every frame.
LIVE_TRANSLATION_DEBOUNCE = 1.2


# If no new sign is detected for this amount of time,
# consider the sentence finished.
#
# Increase this if the signer naturally pauses between words.
SENTENCE_IDLE_SECONDS = 4.0


# Keep the completed subtitle visible before fading.
SUBTITLE_HOLD_SECONDS = 3.0


# Fade duration.
SUBTITLE_FADE_SECONDS = 1.5


# Automatically speak the FINAL translated sentence.
AUTO_SPEAK_FINAL_SENTENCE = True


# ============================================================
# SUBTITLE SETTINGS
# ============================================================

SUBTITLE_WIDTH_RATIO = 0.86

SUBTITLE_BOTTOM_MARGIN = 35

SUBTITLE_SIDE_PADDING = 35

SUBTITLE_TOP_PADDING = 22

SUBTITLE_BOTTOM_PADDING = 22

SUBTITLE_RADIUS = 28

ENGLISH_FONT_SIZE = 34

TRANSLATION_FONT_SIZE = 30

MIN_FONT_SIZE = 20

LINE_SPACING = 10


# ============================================================
# VIRTUAL CAMERA SETTINGS
# ============================================================

VIRTUAL_CAMERA_WIDTH = 1280
VIRTUAL_CAMERA_HEIGHT = 720
VIRTUAL_CAMERA_FPS = 30

VIRTUAL_CAMERA_BACKEND = "unitycapture"


# ============================================================
# WEBSOCKET
# ============================================================

WEBSOCKET_URL = "ws://127.0.0.1:8000/ws"


# ============================================================
# MEDIAPIPE
# ============================================================

MP_TASKS = mp.tasks
MP_VISION = MP_TASKS.vision

BaseOptions = MP_TASKS.BaseOptions
HandLandmarker = MP_VISION.HandLandmarker
HandLandmarkerOptions = MP_VISION.HandLandmarkerOptions
VisionRunningMode = MP_VISION.RunningMode


# ============================================================
# FONT FINDER
# ============================================================

def find_font():
    """
    Find a Windows font capable of rendering English + Hindi.

    Nirmala UI is preferred because it supports Devanagari.
    """

    if ImageFont is None:
        return None

    candidates = [
        r"C:\Windows\Fonts\Nirmala.ttf",
        r"C:\Windows\Fonts\NirmalaUI.ttf",
        r"C:\Windows\Fonts\mangal.ttf",
        r"C:\Windows\Fonts\Mangal.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]

    for path in candidates:
        if os.path.exists(path):
            print("[SUBTITLE] Using font:", path)
            return path

    print("[SUBTITLE] No suitable Windows font found.")

    return None


FONT_PATH = find_font()


# ============================================================
# FONT CACHE
# ============================================================

FONT_CACHE = {}


def get_font(size):
    if ImageFont is None:
        return None

    key = int(size)

    if key in FONT_CACHE:
        return FONT_CACHE[key]

    if FONT_PATH is not None:
        try:
            font = ImageFont.truetype(
                FONT_PATH,
                key
            )

            FONT_CACHE[key] = font

            return font

        except Exception:
            pass

    try:
        font = ImageFont.load_default()

        FONT_CACHE[key] = font

        return font

    except Exception:
        return None


# ============================================================
# TEXT MEASUREMENT
# ============================================================

def text_width(draw, text, font):
    try:
        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font
        )

        return bbox[2] - bbox[0]

    except Exception:
        try:
            return draw.textlength(
                text,
                font=font
            )
        except Exception:
            return len(text) * 15


def text_height(draw, text, font):
    try:
        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font
        )

        return bbox[3] - bbox[1]

    except Exception:
        return int(font.size if hasattr(font, "size") else 30)


# ============================================================
# WORD WRAPPING
# ============================================================

def wrap_text(draw, text, font, max_width):
    """
    Wrap text by words.

    There is NO sentence-length limit here.
    Long sentences simply use multiple lines.
    """

    if not text:
        return []

    text = str(text).strip()

    if not text:
        return []

    words = text.split()

    lines = []

    current = ""

    for word in words:

        candidate = word if not current else current + " " + word

        if text_width(draw, candidate, font) <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)

        # Handle a single word longer than the available width.
        if text_width(draw, word, font) > max_width:

            partial = ""

            for character in word:

                test = partial + character

                if text_width(draw, test, font) <= max_width:
                    partial = test
                else:
                    if partial:
                        lines.append(partial)

                    partial = character

            current = partial

        else:
            current = word

    if current:
        lines.append(current)

    return lines


# ============================================================
# SUBTITLE DRAWING
# ============================================================

def draw_subtitle(
    frame,
    english,
    translated,
    alpha=1.0
):
    """
    Draws English + Hindi directly onto the camera frame.

    This is the subtitle that will travel through:

        Unity Capture
            ↓
        OBS
            ↓
        OBS Virtual Camera
            ↓
        Google Meet

    Google Meet itself does NOT generate these subtitles.
    """

    if Image is None or ImageDraw is None:
        return frame

    if not english and not translated:
        return frame

    alpha = max(
        0.0,
        min(1.0, float(alpha))
    )

    if alpha <= 0:
        return frame

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Convert OpenCV BGR -> PIL RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb).convert("RGBA")

    overlay = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(overlay)

    # --------------------------------------------------------
    # Subtitle box dimensions
    # --------------------------------------------------------

    box_width = int(
        width * SUBTITLE_WIDTH_RATIO
    )

    box_x = int(
        (width - box_width) / 2
    )

    max_text_width = (
        box_width
        - SUBTITLE_SIDE_PADDING * 2
    )

    # --------------------------------------------------------
    # Determine fonts
    # --------------------------------------------------------

    english_size = ENGLISH_FONT_SIZE
    translation_size = TRANSLATION_FONT_SIZE

    while english_size >= MIN_FONT_SIZE:

        english_font = get_font(
            english_size
        )

        translation_font = get_font(
            translation_size
        )

        english_lines = wrap_text(
            draw,
            english,
            english_font,
            max_text_width
        )

        translation_lines = wrap_text(
            draw,
            translated,
            translation_font,
            max_text_width
        )

        total_lines = (
            len(english_lines)
            + len(translation_lines)
        )

        english_line_height = (
            text_height(
                draw,
                "Ag",
                english_font
            )
        )

        translation_line_height = (
            text_height(
                draw,
                "अ",
                translation_font
            )
        )

        content_height = (
            len(english_lines)
            * english_line_height
            +
            len(translation_lines)
            * translation_line_height
            +
            max(
                0,
                total_lines - 1
            )
            * LINE_SPACING
        )

        box_height = (
            content_height
            + SUBTITLE_TOP_PADDING
            + SUBTITLE_BOTTOM_PADDING
        )

        # Keep reducing the font if the subtitle
        # becomes too tall for the camera.
        if box_height <= int(height * 0.40):
            break

        english_size -= 2

        translation_size = max(
            MIN_FONT_SIZE,
            translation_size - 2
        )

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    box_y = (
        height
        - SUBTITLE_BOTTOM_MARGIN
        - box_height
    )

    box_y = max(
        20,
        int(box_y)
    )

    box_x2 = box_x + box_width
    box_y2 = box_y + box_height

    # --------------------------------------------------------
    # Draw background
    # --------------------------------------------------------

    background_alpha = int(
        215 * alpha
    )

    draw.rounded_rectangle(
        (
            box_x,
            box_y,
            box_x2,
            box_y2
        ),
        radius=SUBTITLE_RADIUS,
        fill=(
            10,
            15,
            25,
            background_alpha
        ),
        outline=(
            255,
            255,
            255,
            int(45 * alpha)
        ),
        width=2
    )

    # --------------------------------------------------------
    # Draw small language indicator
    # --------------------------------------------------------

    indicator_font = get_font(17)

    indicator = "SIGNFLOW"

    draw.text(
        (
            box_x + SUBTITLE_SIDE_PADDING,
            box_y + 8
        ),
        indicator,
        font=indicator_font,
        fill=(
            120,
            210,
            255,
            int(210 * alpha)
        )
    )

    # --------------------------------------------------------
    # Draw English
    # --------------------------------------------------------

    cursor_y = (
        box_y
        + SUBTITLE_TOP_PADDING
        + 22
    )

    english_font = get_font(
        english_size
    )

    translation_font = get_font(
        translation_size
    )

    for line in english_lines:

        draw.text(
            (
                box_x
                + SUBTITLE_SIDE_PADDING,
                cursor_y
            ),
            line,
            font=english_font,
            fill=(
                255,
                255,
                255,
                int(255 * alpha)
            )
        )

        cursor_y += (
            text_height(
                draw,
                line,
                english_font
            )
            + LINE_SPACING
        )

    # --------------------------------------------------------
    # Separator
    # --------------------------------------------------------

    if translated:

        separator_y = cursor_y + 2

        draw.line(
            (
                box_x
                + SUBTITLE_SIDE_PADDING,
                separator_y,

                box_x2
                - SUBTITLE_SIDE_PADDING,
                separator_y
            ),
            fill=(
                255,
                255,
                255,
                int(50 * alpha)
            ),
            width=1
        )

        cursor_y += 12

    # --------------------------------------------------------
    # Draw Hindi translation
    # --------------------------------------------------------

    for line in translation_lines:

        draw.text(
            (
                box_x
                + SUBTITLE_SIDE_PADDING,
                cursor_y
            ),
            line,
            font=translation_font,
            fill=(
                190,
                235,
                255,
                int(255 * alpha)
            )
        )

        cursor_y += (
            text_height(
                draw,
                line,
                translation_font
            )
            + LINE_SPACING
        )

    # --------------------------------------------------------
    # Composite
    # --------------------------------------------------------

    image = Image.alpha_composite(
        image,
        overlay
    )

    result = np.array(
        image.convert("RGB")
    )

    result = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2BGR
    )

    return result


# ============================================================
# LANGUAGE WORKER
# ============================================================

class LiveLanguageWorker:
    """
    Runs Qwen + Argos in a background thread.

    Main camera loop NEVER waits for Qwen/Argos.
    """

    def __init__(
        self,
        target_language,
        result_queue
    ):
        self.target_language = target_language

        self.result_queue = result_queue

        self.condition = threading.Condition()

        self.pending_words = None

        self.pending_version = 0

        self.pending_time = 0

        self.running = True

        self.thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )

        self.thread.start()

    def submit(self, words):

        if not words:
            return

        words = [
            str(word).strip()
            for word in words
            if word and str(word).strip()
        ]

        if not words:
            return

        with self.condition:

            self.pending_version += 1

            self.pending_words = list(words)

            self.pending_time = time.monotonic()

            self.condition.notify()

    def stop(self):

        with self.condition:

            self.running = False

            self.condition.notify()

        if self.thread.is_alive():
            self.thread.join(
                timeout=2
            )

    def _worker_loop(self):

        while True:

            with self.condition:

                while (
                    self.running
                    and self.pending_words is None
                ):
                    self.condition.wait()

                if not self.running:
                    return

                version = self.pending_version

                words = list(
                    self.pending_words
                )

                submitted_at = self.pending_time

            # ------------------------------------------------
            # Debounce
            # ------------------------------------------------

            while self.running:

                elapsed = (
                    time.monotonic()
                    - submitted_at
                )

                remaining = (
                    LIVE_TRANSLATION_DEBOUNCE
                    - elapsed
                )

                if remaining <= 0:
                    break

                time.sleep(
                    min(
                        remaining,
                        0.1
                    )
                )

                with self.condition:

                    # Newer words arrived.
                    if (
                        self.pending_version
                        != version
                    ):
                        break

            if not self.running:
                return

            # ------------------------------------------------
            # Check for newer sentence before processing.
            # ------------------------------------------------

            with self.condition:

                if (
                    self.pending_version
                    != version
                ):
                    continue

                words = list(
                    self.pending_words
                )

                self.pending_words = None

            # ------------------------------------------------
            # Qwen + Argos
            # ------------------------------------------------

            try:

                print()
                print(
                    "[LIVE LANGUAGE] Processing:"
                )

                print(
                    " ".join(words)
                )

                english, translated = (
                    generate_language_result(
                        words,
                        self.target_language
                    )
                )

                if english:

                    self.result_queue.put(
                        (
                            version,
                            words,
                            english,
                            translated
                        )
                    )

            except Exception as e:

                print(
                    "[LIVE LANGUAGE] ERROR:"
                )

                print(e)

                traceback.print_exc()


# ============================================================
# TTS WORKER
# ============================================================

class TTSWorker:
    """
    Runs TTS separately so audio generation/playback
    never freezes the camera.
    """

    def __init__(self):

        self.queue = queue.Queue()

        self.running = True

        self.thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )

        self.thread.start()

    def speak(self, text):

        if not text:
            return

        self.queue.put(
            str(text)
        )

    def stop(self):

        self.running = False

        self.queue.put(None)

        if self.thread.is_alive():

            self.thread.join(
                timeout=3
            )

    def _worker_loop(self):

        while self.running:

            try:

                text = self.queue.get(
                    timeout=0.2
                )

            except queue.Empty:

                continue

            if text is None:
                break

            try:

                print()
                print(
                    "[TTS] Speaking final sentence..."
                )

                text_to_speech(text)

                print(
                    "[TTS] Finished."
                )

            except Exception as e:

                print(
                    "[TTS] ERROR:"
                )

                print(e)

                traceback.print_exc()


# ============================================================
# WEBSOCKET HELPER
# ============================================================

class BackendConnection:

    def __init__(self, url):

        self.url = url

        self.ws = None

        self.last_attempt = 0

    def connect(self):

        if websocket is None:

            print(
                "[WEBSOCKET] websocket-client not installed."
            )

            return False

        now = time.monotonic()

        if now - self.last_attempt < 3:

            return False

        self.last_attempt = now

        try:

            print(
                "[WEBSOCKET] Connecting..."
            )

            self.ws = websocket.create_connection(
                self.url,
                timeout=1
            )

            print(
                "[WEBSOCKET] Connected."
            )

            try:

                self.ws.send(
                    json.dumps(
                        {
                            "type": "register",
                            "client": "signflow"
                        }
                    )
                )

            except Exception:
                pass

            return True

        except Exception as e:

            print(
                "[WEBSOCKET] Connection failed:",
                e
            )

            self.ws = None

            return False

    def send_json(self, data):

        if self.ws is None:

            self.connect()

        if self.ws is None:
            return False

        try:

            self.ws.send(
                json.dumps(data)
            )

            return True

        except Exception as e:

            print(
                "[WEBSOCKET] JSON send failed:",
                e
            )

            self.close()

            return False

    def send_frame(self, frame):

        if self.ws is None:

            self.connect()

        if self.ws is None:
            return False

        try:

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    WEBSOCKET_JPEG_QUALITY
                ]
            )

            if not ok:
                return False

            self.ws.send_binary(
                encoded.tobytes()
            )

            return True

        except Exception as e:

            print(
                "[WEBSOCKET] Frame send failed:",
                e
            )

            self.close()

            return False

    def close(self):

        if self.ws is not None:

            try:
                self.ws.close()
            except Exception:
                pass

        self.ws = None


# ============================================================
# MEDIAPIPE LANDMARK EXTRACTION
# ============================================================

def create_hand_landmarker():

    if not os.path.exists(
        HAND_LANDMARKER_PATH
    ):

        raise FileNotFoundError(
            "hand_landmarker.task not found:\n"
            + HAND_LANDMARKER_PATH
        )

    base_options = BaseOptions(
        model_asset_path=HAND_LANDMARKER_PATH
    )

    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2
    )

    return HandLandmarker.create_from_options(
        options
    )


def extract_features(result):

    if not result.hand_landmarks:

        return None

    # --------------------------------------------------------
    # The existing ISL Random Forest uses the first detected
    # hand's 21 x/y/z landmarks = 63 features.
    # --------------------------------------------------------

    hand = result.hand_landmarks[0]

    features = []

    for landmark in hand:

        features.extend(
            [
                float(landmark.x),
                float(landmark.y),
                float(landmark.z)
            ]
        )

    if len(features) != 63:

        return None

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "ISL model not found:\n"
            + MODEL_PATH
        )

    print(
        "[MODEL] Loading:",
        MODEL_PATH
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "[MODEL] Random Forest loaded."
    )

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict_word(model, features):

    if features is None:
        return None

    try:

        features_2d = features.reshape(
            1,
            -1
        )

        prediction = model.predict(
            features_2d
        )

        if prediction is None:
            return None

        if len(prediction) == 0:
            return None

        word = str(
            prediction[0]
        ).strip()

        if not word:
            return None

        return word

    except Exception as e:

        print(
            "[MODEL] Prediction error:",
            e
        )

        return None


# ============================================================
# LANDMARK DRAWING
# ============================================================

def draw_hand_landmarks(
    frame,
    result
):

    if not result.hand_landmarks:
        return frame

    height, width = frame.shape[:2]

    for hand in result.hand_landmarks:

        points = []

        for landmark in hand:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            points.append(
                (x, y)
            )

        # ----------------------------------------------------
        # Draw points
        # ----------------------------------------------------

        for x, y in points:

            cv2.circle(
                frame,
                (x, y),
                3,
                (80, 220, 255),
                -1
            )

        # ----------------------------------------------------
        # MediaPipe hand connections
        # ----------------------------------------------------

        connections = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),

            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),

            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),

            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),

            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),

            (5, 9),
            (9, 13),
            (13, 17),
        ]

        for a, b in connections:

            if (
                a < len(points)
                and b < len(points)
            ):

                cv2.line(
                    frame,
                    points[a],
                    points[b],
                    (80, 200, 255),
                    2
                )

    return frame


# ============================================================
# TOP STATUS UI
# ============================================================

def draw_status(
    frame,
    words,
    english,
    translated,
    processing=False
):

    height, width = frame.shape[:2]

    # Top translucent bar
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, 64),
        (8, 12, 20),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.78,
        frame,
        0.22,
        0
    )

    # --------------------------------------------------------
    # SignFlow label
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "SIGNFLOW",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (120, 220, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if processing:

        status = "TRANSLATING..."

    elif words:

        status = "LISTENING"

    else:

        status = "READY"

    cv2.putText(
        frame,
        status,
        (width - 190, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (220, 235, 245),
        2,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# SUBTITLE STATE
# ============================================================

class SubtitleState:

    def __init__(self):

        self.english = ""

        self.translated = ""

        self.last_update = 0

        self.fade_start = None

        self.fade_end = None

    def update(
        self,
        english,
        translated
    ):

        self.english = (
            english or ""
        ).strip()

        self.translated = (
            translated or ""
        ).strip()

        self.last_update = (
            time.monotonic()
        )

        self.fade_start = None

        self.fade_end = None

    def start_fade(self):

        if not self.english and not self.translated:
            return

        now = time.monotonic()

        self.fade_start = (
            now
            + SUBTITLE_HOLD_SECONDS
        )

        self.fade_end = (
            self.fade_start
            + SUBTITLE_FADE_SECONDS
        )

    def clear(self):

        self.english = ""

        self.translated = ""

        self.last_update = 0

        self.fade_start = None

        self.fade_end = None

    def alpha(self):

        if not self.english and not self.translated:
            return 0.0

        if self.fade_start is None:
            return 1.0

        now = time.monotonic()

        if now < self.fade_start:
            return 1.0

        if now >= self.fade_end:
            self.clear()
            return 0.0

        remaining = (
            self.fade_end - now
        )

        return max(
            0.0,
            min(
                1.0,
                remaining
                / SUBTITLE_FADE_SECONDS
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SIGNFLOW REALTIME RECOGNITION")
    print("=" * 70)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()

    model = load_model()

    # --------------------------------------------------------
    # Word stream
    # --------------------------------------------------------

    print()

    try:

        word_stream = WordStream(
            stable_frames=5,
            cooldown=0.8
        )

    except TypeError:

        try:

            word_stream = WordStream(
                stable_frames=5
            )

        except TypeError:

            word_stream = WordStream()

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    print()

    print(
        "[MEDIAPIPE] Loading hand landmarker..."
    )

    landmarker = create_hand_landmarker()

    print(
        "[MEDIAPIPE] Ready."
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    print()

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        print(
            "[CAMERA] Could not open camera."
        )

        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS
    )

    actual_width = int(
        camera.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        camera.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        "[CAMERA] Resolution:",
        actual_width,
        "x",
        actual_height
    )

    # --------------------------------------------------------
    # Unity Video Capture
    # --------------------------------------------------------

    virtual_cam = None

    if pyvirtualcam is not None:

        try:

            print()

            print(
                "[VIRTUAL CAMERA] Connecting to Unity Video Capture..."
            )

            virtual_cam = pyvirtualcam.Camera(
                width=VIRTUAL_CAMERA_WIDTH,
                height=VIRTUAL_CAMERA_HEIGHT,
                fps=VIRTUAL_CAMERA_FPS,
                fmt=pyvirtualcam.PixelFormat.RGB,
                backend=VIRTUAL_CAMERA_BACKEND
            )

            print(
                "[VIRTUAL CAMERA] Connected:",
                virtual_cam.device
            )

            print(
                "[VIRTUAL CAMERA] Resolution:",
                VIRTUAL_CAMERA_WIDTH,
                "x",
                VIRTUAL_CAMERA_HEIGHT
            )

            print(
                "[VIRTUAL CAMERA] FPS:",
                VIRTUAL_CAMERA_FPS
            )

        except Exception as e:

            print(
                "[VIRTUAL CAMERA] ERROR:"
            )

            print(e)

            traceback.print_exc()

            virtual_cam = None

    else:

        print(
            "[VIRTUAL CAMERA] pyvirtualcam is not installed."
        )

    # --------------------------------------------------------
    # Backend websocket
    # --------------------------------------------------------

    backend = BackendConnection(
        WEBSOCKET_URL
    )

    # --------------------------------------------------------
    # Language worker
    # --------------------------------------------------------

    language_results = queue.Queue()

    language_worker = LiveLanguageWorker(
        TARGET_LANGUAGE,
        language_results
    )

    # --------------------------------------------------------
    # TTS worker
    # --------------------------------------------------------

    tts_worker = TTSWorker()

    # --------------------------------------------------------
    # Subtitle state
    # --------------------------------------------------------

    subtitles = SubtitleState()

    # --------------------------------------------------------
    # Sentence state
    # --------------------------------------------------------

    current_words = []

    current_version = 0

    latest_translation_version = 0

    last_word_time = None

    sentence_finished = False

    processing_language = False

    # --------------------------------------------------------
    # Frame timing
    # --------------------------------------------------------

    frame_counter = 0

    last_websocket_send = 0

    last_virtual_frame_time = time.monotonic()

    # --------------------------------------------------------
    # Recognition history
    # --------------------------------------------------------

    prediction_history = []

    HISTORY_SIZE = 8

    # --------------------------------------------------------
    # Unknown handling
    # --------------------------------------------------------

    unknown_frames = 0

    UNKNOWN_RESET_FRAMES = 8

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONTROLS")
    print("=" * 70)
    print("ENTER     = finish sentence + speak")
    print("BACKSPACE = remove last word")
    print("C         = clear sentence")
    print("Q         = quit")
    print("=" * 70)

    # --------------------------------------------------------
    # Helper: submit translation
    # --------------------------------------------------------

    def submit_current_sentence():

        nonlocal current_version
        nonlocal processing_language

        words = list(
            current_words
        )

        if not words:
            return

        current_version += 1

        processing_language = True

        language_worker.submit(
            words
        )

    # --------------------------------------------------------
    # Helper: finalize sentence
    # --------------------------------------------------------

    def finalize_sentence(
        automatic=False
    ):

        nonlocal current_words
        nonlocal sentence_finished
        nonlocal processing_language
        nonlocal latest_translation_version
        nonlocal last_word_time

        if not current_words:
            return

        print()
        print("=" * 70)

        if automatic:

            print(
                "AUTOMATIC SENTENCE COMPLETE"
            )

        else:

            print(
                "MANUAL SENTENCE COMPLETE"
            )

        print(
            "Words:",
            " ".join(current_words)
        )

        print(
            "English:",
            subtitles.english
        )

        print(
            "Hindi:",
            subtitles.translated
        )

        print("=" * 70)

        # ----------------------------------------------------
        # If translation hasn't finished yet, wait for the
        # background worker rather than speaking an empty result.
        # ----------------------------------------------------

        if (
            not subtitles.translated
            and subtitles.english
        ):

            print(
                "[FINALIZE] Translation not ready yet."
            )

        # ----------------------------------------------------
        # Speak final Hindi once.
        # ----------------------------------------------------

        if (
            AUTO_SPEAK_FINAL_SENTENCE
            and subtitles.translated
        ):

            tts_worker.speak(
                subtitles.translated
            )

        # ----------------------------------------------------
        # Start subtitle fade.
        # ----------------------------------------------------

        subtitles.start_fade()

        # ----------------------------------------------------
        # Send final sentence to backend/frontend.
        # ----------------------------------------------------

        backend.send_json(
            {
                "type": "sentence_result",
                "sentence": subtitles.english,
                "translation": subtitles.translated,
                "language": TARGET_LANGUAGE
            }
        )

        # ----------------------------------------------------
        # Clear recognition sentence immediately.
        #
        # Subtitle remains visible while fading, but new signs
        # can immediately start a new sentence.
        # ----------------------------------------------------

        try:

            word_stream.clear()

        except Exception:

            pass

        current_words = []

        sentence_finished = True

        processing_language = False

        last_word_time = None

        latest_translation_version = 0

        prediction_history.clear()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            ret, raw_frame = camera.read()

            if not ret:

                print(
                    "[CAMERA] Failed to read frame."
                )

                time.sleep(0.05)

                continue

            frame_counter += 1

            # =================================================
            # RECOGNITION FRAME
            # =================================================
            #
            # We continue using a mirrored frame for recognition
            # because your existing model was working with this
            # orientation.
            #
            # IMPORTANT:
            # The OUTGOING video is flipped back later.
            #
            # This means:
            #
            # Recognition = existing working orientation
            # Meet video   = normal/unmirrored orientation
            #
            # =================================================

            recognition_frame = cv2.flip(
                raw_frame,
                1
            )

            # -------------------------------------------------
            # Resize processing frame
            # -------------------------------------------------

            processing_frame = cv2.resize(
                recognition_frame,
                (
                    PROCESS_WIDTH,
                    PROCESS_HEIGHT
                ),
                interpolation=cv2.INTER_AREA
            )

            # -------------------------------------------------
            # MediaPipe image
            # -------------------------------------------------

            rgb_processing = cv2.cvtColor(
                processing_frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_processing
            )

            # -------------------------------------------------
            # Detect hands
            # -------------------------------------------------

            result = landmarker.detect(
                mp_image
            )

            # -------------------------------------------------
            # Features
            # -------------------------------------------------

            features = extract_features(
                result
            )

            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------

            predicted_word = predict_word(
                model,
                features
            )

            # -------------------------------------------------
            # WordStream
            # -------------------------------------------------

            new_word = None

            if predicted_word:

                try:

                    new_word = word_stream.process(
                        predicted_word
                    )

                except TypeError:

                    try:

                        new_word = word_stream.process(
                            predicted_word,
                            features
                        )

                    except Exception as e:

                        print(
                            "[WORDSTREAM] Error:",
                            e
                        )

            # -------------------------------------------------
            # New stable word
            # -------------------------------------------------

            if new_word:

                new_word = str(
                    new_word
                ).strip()

                if new_word:

                    print()
                    print(
                        "[RECOGNITION] New word:",
                        new_word
                    )

                    current_words = list(
                        word_stream.get_words()
                    )

                    if not current_words:

                        current_words = (
                            current_words
                            + [new_word]
                        )

                    sentence_finished = False

                    last_word_time = (
                        time.monotonic()
                    )

                    # A new sentence starts.
                    # Remove the previous fading subtitle.
                    if subtitles.fade_start is not None:

                        subtitles.clear()

                    # ------------------------------------------------
                    # Automatically ask Qwen/Argos to update.
                    # ------------------------------------------------

                    submit_current_sentence()

                    # ------------------------------------------------
                    # Send live recognition state.
                    # ------------------------------------------------

                    backend.send_json(
                        {
                            "type": "sentence_state",
                            "sentence": " ".join(
                                current_words
                            ),
                            "words": current_words
                        }
                    )

            # -------------------------------------------------
            # Get current WordStream sentence.
            # -------------------------------------------------

            try:

                stream_words = list(
                    word_stream.get_words()
                )

                if stream_words:

                    current_words = (
                        stream_words
                    )

            except Exception:

                pass

            # -------------------------------------------------
            # Process background language results.
            # -------------------------------------------------

            while True:

                try:

                    (
                        version,
                        words,
                        english,
                        translated
                    ) = language_results.get_nowait()

                except queue.Empty:

                    break

                # ---------------------------------------------
                # Ignore stale results.
                #
                # Example:
                #
                # Qwen was processing:
                # "tomorrow you"
                #
                # meanwhile signer added:
                # "come"
                #
                # We don't want the old result replacing
                # the newer sentence.
                # ---------------------------------------------

                if version != current_version:

                    print(
                        "[LIVE LANGUAGE] Ignoring stale result."
                    )

                    continue

                processing_language = False

                latest_translation_version = (
                    version
                )

                print()
                print(
                    "[LIVE LANGUAGE] English:",
                    english
                )

                print(
                    "[LIVE LANGUAGE] Hindi:",
                    translated
                )

                # ------------------------------------------------
                # Update subtitle.
                # ------------------------------------------------

                subtitles.update(
                    english,
                    translated
                )

                # ------------------------------------------------
                # Send to frontend/backend as well.
                # ------------------------------------------------

                backend.send_json(
                    {
                        "type": "sentence_result",
                        "sentence": english,
                        "translation": translated,
                        "language": TARGET_LANGUAGE,
                        "live": True
                    }
                )

            # -------------------------------------------------
            # Automatic sentence completion.
            # -------------------------------------------------

            if (
                current_words
                and last_word_time is not None
                and not sentence_finished
            ):

                idle_time = (
                    time.monotonic()
                    - last_word_time
                )

                if (
                    idle_time
                    >= SENTENCE_IDLE_SECONDS
                    and latest_translation_version
                    == current_version
                    and subtitles.translated
                ):

                    finalize_sentence(
                        automatic=True
                    )

            # =================================================
            # BUILD OUTGOING VIDEO
            # =================================================

            # -------------------------------------------------
            # Flip recognition frame back to normal orientation.
            #
            # THIS is the frame sent to Meet.
            #
            # Therefore the video is NOT mirrored.
            # -------------------------------------------------

            output_frame = cv2.flip(
                recognition_frame,
                1
            )

            # -------------------------------------------------
            # Draw hand landmarks.
            #
            # We draw them on the normal output frame by
            # converting the MediaPipe coordinates back to
            # output orientation.
            # -------------------------------------------------

            output_frame = draw_hand_landmarks(
                output_frame,
                result
            )

            # -------------------------------------------------
            # Subtitle
            # -------------------------------------------------

            subtitle_alpha = subtitles.alpha()

            if subtitle_alpha > 0:

                output_frame = draw_subtitle(
                    output_frame,
                    subtitles.english,
                    subtitles.translated,
                    subtitle_alpha
                )

            # -------------------------------------------------
            # Status bar
            # -------------------------------------------------

            output_frame = draw_status(
                output_frame,
                current_words,
                subtitles.english,
                subtitles.translated,
                processing_language
            )

            # =================================================
            # WEBSOCKET FRAME
            # =================================================

            now = time.monotonic()

            if (
                now - last_websocket_send
                >= WEBSOCKET_SEND_INTERVAL
            ):

                backend.send_frame(
                    output_frame
                )

                last_websocket_send = now

            # =================================================
            # UNITY VIRTUAL CAMERA
            # =================================================

            if virtual_cam is not None:

                try:

                    virtual_frame = output_frame

                    if (
                        virtual_frame.shape[1]
                        != VIRTUAL_CAMERA_WIDTH
                        or
                        virtual_frame.shape[0]
                        != VIRTUAL_CAMERA_HEIGHT
                    ):

                        virtual_frame = cv2.resize(
                            virtual_frame,
                            (
                                VIRTUAL_CAMERA_WIDTH,
                                VIRTUAL_CAMERA_HEIGHT
                            ),
                            interpolation=cv2.INTER_AREA
                        )

                    virtual_rgb = cv2.cvtColor(
                        virtual_frame,
                        cv2.COLOR_BGR2RGB
                    )

                    virtual_cam.send(
                        virtual_rgb
                    )

                    virtual_cam.sleep_until_next_frame()

                except Exception as e:

                    print(
                        "[VIRTUAL CAMERA] Send error:",
                        e
                    )

                    try:
                        virtual_cam.close()
                    except Exception:
                        pass

                    virtual_cam = None

            # =================================================
            # KEYBOARD CONTROLS
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            # -------------------------------------------------
            # Q
            # -------------------------------------------------

            if key == ord("q"):

                print(
                    "[SYSTEM] Quit requested."
                )

                break

            # -------------------------------------------------
            # C
            # -------------------------------------------------

            elif key == ord("c"):

                print(
                    "[SYSTEM] Clearing sentence."
                )

                try:

                    word_stream.clear()

                except Exception:

                    pass

                current_words = []

                prediction_history.clear()

                subtitles.clear()

                sentence_finished = False

                processing_language = False

                last_word_time = None

                current_version += 1

                latest_translation_version = 0

                backend.send_json(
                    {
                        "type": "sentence_state",
                        "sentence": "",
                        "words": []
                    }
                )

            # -------------------------------------------------
            # BACKSPACE
            # -------------------------------------------------

            elif key == 8:

                print(
                    "[SYSTEM] Removing last word."
                )

                try:

                    word_stream.remove_last_word()

                except Exception as e:

                    print(
                        "[WORDSTREAM] Remove error:",
                        e
                    )

                try:

                    current_words = list(
                        word_stream.get_words()
                    )

                except Exception:

                    current_words = []

                if current_words:

                    last_word_time = (
                        time.monotonic()
                    )

                    submit_current_sentence()

                else:

                    subtitles.clear()

                    last_word_time = None

                    processing_language = False

            # -------------------------------------------------
            # ENTER
            # -------------------------------------------------

            elif key == 13:

                print(
                    "[SYSTEM] Manual sentence completion."
                )

                if current_words:

                    # If latest translation is ready,
                    # finalize immediately.
                    if (
                        latest_translation_version
                        == current_version
                        and subtitles.translated
                    ):

                        finalize_sentence(
                            automatic=False
                        )

                    else:

                        print(
                            "[SYSTEM] Waiting for latest translation..."
                        )

                        # Ask worker one final time.
                        submit_current_sentence()

                        # Give the worker some time to finish.
                        wait_start = time.monotonic()

                        while (
                            time.monotonic()
                            - wait_start
                            < 8.0
                        ):

                            try:

                                (
                                    version,
                                    words,
                                    english,
                                    translated
                                ) = (
                                    language_results.get(
                                        timeout=0.2
                                    )
                                )

                            except queue.Empty:

                                continue

                            if version != current_version:

                                continue

                            subtitles.update(
                                english,
                                translated
                            )

                            latest_translation_version = (
                                version
                            )

                            processing_language = False

                            break

                        finalize_sentence(
                            automatic=False
                        )

            # -------------------------------------------------
            # Escape also quits
            # -------------------------------------------------

            elif key == 27:

                break

    except KeyboardInterrupt:

        print()
        print(
            "[SYSTEM] Keyboard interrupt."
        )

    except Exception as e:

        print()
        print(
            "=" * 70
        )

        print(
            "[SYSTEM] FATAL ERROR"
        )

        print(e)

        traceback.print_exc()

        print(
            "=" * 70
        )

    finally:

        print()
        print(
            "[SYSTEM] Cleaning up..."
        )

        # ----------------------------------------------------
        # Stop language worker
        # ----------------------------------------------------

        try:

            language_worker.stop()

        except Exception:

            pass

        # ----------------------------------------------------
        # Stop TTS worker
        # ----------------------------------------------------

        try:

            tts_worker.stop()

        except Exception:

            pass

        # ----------------------------------------------------
        # Close camera
        # ----------------------------------------------------

        try:

            camera.release()

        except Exception:

            pass

        # ----------------------------------------------------
        # Close MediaPipe
        # ----------------------------------------------------

        try:

            landmarker.close()

        except Exception:

            pass

        # ----------------------------------------------------
        # Close websocket
        # ----------------------------------------------------

        try:

            backend.close()

        except Exception:

            pass

        # ----------------------------------------------------
        # Close Unity Capture
        # ----------------------------------------------------

        try:

            if virtual_cam is not None:

                virtual_cam.close()

        except Exception:

            pass

        # ----------------------------------------------------
        # OpenCV
        # ----------------------------------------------------

        try:

            cv2.destroyAllWindows()

        except Exception:

            pass

        print(
            "[SYSTEM] SignFlow stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()