import os
import sys
import json
import time

from collections import Counter

import cv2
import numpy as np
import joblib
import mediapipe as mp


from PIL import Image, ImageDraw, ImageFont




PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models",
    "isl"
)

VISION_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "app",
    "vision"
)

LANGUAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "app",
    "language"
)

# Make relative paths inside the language module behave
# exactly the same way as when running from the project root.
os.chdir(PROJECT_ROOT)

sys.path.insert(0, LANGUAGE_DIR)
sys.path.insert(0, VISION_DIR)


# ============================================================
# LANGUAGE IMPORTS
# ============================================================

from .gesture_state import GestureState
from .primary_person_filter import PrimaryUserHandFilter

# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "isl_static_model.pkl"
)

TRAINING_DATA_PATH = os.path.join(
    MODEL_DIR,
    "isl_training_data.npz"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "isl_model_config.json"
)

HAND_LANDMARKER_PATH = os.path.join(
    VISION_DIR,
    "hand_landmarker.task"
)

POSE_LANDMARKER_PATH = os.path.join(
    MODEL_DIR,
    "pose_landmarker_full.task"
)

POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0


# ------------------------------------------------------------
# Recognition stability
# ------------------------------------------------------------

HISTORY_SIZE = 8

STABLE_REQUIRED = 5


# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------

CAMERA_WIDTH = 1280

CAMERA_HEIGHT = 720


# ------------------------------------------------------------
# Window
# ------------------------------------------------------------

WINDOW_WIDTH = 1100

WINDOW_HEIGHT = 700


# ------------------------------------------------------------
# Normal Windows Hindi font
# ------------------------------------------------------------

HINDI_FONT_PATH = r"C:\Windows\Fonts\mangal.ttf"


# ============================================================
# CHECK HINDI FONT
# ============================================================

if not os.path.exists(HINDI_FONT_PATH):

    print()
    print("[FONT ERROR] Mangal font not found:")
    print(HINDI_FONT_PATH)
    print()

    print("Checking alternative Windows fonts...")

    alternative_fonts = [

        r"C:\Windows\Fonts\mangalb.ttf",

        r"C:\Windows\Fonts\Nirmala.ttc"
    ]

    found_font = None

    for font_path in alternative_fonts:

        if os.path.exists(font_path):

            found_font = font_path

            break

    if found_font:

        HINDI_FONT_PATH = found_font

        print(
            "[FONT] Using:",
            HINDI_FONT_PATH
        )

    else:

        print(
            "[FONT ERROR] No usable Unicode font found."
        )

        sys.exit(1)

else:

    print(
        "[FONT] Using normal Windows font:",
        HINDI_FONT_PATH
    )


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print()
print("[MODEL] Checking files...")


if not os.path.exists(MODEL_PATH):

    print("[ERROR] Model not found:")
    print(MODEL_PATH)

    sys.exit(1)


if not os.path.exists(TRAINING_DATA_PATH):

    print("[ERROR] Training data not found:")
    print(TRAINING_DATA_PATH)

    sys.exit(1)


if not os.path.exists(CONFIG_PATH):

    print("[ERROR] Model config not found:")
    print(CONFIG_PATH)

    sys.exit(1)


if not os.path.exists(HAND_LANDMARKER_PATH):

    print("[ERROR] Hand landmarker not found:")
    print(HAND_LANDMARKER_PATH)

    sys.exit(1)


# The primary-person filter uses the MediaPipe Tasks Pose Landmarker.
# Download the official model once if it is not already present.
if not os.path.exists(POSE_LANDMARKER_PATH):

    print("[POSE] Pose landmarker model not found.")
    print("[POSE] Downloading official MediaPipe Pose Landmarker...")
    print(POSE_LANDMARKER_PATH)

    try:
        from urllib.request import urlretrieve

        urlretrieve(
            POSE_LANDMARKER_URL,
            POSE_LANDMARKER_PATH
        )
        print("[POSE] Pose landmarker downloaded.")
    except Exception as error:
        print("[POSE ERROR] Could not download pose model:")
        print(repr(error))
        print("[POSE] Download it manually from:")
        print(POSE_LANDMARKER_URL)
        print("[POSE] Save it as:")
        print(POSE_LANDMARKER_PATH)
        sys.exit(1)


# ============================================================
# LOAD MODEL
# ============================================================

print("[MODEL] Loading ISL model...")

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

training_data = np.load(
    TRAINING_DATA_PATH,
    allow_pickle=True
)

X_train = training_data["X"]

y_train = training_data["y"]


print(
    "[MODEL] Training data:",
    X_train.shape
)

print(
    "[MODEL] Classes:",
    list(model.classes_)
)


# ============================================================
# LOAD MODEL CONFIG
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:

    model_config = json.load(f)


print(
    "[MODEL] Config loaded"
)


# ============================================================
# EXPECTED HAND COUNT
# ============================================================

expected_hand_count = {}


for class_name in np.unique(y_train):

    class_indices = np.where(
        y_train == class_name
    )[0]

    counts = []

    for index in class_indices:

        sample = X_train[index]

        left_hand = np.any(
            np.abs(
                sample[:63]
            ) > 1e-6
        )

        right_hand = np.any(
            np.abs(
                sample[63:126]
            ) > 1e-6
        )

        hand_count = (
            int(left_hand)
            +
            int(right_hand)
        )

        counts.append(
            hand_count
        )

    if counts:

        expected_hand_count[class_name] = (
            Counter(
                counts
            ).most_common(1)[0][0]
        )


print(
    "[MODEL] Expected hand counts ready"
)


# ============================================================
# REJECTION THRESHOLDS
# ============================================================

rejection_thresholds = {}


for class_name in np.unique(y_train):

    class_indices = np.where(
        y_train == class_name
    )[0]

    class_samples = X_train[
        class_indices
    ]

    if len(class_samples) == 0:

        continue

    distances = []

    for sample in class_samples:

        diff = (
            class_samples
            -
            sample
        )

        d = np.linalg.norm(
            diff,
            axis=1
        )

        # Remove self-distance.
        d = d[
            d > 1e-6
        ]

        if len(d) > 0:

            distances.append(
                np.min(d)
            )

    if distances:

        rejection_thresholds[class_name] = (
            np.percentile(
                distances,
                95
            )
            * 1.20
        )


print(
    "[MODEL] Rejection thresholds ready"
)


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = (
    mp.tasks.BaseOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(result):

    features = np.zeros(
        126,
        dtype=np.float32
    )

    if not result.hand_landmarks:

        return features

    for hand_index, hand_landmarks in enumerate(
        result.hand_landmarks[:2]
    ):

        handedness = result.handedness[
            hand_index
        ][0].category_name

        if handedness == "Left":

            offset = 0

        else:

            offset = 63

        wrist = hand_landmarks[0]

        wrist_x = wrist.x
        wrist_y = wrist.y
        wrist_z = wrist.z

        middle_mcp = hand_landmarks[9]

        scale = np.sqrt(

            (middle_mcp.x - wrist_x) ** 2
            +
            (middle_mcp.y - wrist_y) ** 2
            +
            (middle_mcp.z - wrist_z) ** 2
        )

        if scale < 1e-6:

            scale = 1.0

        for landmark_index, landmark in enumerate(
            hand_landmarks
        ):

            x = (
                landmark.x
                -
                wrist_x
            ) / scale

            y = (
                landmark.y
                -
                wrist_y
            ) / scale

            z = (
                landmark.z
                -
                wrist_z
            ) / scale

            feature_index = (
                offset
                +
                landmark_index * 3
            )

            features[
                feature_index
            ] = x

            features[
                feature_index + 1
            ] = y

            features[
                feature_index + 2
            ] = z

    return features


# ============================================================
# SIGN PREDICTION
# ============================================================

def get_prediction(result):

    if not result.hand_landmarks:

        return (
            "No hand detected",
            0.0
        )

    features = extract_features(
        result
    )

    hand_count = len(
        result.hand_landmarks
    )

    prediction = model.predict(
        [features]
    )[0]

    prediction = str(
        prediction
    )


    # --------------------------------------------------------
    # Expected hand count
    # --------------------------------------------------------

    expected = expected_hand_count.get(
        prediction
    )

    if expected is not None:

        if hand_count != expected:

            return (
                "Unknown",
                999.0
            )


    # --------------------------------------------------------
    # Distance rejection
    # --------------------------------------------------------

    class_indices = np.where(
        y_train == prediction
    )[0]

    if len(class_indices) == 0:

        return (
            "Unknown",
            999.0
        )

    class_samples = X_train[
        class_indices
    ]

    distances = np.linalg.norm(
        class_samples - features,
        axis=1
    )

    min_distance = float(
        np.min(distances)
    )

    threshold = rejection_thresholds.get(
        prediction,
        float("inf")
    )

    if min_distance > threshold:

        return (
            "Unknown",
            min_distance
        )

    return (
        prediction,
        min_distance
    )


# ============================================================
# DRAW HANDS
# ============================================================

def draw_hands(
    frame,
    result
):

    if not result.hand_landmarks:

        return

    connections = (
        mp.tasks.vision
        .HandLandmarksConnections
        .HAND_CONNECTIONS
    )

    height, width, _ = frame.shape

    for hand_landmarks in result.hand_landmarks:

        # ----------------------------------------------------
        # Connections
        # ----------------------------------------------------

        for connection in connections:

            start = hand_landmarks[
                connection.start
            ]

            end = hand_landmarks[
                connection.end
            ]

            x1 = int(
                start.x * width
            )

            y1 = int(
                start.y * height
            )

            x2 = int(
                end.x * width
            )

            y2 = int(
                end.y * height
            )

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (100, 220, 255),
                2,
                cv2.LINE_AA
            )


        # ----------------------------------------------------
        # Landmarks
        # ----------------------------------------------------

        for landmark in hand_landmarks:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            cv2.circle(
                frame,
                (x, y),
                4,
                (255, 255, 255),
                -1,
                cv2.LINE_AA
            )


# ============================================================
# GLASS OVERLAY
# ============================================================

def rounded_overlay(
    frame,
    x1,
    y1,
    x2,
    y2,
    alpha=0.50,
    radius=18
):

    overlay = frame.copy()

    color = (
        18,
        24,
        35
    )

    cv2.rectangle(
        overlay,
        (
            x1 + radius,
            y1
        ),
        (
            x2 - radius,
            y2
        ),
        color,
        -1
    )

    cv2.rectangle(
        overlay,
        (
            x1,
            y1 + radius
        ),
        (
            x2,
            y2 - radius
        ),
        color,
        -1
    )

    cv2.circle(
        overlay,
        (
            x1 + radius,
            y1 + radius
        ),
        radius,
        color,
        -1
    )

    cv2.circle(
        overlay,
        (
            x2 - radius,
            y1 + radius
        ),
        radius,
        color,
        -1
    )

    cv2.circle(
        overlay,
        (
            x1 + radius,
            y2 - radius
        ),
        radius,
        color,
        -1
    )

    cv2.circle(
        overlay,
        (
            x2 - radius,
            y2 - radius
        ),
        radius,
        color,
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )


# ============================================================
# NORMAL TEXT
# ============================================================

def draw_text(
    frame,
    text,
    position,
    font_scale=0.7,
    thickness=2,
    color=(255, 255, 255)
):

    cv2.putText(
        frame,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# HINDI TEXT USING MANGAL
# ============================================================

def draw_hindi_text(
    frame,
    text,
    position,
    font_size=27,
    color=(205, 225, 255)
):

    if not text:

        return

    try:

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            rgb_frame
        )

        draw = ImageDraw.Draw(
            pil_image
        )

        font = ImageFont.truetype(
            HINDI_FONT_PATH,
            font_size
        )

        rgb_color = (
            color[2],
            color[1],
            color[0]
        )

        draw.text(
            position,
            str(text),
            font=font,
            fill=rgb_color
        )

        frame[:] = cv2.cvtColor(
            np.array(pil_image),
            cv2.COLOR_RGB2BGR
        )

    except Exception as error:

        print(
            "[HINDI RENDER ERROR]",
            error
        )


# ============================================================
# TOP CURRENT SIGN
# ============================================================

def draw_top_sign(
    frame,
    current_sign
):

    if current_sign in (
        "",
        "Unknown",
        "No hand detected"
    ):

        return

    height, width, _ = frame.shape

    text = (
        "SIGN  •  "
        +
        current_sign.upper()
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.60

    thickness = 2

    (
        text_width,
        text_height
    ), _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    padding_x = 22

    padding_y = 13

    box_width = (
        text_width
        +
        padding_x * 2
    )

    box_height = (
        text_height
        +
        padding_y * 2
    )

    x1 = (
        width
        -
        box_width
    ) // 2

    y1 = 18

    x2 = x1 + box_width

    y2 = y1 + box_height

    rounded_overlay(
        frame,
        x1,
        y1,
        x2,
        y2,
        0.55,
        16
    )

    draw_text(
        frame,
        text,
        (
            x1 + padding_x,
            y1
            +
            padding_y
            +
            text_height
            -
            2
        ),
        font_scale,
        thickness
    )


# ============================================================
# RAW SIGN SEQUENCE
# ============================================================

def draw_raw_signs(
    frame,
    sentence_words
):

    if not sentence_words:

        return

    height, width, _ = frame.shape

    visible_words = sentence_words[-8:]

    raw_text = "  →  ".join(
        word.upper()
        for word in visible_words
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.50

    thickness = 1

    (
        text_width,
        text_height
    ), _ = cv2.getTextSize(
        raw_text,
        font,
        font_scale,
        thickness
    )

    padding_x = 20

    padding_y = 11

    box_width = min(
        text_width
        +
        padding_x * 2,
        width - 60
    )

    box_height = (
        text_height
        +
        padding_y * 2
    )

    x1 = (
        width
        -
        box_width
    ) // 2

    y1 = 72

    x2 = x1 + box_width

    y2 = y1 + box_height

    rounded_overlay(
        frame,
        x1,
        y1,
        x2,
        y2,
        0.42,
        14
    )

    if text_width > (
        box_width
        -
        padding_x * 2
    ):

        font_scale = 0.38

    draw_text(
        frame,
        raw_text,
        (
            x1 + padding_x,
            y1
            +
            padding_y
            +
            text_height
            -
            2
        ),
        font_scale,
        thickness,
        (220, 235, 255)
    )


# ============================================================
# BOTTOM LANGUAGE RESULT
# ============================================================

def draw_bottom_language(
    frame,
    english,
    hindi,
    tts_status
):

    if not english and not hindi:

        return

    height, width, _ = frame.shape

    panel_width = min(
        850,
        width - 60
    )

    panel_height = 145

    x1 = (
        width
        -
        panel_width
    ) // 2

    y2 = height - 22

    y1 = y2 - panel_height

    rounded_overlay(
        frame,
        x1,
        y1,
        x1 + panel_width,
        y2,
        0.52,
        20
    )


    # ========================================================
    # ENGLISH
    # ========================================================

    if english:

        draw_text(
            frame,
            english,
            (
                x1 + 22,
                y1 + 42
            ),
            0.68,
            2,
            (255, 255, 255)
        )


    # ========================================================
    # HINDI
    # ========================================================

    if hindi:

        draw_hindi_text(
            frame,
            hindi,
            (
                x1 + 22,
                y1 + 78
            ),
            font_size=27,
            color=(205, 225, 255)
        )


    # ========================================================
    # TTS STATUS
    # ========================================================

    if tts_status:

        (
            status_width,
            status_height
        ), _ = cv2.getTextSize(
            tts_status,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            1
        )

        draw_text(
            frame,
            tts_status,
            (
                x1
                +
                panel_width
                -
                status_width
                -
                20,
                y2 - 18
            ),
            0.42,
            1,
            (180, 220, 255)
        )


# ============================================================
# CONTROLS
# ============================================================

def draw_controls(
    frame
):

    height, width, _ = frame.shape

    text = (
        "ENTER: Translate + Speak   "
        "BACKSPACE: Undo   "
        "C: Clear   "
        "Q: Quit"
    )

    font_scale = 0.38

    thickness = 1

    (
        text_width,
        text_height
    ), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness
    )

    padding = 9

    x1 = 12

    y2 = height - 12

    x2 = (
        x1
        +
        text_width
        +
        padding * 2
    )

    y1 = (
        y2
        -
        text_height
        -
        padding * 2
    )

    rounded_overlay(
        frame,
        x1,
        y1,
        x2,
        y2,
        0.30,
        9
    )

    draw_text(
        frame,
        text,
        (
            x1 + padding,
            y2 - padding
        ),
        font_scale,
        thickness,
        (225, 230, 240)
    )


# ============================================================
# MAIN
# ============================================================

def make_meaningful_sentence(words):
    """Convert recognized sign keywords into a simple sentence offline."""
    words = [str(w).strip().lower() for w in words if w and str(w).strip()]
    if not words:
        return ""

    phrase_map = {
        ("you", "name", "what"): "What is your name?",
        ("your", "name", "what"): "What is your name?",
        ("name", "what"): "What is your name?",
        ("you", "where"): "Where are you?",
        ("where", "you"): "Where are you?",
        ("you", "what"): "What are you doing?",
        ("you", "help"): "Can you help?",
        ("i", "love", "you"): "I love you.",
        ("hello",): "Hello.",
        ("namaste",): "Namaste.",
        ("sorry",): "Sorry.",
        ("yes",): "Yes.",
        ("no",): "No.",
        ("please",): "Please.",
        ("help",): "Please help me.",
        ("pay", "attention"): "Please pay attention.",
        ("correct",): "Correct.",
    }
    key=tuple(words)
    if key in phrase_map: return phrase_map[key]
    if "i" in words and "love" in words and "you" in words: return "I love you."
    if "name" in words and "what" in words: return "What is your name?"
    if "where" in words and "you" in words: return "Where are you?"
    replacements={"i":"I","hello":"Hello","namaste":"Namaste","sorry":"Sorry","please":"Please","yes":"Yes","no":"No","correct":"Correct"}
    sentence=" ".join(replacements.get(w,w) for w in words).strip()
    if not sentence: return ""
    return sentence[0].upper()+sentence[1:]+("?" if any(w in words for w in ("what","where")) else ".")


def main():

    print()
    print("========================================")
    print("          SIGNFLOW REALTIME")
    print("========================================")
    print()

    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    # ========================================================
    # PRIMARY USER HAND FILTER
    # ========================================================
    # MediaPipe 1.0.1 Tasks Pose Landmarker identifies the primary
    # person's body wrists. The existing HandLandmarker and ISL model
    # remain unchanged.

    primary_hand_filter = None

    # ========================================================
    # MEDIAPIPE OPTIONS
    # ========================================================

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=
            HAND_LANDMARKER_PATH
        ),

        running_mode=
        VisionRunningMode.VIDEO,

        num_hands=2,

        min_hand_detection_confidence=0.5,

        min_hand_presence_confidence=0.5,

        min_tracking_confidence=0.5
    )

    # ========================================================
    # NEW GESTURE + LANGUAGE STATE
    # ========================================================

    gesture_state = GestureState(
        history_size=HISTORY_SIZE,
        stable_required=STABLE_REQUIRED
    )

    sentence_words = []

    current_sign = ""
    english_sentence = ""
    hindi_translation = ""
    tts_status = ""

    # ========================================================
    # WINDOW
    # ========================================================

    window_name = "SignFlow"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        WINDOW_WIDTH,
        WINDOW_HEIGHT
    )

    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_ASPECT_RATIO,
        cv2.WINDOW_KEEPRATIO
    )

    start_time = time.time()

    # ========================================================
    # MEDIAPIPE LOOP
    # ========================================================

    try:

        primary_hand_filter = PrimaryUserHandFilter(
    pose_model_path=POSE_LANDMARKER_PATH,
    max_wrist_distance=0.30,
    min_hand_size=0.075,
    min_wrist_visibility=0.25,
)

        print("[POSE] Primary-person filter ready.")

        with HandLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:

                success, frame = cap.read()

                if not success:
                    print("[ERROR] Failed to read camera.")
                    break

                # ------------------------------------------------
                # Mirror camera
                # ------------------------------------------------

                frame = cv2.flip(
                    frame,
                    1
                )

                # ------------------------------------------------
                # Timestamp
                # ------------------------------------------------

                timestamp_ms = int(
                    (
                        time.time()
                        -
                        start_time
                    )
                    *
                    1000
                )

                # ------------------------------------------------
                # MediaPipe image
                # ------------------------------------------------

                mp_image = mp.Image(

                    image_format=
                    mp.ImageFormat.SRGB,

                    data=cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )
                )

                # ------------------------------------------------
                # Detect hands
                # ------------------------------------------------

                result = landmarker.detect_for_video(

                    mp_image,

                    timestamp_ms
                )

                # =================================================
                # PRIMARY USER FILTER
                # =================================================
                # Keep the existing HandLandmarker result format, but
                # remove hands that are not close to the primary user's
                # corresponding body wrist. The ISL model receives the
                # same 126-feature representation as before.

                filtered_result = primary_hand_filter.filter_result(
                    frame,
                    result,
                    timestamp_ms
                )

                # =================================================
                # PREDICTION
                # =================================================

                prediction, distance = (
                    get_prediction(filtered_result)
                )

                valid_prediction = (
                    prediction != "Unknown"
                    and
                    prediction != "No hand detected"
                )

                # =================================================
                # GESTURE STATE
                # =================================================
                #
                # GestureState owns:
                # - stability
                # - duplicate prevention
                # - deciding when a new word is committed
                #
                # The hand does NOT need to leave the frame.
                # =================================================

                committed_word = gesture_state.update(
                    prediction,
                    valid_prediction
                )

                if committed_word:

                    sentence_words.append(
                        committed_word
                    )

                    print()
                    print(
                        "[SIGN ADDED]",
                        committed_word
                    )

                    print(
                        "[RAW SENTENCE]",
                        " ".join(sentence_words)
                    )

                current_sign = (
                    gesture_state.current_stable
                    or
                    ""
                )

                # =================================================
                # LANGUAGE STATE
                # =================================================
                # Meaningful English is kept locally/offline.
                # It remains visible in the preview until the user
                # clears it or commits another sentence.
                hindi_translation = ""
                tts_status = ""

                # =================================================
                # DRAW
                # =================================================

                draw_hands(
                    frame,
                    filtered_result
                )

                draw_top_sign(
                    frame,
                    current_sign
                )

                draw_raw_signs(
                    frame,
                    sentence_words
                )

                draw_bottom_language(
                    frame,
                    english_sentence,
                    hindi_translation,
                    tts_status
                )

                draw_controls(
                    frame
                )

                # =================================================
                # SHOW
                # =================================================

                cv2.imshow(
                    window_name,
                    frame
                )

                # =================================================
                # KEYBOARD
                # =================================================

                key = cv2.waitKey(1) & 0xFF

                # ------------------------------------------------
                # Q = Quit
                # ------------------------------------------------

                if key == ord("q"):
                    break

                # ------------------------------------------------
                # C = Clear
                # ------------------------------------------------

                elif key == ord("c"):

                    sentence_words.clear()
                    gesture_state.reset()
                    english_sentence = ""
                    hindi_translation = ""
                    tts_status = ""

                    print(
                        "[SENTENCE] Cleared"
                    )

                # ------------------------------------------------
                # BACKSPACE = Undo
                # ------------------------------------------------

                elif key == 8:

                    if sentence_words:

                        removed = sentence_words.pop()

                        previous_word = (
                            sentence_words[-1]
                            if sentence_words
                            else None
                        )

                        gesture_state.undo_commit(
                            previous_word
                        )

                        print(
                            "[SENTENCE] Removed:",
                            removed
                        )

                        print(
                            "[RAW SENTENCE]",
                            " ".join(
                                sentence_words
                            )
                        )

                    else:

                        print(
                            "[SENTENCE] Nothing to remove"
                        )

                # ------------------------------------------------
                # ENTER = Commit raw recognized sentence
                # ------------------------------------------------

                elif key == 13:

                    if not sentence_words:

                        print()
                        print("[SENTENCE] No signs entered")
                        print("[SENTENCE] Make some signs first.")
                        continue

                    raw_sentence = " ".join(sentence_words)
                    english_sentence = make_meaningful_sentence(sentence_words)

                    print()
                    print("========================================")
                    print("       SENTENCE COMMITTED")
                    print("========================================")
                    print("[RAW SIGNS]", sentence_words)
                    print("[ENGLISH]", english_sentence)
                    print("========================================")

                    # Keep the meaningful English sentence on screen.
                    # No Qwen, Argos, translation worker, or network call.
                    english_sentence = make_meaningful_sentence(sentence_words)
                    hindi_translation = ""
                    tts_status = ""

                    sentence_words.clear()
                    gesture_state.reset()
    except KeyboardInterrupt:

        print()
        print(
            "[SIGNFLOW] Interrupted."
        )

    finally:

        # ========================================================
        # CLEANUP
        # ========================================================

        if primary_hand_filter is not None:
            try:
                primary_hand_filter.close()
            except Exception as error:
                print("[POSE CLEANUP ERROR]", repr(error))

        cap.release()

        cv2.destroyAllWindows()

        print()
        print(
            "[SIGNFLOW] Camera stopped."
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()

