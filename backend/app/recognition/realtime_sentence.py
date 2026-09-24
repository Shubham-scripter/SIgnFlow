import os
import sys
import json
import time
import json
import threading
import queue
from collections import deque, Counter

import cv2
import numpy as np
import joblib
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from websockets.sync.client import connect

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
)


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models",
    "isl",
    "isl_static_model.pkl"
)


TRAINING_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models",
    "isl",
    "isl_training_data.npz"
)


CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models",
    "isl",
    "isl_model_config.json"
)


HAND_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "app",
    "vision",
    "hand_landmarker.task"
)


# ============================================================
# LANGUAGE MODULE
# ============================================================

LANGUAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "app",
    "language"
)

sys.path.insert(
    0,
    LANGUAGE_DIR
)


from sentence import process_text


# ============================================================
# WORD STREAM
# ============================================================

from word_stream import WordStream


# ============================================================
# SETTINGS
# ============================================================

TARGET_LANGUAGE = "Hindi"

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Recognition uses a smaller processing frame for speed.
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360

# Send a smooth, reasonably high-quality binary JPEG preview.
WEBCAM_SEND_EVERY_N_FRAMES = 2
WEBCAM_PREVIEW_WIDTH = 960
WEBCAM_JPEG_QUALITY = 60

HISTORY_SIZE = 8

STABLE_REQUIRED = 5

UNKNOWN_RESET_FRAMES = 8

# ============================================================
# WEBSOCKET
# ============================================================

WEBSOCKET_URL = "ws://127.0.0.1:8000/ws"


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading SignFlow model...")

model = joblib.load(
    MODEL_PATH
)

print(
    "Random Forest loaded."
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
    "Training data loaded."
)

print(
    "Training samples:",
    len(X_train)
)


# ============================================================
# LOAD CONFIG
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:

    config = json.load(f)


classes = config.get(
    "classes",
    list(model.classes_)
)


REJECTION_THRESHOLD = config.get(
    "rejection_threshold",
    4.3207
)


print(
    "Unknown threshold:",
    REJECTION_THRESHOLD
)


# ============================================================
# CALCULATE HAND COUNT
# ============================================================

def calculate_hand_count(features):

    left_hand = features[:63]

    right_hand = features[63:126]


    left_present = (
        np.linalg.norm(left_hand) > 0.01
    )


    right_present = (
        np.linalg.norm(right_hand) > 0.01
    )


    return (
        int(left_present)
        +
        int(right_present)
    )


# ============================================================
# BUILD EXPECTED HAND COUNTS
# ============================================================

def build_expected_hand_counts():

    expected = {}


    for gesture in classes:

        mask = (
            y_train == gesture
        )


        gesture_samples = (
            X_train[mask]
        )


        counts = []


        for sample in gesture_samples:

            counts.append(
                calculate_hand_count(
                    sample
                )
            )


        if counts:

            expected[gesture] = (
                Counter(
                    counts
                ).most_common(1)[0][0]
            )


    return expected


EXPECTED_HAND_COUNTS = (
    build_expected_hand_counts()
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
                landmark.x - wrist_x
            ) / scale


            y = (
                landmark.y - wrist_y
            ) / scale


            z = (
                landmark.z - wrist_z
            ) / scale


            index = (
                offset
                +
                landmark_index * 3
            )


            features[index] = x

            features[index + 1] = y

            features[index + 2] = z


    return features


# ============================================================
# GET PREDICTION
# ============================================================

def get_prediction(features):

    hand_count = (
        calculate_hand_count(
            features
        )
    )


    # --------------------------------------------------------
    # No hand
    # --------------------------------------------------------

    if hand_count == 0:

        return (
            "No hand detected",
            0.0
        )


    # --------------------------------------------------------
    # Random Forest prediction
    # --------------------------------------------------------

    prediction = model.predict(
        [features]
    )[0]


    # --------------------------------------------------------
    # Check expected hand count
    # --------------------------------------------------------

    expected_count = (
        EXPECTED_HAND_COUNTS.get(
            prediction
        )
    )


    if (
        expected_count is not None
        and
        hand_count != expected_count
    ):

        return (
            "Unknown",
            999.0
        )


    # --------------------------------------------------------
    # Distance from training samples
    # --------------------------------------------------------

    mask = (
        y_train == prediction
    )


    class_samples = (
        X_train[mask]
    )


    if len(class_samples) == 0:

        return (
            "Unknown",
            999.0
        )


    distances = np.linalg.norm(
        class_samples - features,
        axis=1
    )


    minimum_distance = float(
        np.min(distances)
    )


    # --------------------------------------------------------
    # Unknown rejection
    # --------------------------------------------------------

    if (
        minimum_distance
        >
        REJECTION_THRESHOLD
    ):

        return (
            "Unknown",
            minimum_distance
        )


    return (
        prediction,
        minimum_distance
    )


# ============================================================
# DRAW LANDMARKS
# ============================================================

def draw_landmarks(
    frame,
    result
):

    if not result.hand_landmarks:

        return


    height, width, _ = (
        frame.shape
    )


    for hand_index, hand_landmarks in enumerate(
        result.hand_landmarks
    ):

        # ----------------------------------------------------
        # Draw skeleton
        # ----------------------------------------------------

        for start, end in HAND_CONNECTIONS:

            start_point = (
                hand_landmarks[start]
            )

            end_point = (
                hand_landmarks[end]
            )


            x1 = int(
                start_point.x * width
            )

            y1 = int(
                start_point.y * height
            )


            x2 = int(
                end_point.x * width
            )

            y2 = int(
                end_point.y * height
            )


            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


        # ----------------------------------------------------
        # Draw landmarks
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
                (0, 255, 0),
                -1
            )


        # ----------------------------------------------------
        # Draw handedness
        # ----------------------------------------------------

        handedness = (
            result.handedness[
                hand_index
            ][0]
        )


        label = (
            f"{handedness.category_name} "
            f"{handedness.score:.2f}"
        )


        wrist = (
            hand_landmarks[0]
        )


        label_x = int(
            wrist.x * width
        )


        label_y = int(
            wrist.y * height
        ) - 15


        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


# ============================================================
# WEBSOCKET HELPER
# ============================================================

def send_websocket_message(websocket, message):
    if websocket is None:
        return

    try:
        websocket.send(json.dumps(message))
    except Exception as e:
        print("WebSocket send error:", e)


def send_websocket_frame(websocket, frame):
    """Send the processed camera frame as binary JPEG bytes.

    Binary frames avoid the large Base64 + JSON overhead that was
    causing the frontend preview to lag.
    """

    if websocket is None:
        return

    try:
        height, width = frame.shape[:2]
        target_width = WEBCAM_PREVIEW_WIDTH

        if width > target_width:
            scale = target_width / width
            frame = cv2.resize(
                frame,
                (target_width, int(height * scale)),
                interpolation=cv2.INTER_AREA
            )

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, WEBCAM_JPEG_QUALITY]
        )

        if not success:
            return

        websocket.send(encoded.tobytes())

    except Exception as e:
        print("WebSocket frame send error:", e)


def receive_control_messages(websocket, control_queue):
    """Receive frontend controls without blocking recognition."""

    if websocket is None:
        return

    while True:
        try:
            message = websocket.recv()

            if isinstance(message, bytes):
                continue

            data = json.loads(message)

            if data.get("type") == "control":
                control_queue.put(data.get("action"))

        except Exception:
            break


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "Starting SignFlow realtime recognition..."
    )
    print()

    # --------------------------------------------------------
    # WebSocket connection
    # --------------------------------------------------------

    websocket = None
    control_queue = queue.Queue()
    control_thread = None

    try:
        print("Connecting to SignFlow backend...")
        websocket = connect(WEBSOCKET_URL, max_size=4 * 1024 * 1024)
        print("Connected to SignFlow backend!")

        # Tell FastAPI that this socket belongs to the recognition process.
        websocket.send(json.dumps({
            "type": "register",
            "role": "recognition"
        }))

        control_thread = threading.Thread(
            target=receive_control_messages,
            args=(websocket, control_queue),
            daemon=True
        )
        control_thread.start()

    except Exception as e:
        print("WebSocket connection failed:", e)
        print("Recognition will continue without WebSocket.")

    print()


    print("Controls:")

    print(
        "  ENTER     = finish sentence + translate + speak"
    )

    print(
        "  BACKSPACE = remove last word"
    )

    print(
        "  C         = clear sentence"
    )

    print(
        "  Q         = quit"
    )

    print()


    # --------------------------------------------------------
    # Word Stream
    # --------------------------------------------------------

    word_stream = WordStream(
        stable_frames=STABLE_REQUIRED
    )


    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    base_options = (
        python.BaseOptions(
            model_asset_path=HAND_MODEL_PATH
        )
    )


    options = (
        vision.HandLandmarkerOptions(

            base_options=base_options,

            running_mode=vision.RunningMode.VIDEO,

            num_hands=2,

            min_hand_detection_confidence=0.5,

            min_hand_presence_confidence=0.5,

            min_tracking_confidence=0.5

        )
    )


    detector = (
        vision.HandLandmarker.create_from_options(
            options
        )
    )


    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    # Keep the camera queue tiny so old frames do not build up and
    # create visible latency.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


    if not cap.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        detector.close()

        return


    # --------------------------------------------------------
    # Prediction history
    # --------------------------------------------------------

    prediction_history = deque(
        maxlen=HISTORY_SIZE
    )


    unknown_frames = 0


    last_prediction = (
        "Waiting..."
    )


    last_distance = 0.0


    start_time = time.time()

    # Camera preview is intentionally sent much less often than
    # recognition frames. Recognition itself still runs every frame.
    frame_counter = 0


    # ========================================================
    # CAMERA LOOP
    # ========================================================

    running = True

    while running:

        success, frame = (
            cap.read()
        )


        if not success:

            print(
                "Could not read camera frame."
            )

            break


        # ----------------------------------------------------
        # Mirror camera
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # Create a smaller processing frame for MediaPipe.
        # The original 720p frame is kept for the frontend so the
        # displayed video remains sharp.
        # ----------------------------------------------------

        processing_frame = cv2.resize(
            frame,
            (PROCESS_WIDTH, PROCESS_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        rgb = cv2.cvtColor(
            processing_frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp_ms = int(
            (
                time.time()
                -
                start_time
            )
            * 1000
        )


        # ----------------------------------------------------
        # Detect hands
        # ----------------------------------------------------

        result = (
            detector.detect_for_video(
                mp_image,
                timestamp_ms
            )
        )


        # ----------------------------------------------------
        # Draw landmarks
        # ----------------------------------------------------

        draw_landmarks(
            frame,
            result
        )


        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features = (
            extract_features(
                result
            )
        )


        # ----------------------------------------------------
        # Recognize gesture
        # ----------------------------------------------------

        prediction, distance = (
            get_prediction(
                features
            )
        )


        last_prediction = prediction

        last_distance = distance


        # ----------------------------------------------------
        # Unknown / no hand
        # ----------------------------------------------------

        if (
            prediction == "Unknown"
            or
            prediction == "No hand detected"
        ):

            unknown_frames += 1

            prediction_history.clear()


        else:

            unknown_frames = 0

            prediction_history.append(
                prediction
            )


        # ----------------------------------------------------
        # Find stable prediction
        # ----------------------------------------------------

        stable_word = None


        if (
            len(prediction_history)
            >=
            STABLE_REQUIRED
        ):

            most_common = (
                Counter(
                    prediction_history
                ).most_common(1)[0]
            )


            candidate_word = (
                most_common[0]
            )


            candidate_count = (
                most_common[1]
            )


            if (
                candidate_count
                >=
                STABLE_REQUIRED
            ):

                stable_word = (
                    candidate_word
                )


        # ----------------------------------------------------
        # FRONTEND CONTROLS
        # ----------------------------------------------------

        while not control_queue.empty():

            try:
                action = control_queue.get_nowait()
            except queue.Empty:
                break

            if action == "enter":

                words = word_stream.get_words()

                if words:
                    print()
                    print("================================")
                    print("Recognized words:", words)
                    print("Sending to language pipeline...")
                    print("================================")

                    try:
                        sentence = " ".join(words)
                        translated = process_text(
                            words,
                            TARGET_LANGUAGE
                        )

                        send_websocket_message(
                            websocket,
                            {
                                "type": "sentence_result",
                                "sentence": sentence,
                                "translation": translated
                            }
                        )

                        print("Translation complete:")
                        print(translated)

                    except Exception as e:

                        print("Language pipeline error:", e)

                        send_websocket_message(
                            websocket,
                            {
                                "type": "error",
                                "message": "Language pipeline error",
                                "details": str(e)
                            }
                        )

                    word_stream.clear()
                    prediction_history.clear()
                    unknown_frames = 0

            elif action == "backspace":

                removed_word = word_stream.remove_last_word()
                prediction_history.clear()

                print(
                    "Removed:" if removed_word else "No word to remove.",
                    removed_word if removed_word else ""
                )

                send_websocket_message(
                    websocket,
                    {
                        "type": "sentence_state",
                        "sentence": word_stream.get_sentence()
                    }
                )

            elif action == "clear":

                word_stream.clear()
                prediction_history.clear()
                unknown_frames = 0

                print("Sentence cleared.")

                send_websocket_message(
                    websocket,
                    {
                        "type": "sentence_state",
                        "sentence": ""
                    }
                )

            elif action == "quit":

                print("Stop requested from frontend.")
                running = False
                break


        if not running:
            break


        # ----------------------------------------------------
        # Send to WordStream
        # ----------------------------------------------------

        if stable_word is not None:

            new_word = (
                word_stream.process(
                    stable_word
                )
            )


            if new_word is not None:

                print(
                    "New word:",
                    new_word
                )

                current_sentence_for_websocket = (
                    word_stream.get_sentence()
                )

                send_websocket_message(
                    websocket,
                    {
                        "type": "sign",
                        "sign": new_word,
                        "sentence": current_sentence_for_websocket
                    }
                )


        else:

            word_stream.process(
                None
            )


        # ----------------------------------------------------
        # Reset recognition after hands disappear
        # ----------------------------------------------------

        if (
            unknown_frames
            >=
            UNKNOWN_RESET_FRAMES
        ):

            prediction_history.clear()


        # ----------------------------------------------------
        # Current sentence
        # ----------------------------------------------------

        current_sentence = (
            word_stream.get_sentence()
        )


        if not current_sentence:

            current_sentence = "..."


        # ====================================================
        # UI PANEL
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 155),
            (0, 0, 0),
            -1
        )


        # ----------------------------------------------------
        # Current sign
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Sign: {last_prediction}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Distance
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Distance: {last_distance:.2f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Sentence
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Sentence:",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            current_sentence,
            (130, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Controls
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "SignFlow AI Camera",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1
        )


        # ----------------------------------------------------
        # Send camera frame to frontend.
        # This is binary JPEG, not Base64 JSON.
        # ----------------------------------------------------

        frame_counter += 1

        if frame_counter % WEBCAM_SEND_EVERY_N_FRAMES == 0:
            send_websocket_frame(
                websocket,
                frame
            )


        # ----------------------------------------------------
        # Keep the frontend recognition panel live.
        # ----------------------------------------------------

        if frame_counter % 5 == 0:
            send_websocket_message(
                websocket,
                {
                    "type": "recognition_live",
                    "sign": last_prediction,
                    "distance": round(float(last_distance), 3),
                    "sentence": word_stream.get_sentence()
                }
            )


    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    detector.close()

    if websocket is not None:
        try:
            websocket.close()
            print("WebSocket disconnected.")
        except Exception as e:
            print("WebSocket close error:", e)


    print()
    print(
        "SignFlow stopped."
    )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()