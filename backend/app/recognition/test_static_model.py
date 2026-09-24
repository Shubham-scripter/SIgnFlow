import os
import sys
import json
import cv2
import joblib
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# PROJECT PATHS
# ==========================================

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

HAND_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "app",
    "vision",
    "hand_landmarker.task"
)


# ==========================================
# SETTINGS
# ==========================================

FEATURE_COUNT = 126

# Minimum number of hands required for
# Namaste-style two-hand recognition.
#
# This is NOT hardcoded into the gesture
# name. It is learned from the dataset.

HAND_FEATURE_THRESHOLD = 0.01


# ==========================================
# LOAD MODEL FILES
# ==========================================

print()
print("Loading SignFlow model...")


if not os.path.exists(MODEL_PATH):

    print()
    print("ERROR: Model not found:")
    print(MODEL_PATH)

    sys.exit()


if not os.path.exists(TRAINING_DATA_PATH):

    print()
    print("ERROR: Training data not found:")
    print(TRAINING_DATA_PATH)

    sys.exit()


if not os.path.exists(CONFIG_PATH):

    print()
    print("ERROR: Model configuration not found:")
    print(CONFIG_PATH)

    sys.exit()


# ==========================================
# LOAD RANDOM FOREST
# ==========================================

model = joblib.load(
    MODEL_PATH
)


# ==========================================
# LOAD TRAINING DATA
# ==========================================

training_data = np.load(
    TRAINING_DATA_PATH
)

training_samples = training_data[
    "X"
]

training_labels = training_data[
    "y"
]


# ==========================================
# LOAD CONFIGURATION
# ==========================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:

    config = json.load(
        file
    )


threshold = float(
    config["threshold"]
)

classes = config[
    "classes"
]


# ==========================================
# MODEL INFORMATION
# ==========================================

print()
print("Model loaded!")

print()
print("Gestures:")

for gesture in classes:

    print(
        f"  - {gesture}"
    )

print()

print(
    f"Training samples: "
    f"{len(training_samples)}"
)

print(
    f"Distance threshold: "
    f"{threshold:.4f}"
)


# ==========================================
# HAND COUNT FROM FEATURES
# ==========================================

def get_hand_count_from_features(
    features
):

    """
    Determine whether the feature vector
    contains 0, 1, or 2 hands.

    126 total features:

        Left hand  = 63
        Right hand = 63
    """

    left_hand = features[
        0:63
    ]

    right_hand = features[
        63:126
    ]


    left_norm = np.linalg.norm(
        left_hand
    )

    right_norm = np.linalg.norm(
        right_hand
    )


    left_present = (
        left_norm >
        HAND_FEATURE_THRESHOLD
    )

    right_present = (
        right_norm >
        HAND_FEATURE_THRESHOLD
    )


    count = 0


    if left_present:

        count += 1


    if right_present:

        count += 1


    return count


# ==========================================
# LEARN EXPECTED HAND COUNT
# ==========================================

def calculate_expected_hand_counts():

    """
    Learn how many hands each gesture
    normally uses from its dataset.

    This keeps the system universal.

    Example:

        namaste -> 2 hands
        hello   -> 1 hand
    """

    expected_counts = {}


    for gesture in classes:

        gesture_samples = (
            training_samples[
                training_labels == gesture
            ]
        )


        if len(
            gesture_samples
        ) == 0:

            continue


        hand_counts = []


        for sample in gesture_samples:

            count = (
                get_hand_count_from_features(
                    sample
                )
            )

            hand_counts.append(
                count
            )


        if len(
            hand_counts
        ) == 0:

            continue


        values, counts = np.unique(
            hand_counts,
            return_counts=True
        )


        most_common_index = (
            np.argmax(
                counts
            )
        )


        expected_count = int(
            values[
                most_common_index
            ]
        )


        expected_counts[
            gesture
        ] = expected_count


    return expected_counts


expected_hand_counts = (
    calculate_expected_hand_counts()
)


# ==========================================
# SHOW EXPECTED HAND COUNTS
# ==========================================

print()

print(
    "Expected hand count:"
)


for gesture in classes:

    count = expected_hand_counts.get(
        gesture,
        0
    )

    print(
        f"  {gesture}: "
        f"{count} hand(s)"
    )


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(
    result
):

    features = np.zeros(
        FEATURE_COUNT,
        dtype=np.float32
    )


    if not result.hand_landmarks:

        return features


    for hand_index, hand_landmarks in enumerate(
        result.hand_landmarks[:2]
    ):

        handedness = (
            result.handedness[
                hand_index
            ][0].category_name
        )


        # Left hand occupies
        # features 0-62.

        if handedness == "Left":

            offset = 0

        # Right hand occupies
        # features 63-125.

        else:

            offset = 63


        # ==================================
        # WRIST
        # ==================================

        wrist = hand_landmarks[
            0
        ]


        wrist_x = wrist.x
        wrist_y = wrist.y
        wrist_z = wrist.z


        # ==================================
        # SCALE
        # ==================================

        middle_mcp = (
            hand_landmarks[
                9
            ]
        )


        scale = np.sqrt(
            (middle_mcp.x - wrist_x) ** 2
            +
            (middle_mcp.y - wrist_y) ** 2
            +
            (middle_mcp.z - wrist_z) ** 2
        )


        if scale < 1e-6:

            scale = 1.0


        # ==================================
        # LANDMARKS
        # ==================================

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


            features[
                index
            ] = x


            features[
                index + 1
            ] = y


            features[
                index + 2
            ] = z


    return features


# ==========================================
# DISTANCE CALCULATION
# ==========================================

def calculate_distance(
    features,
    gesture
):

    gesture_samples = (
        training_samples[
            training_labels == gesture
        ]
    )


    if len(
        gesture_samples
    ) == 0:

        return float(
            "inf"
        )


    distances = np.linalg.norm(
        gesture_samples
        -
        features,
        axis=1
    )


    return float(
        np.min(
            distances
        )
    )


# ==========================================
# FIND CLOSEST GESTURE
# ==========================================

def find_closest_gesture(
    features
):

    best_gesture = None

    best_distance = float(
        "inf"
    )


    for gesture in classes:

        distance = (
            calculate_distance(
                features,
                gesture
            )
        )


        if distance < best_distance:

            best_distance = (
                distance
            )

            best_gesture = (
                gesture
            )


    return (
        best_gesture,
        best_distance
    )


# ==========================================
# MEDIAPIPE
# ==========================================

base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)


options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


detector = (
    vision.HandLandmarker
    .create_from_options(
        options
    )
)


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(
    0
)


if not camera.isOpened():

    print()
    print(
        "ERROR: Could not open camera."
    )

    detector.close()

    sys.exit()


# ==========================================
# START
# ==========================================

print()

print(
    "=============================="
)

print(
    " SIGNFLOW LIVE TEST"
)

print(
    "=============================="
)

print()

print(
    "Recognition mode:"
)

print(
    "Distance-based recognition"
)

print(
    "Random Forest confidence: DISABLED"
)

print()

print(
    "Press Q to quit."
)

print()


# ==========================================
# TIMESTAMP
# ==========================================

frame_timestamp = 0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = (
        camera.read()
    )


    if not success:

        print(
            "Camera error."
        )

        break


    # ======================================
    # MIRROR CAMERA
    # ======================================

    frame = cv2.flip(
        frame,
        1
    )


    # ======================================
    # RGB
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ======================================
    # MEDIAPIPE IMAGE
    # ======================================

    mp_image = mp.Image(
        image_format=(
            mp.ImageFormat.SRGB
        ),
        data=rgb_frame
    )


    # ======================================
    # DETECT HANDS
    # ======================================

    result = (
        detector.detect_for_video(
            mp_image,
            frame_timestamp
        )
    )


    frame_timestamp += 1


    # ======================================
    # DRAW LANDMARKS
    # ======================================

    if result.hand_landmarks:

        height, width, _ = (
            frame.shape
        )


        for hand in (
            result.hand_landmarks
        ):

            for landmark in hand:

                x = int(
                    landmark.x
                    *
                    width
                )

                y = int(
                    landmark.y
                    *
                    height
                )


                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )


    # ======================================
    # PREDICTION
    # ======================================

    if not result.hand_landmarks:

        # ----------------------------------
        # NO HAND
        # ----------------------------------

        prediction = (
            "No hand detected"
        )

        distance = 0.0

        detected_hand_count = 0


    else:

        # ----------------------------------
        # COUNT HANDS
        # ----------------------------------

        detected_hand_count = len(
            result.hand_landmarks
        )


        # ----------------------------------
        # EXTRACT FEATURES
        # ----------------------------------

        features = (
            extract_features(
                result
            )
        )


        # ----------------------------------
        # FIND CLOSEST GESTURE
        # ----------------------------------

        closest_gesture, distance = (
            find_closest_gesture(
                features
            )
        )


        # ----------------------------------
        # EXPECTED HAND COUNT
        # ----------------------------------

        expected_hand_count = (
            expected_hand_counts.get(
                closest_gesture,
                0
            )
        )


        # ----------------------------------
        # HAND COUNT CHECK
        # ----------------------------------

        correct_hand_count = (
            detected_hand_count
            ==
            expected_hand_count
        )


        # ----------------------------------
        # DISTANCE CHECK
        # ----------------------------------

        close_enough = (
            distance
            <=
            threshold
        )


        # ----------------------------------
        # FINAL DECISION
        # ----------------------------------

        if (
            correct_hand_count
            and
            close_enough
        ):

            prediction = (
                closest_gesture
            )

        else:

            prediction = (
                "Unknown"
            )


    # ======================================
    # DISPLAY MAIN RESULT
    # ======================================

    if prediction == "No hand detected":

        display_text = (
            "NO HAND DETECTED"
        )


    elif prediction == "Unknown":

        display_text = (
            "UNKNOWN"
        )


    else:

        display_text = (
            prediction.upper()
        )


    cv2.putText(
        frame,
        display_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3
    )


    # ======================================
    # DISPLAY HAND COUNT
    # ======================================

    cv2.putText(
        frame,
        f"Hands: {detected_hand_count}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ======================================
    # DISPLAY DISTANCE
    # ======================================

    if result.hand_landmarks:

        distance_text = (
            f"Distance: "
            f"{distance:.3f}"
        )

    else:

        distance_text = (
            "Distance: ---"
        )


    cv2.putText(
        frame,
        distance_text,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ======================================
    # DISPLAY THRESHOLD
    # ======================================

    cv2.putText(
        frame,
        f"Threshold: {threshold:.3f}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ======================================
    # DISPLAY INSTRUCTION
    # ======================================

    cv2.putText(
        frame,
        "Q = Quit",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "SignFlow - Live Test",
        frame
    )


    # ======================================
    # KEYBOARD
    # ======================================

    key = (
        cv2.waitKey(1)
        &
        0xFF
    )


    if key == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

detector.close()

cv2.destroyAllWindows()


print()

print(
    "Live test stopped."
)