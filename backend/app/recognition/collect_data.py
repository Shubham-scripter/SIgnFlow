import os
import sys
import cv2
import csv
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

sys.path.insert(0, PROJECT_ROOT)

HAND_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "app",
    "vision",
    "hand_landmarker.task"
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "data",
    "isl"
)

os.makedirs(DATASET_DIR, exist_ok=True)


# ==========================================
# GESTURE NAME
# ==========================================

gesture = input(
    "Enter gesture name (example: namaste): "
).strip().lower()

if not gesture:
    print("Gesture name cannot be empty.")
    sys.exit()


output_file = os.path.join(
    DATASET_DIR,
    f"{gesture}.csv"
)


# ==========================================
# FEATURE EXTRACTION
# ==========================================

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

        handedness = (
            result.handedness[
                hand_index
            ][0].category_name
        )

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
            (middle_mcp.x - wrist_x) ** 2 +
            (middle_mcp.y - wrist_y) ** 2 +
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
                offset +
                landmark_index * 3
            )

            features[index] = x
            features[index + 1] = y
            features[index + 2] = z

    return features


# ==========================================
# HAND CONNECTIONS
# ==========================================

HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17)
]


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

detector = vision.HandLandmarker.create_from_options(
    options
)


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    detector.close()

    sys.exit()


# ==========================================
# CSV
# ==========================================

file_exists = os.path.exists(
    output_file
)

csv_file = open(
    output_file,
    "a",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(
    csv_file
)


if not file_exists:

    header = []

    for hand in range(2):

        for landmark in range(21):

            header.append(
                f"hand{hand}_x{landmark}"
            )

            header.append(
                f"hand{hand}_y{landmark}"
            )

            header.append(
                f"hand{hand}_z{landmark}"
            )

    writer.writerow(header)


# ==========================================
# START
# ==========================================

print()
print("========================================")
print("       SIGNFLOW DATA COLLECTOR")
print("========================================")
print()
print("Gesture:", gesture)
print()
print("SPACE = Capture sample")
print("Q     = Quit")
print()
print("Move your hand around between samples.")
print("Change distance and angle slightly.")
print()


frame_timestamp = 0
sample_count = 0


# ==========================================
# CAMERA LOOP
# ==========================================

while True:

    success, frame = camera.read()

    if not success:

        print(
            "ERROR: Could not read camera."
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
    # FRAME SIZE
    # ======================================

    height, width, _ = frame.shape


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
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # ======================================
    # DETECT HANDS
    # ======================================

    result = detector.detect_for_video(
        mp_image,
        frame_timestamp
    )

    frame_timestamp += 1


    # ======================================
    # HAND COUNT
    # ======================================

    hand_count = len(
        result.hand_landmarks
    )


    # ======================================
    # DRAW HANDS
    # ======================================

    for hand_index, hand_landmarks in enumerate(
        result.hand_landmarks[:2]
    ):

        # ==================================
        # HANDEDNESS
        # ==================================

        handedness_data = (
            result.handedness[
                hand_index
            ][0]
        )

        hand_name = (
            handedness_data.category_name
        )

        hand_score = (
            handedness_data.score
        )


        # ==================================
        # LANDMARK PIXEL POSITIONS
        # ==================================

        points = []

        for landmark in hand_landmarks:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            points.append(
                (x, y)
            )


        # ==================================
        # DRAW CONNECTION LINES
        # ==================================

        for start_index, end_index in (
            HAND_CONNECTIONS
        ):

            if (
                start_index < len(points)
                and
                end_index < len(points)
            ):

                cv2.line(
                    frame,
                    points[start_index],
                    points[end_index],
                    (255, 200, 0),
                    2
                )


        # ==================================
        # DRAW LANDMARK DOTS
        # ==================================

        for point in points:

            cv2.circle(
                frame,
                point,
                5,
                (0, 255, 0),
                -1
            )


        # ==================================
        # HAND LABEL
        # ==================================

        if points:

            wrist_x, wrist_y = points[0]

            label = (
                f"{hand_name} "
                f"{hand_score * 100:.0f}%"
            )

            cv2.putText(
                frame,
                label,
                (
                    wrist_x + 10,
                    wrist_y - 15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )


    # ======================================
    # DISPLAY INFORMATION
    # ======================================

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"Samples: {sample_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Hands detected: {hand_count}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "SPACE = Capture | Q = Quit",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "SignFlow - Dataset Collector",
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


    # ======================================
    # CAPTURE SAMPLE
    # ======================================

    if key == ord(" "):

        if not result.hand_landmarks:

            print(
                "No hand detected. "
                "Show your hand first."
            )

            continue


        # ==================================
        # EXTRACT FEATURES
        # ==================================

        features = extract_features(
            result
        )


        # ==================================
        # SAVE
        # ==================================

        writer.writerow(
            features.tolist()
        )

        csv_file.flush()


        sample_count += 1


        print(
            f"Captured sample "
            f"{sample_count} | "
            f"Hands detected: "
            f"{hand_count}"
        )


    # ======================================
    # QUIT
    # ======================================

    if key == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

csv_file.close()

camera.release()

detector.close()

cv2.destroyAllWindows()


print()
print("Dataset collection stopped.")
print(
    f"Saved dataset: {output_file}"
)
print(
    f"Total samples: {sample_count}"
)