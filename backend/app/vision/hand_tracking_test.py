import os
import sys
import cv2
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

HAND_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "app",
    "vision",
    "hand_landmarker.task"
)


# ==========================================
# CHECK MODEL
# ==========================================

if not os.path.exists(HAND_MODEL_PATH):

    print()
    print("ERROR: Hand landmarker model not found:")
    print(HAND_MODEL_PATH)
    print()

    sys.exit()


# ==========================================
# MEDIAPIPE SETTINGS
# ==========================================

base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)


options = vision.HandLandmarkerOptions(
    base_options=base_options,

    # We want continuous tracking.
    running_mode=vision.RunningMode.VIDEO,

    # Maximum two hands.
    num_hands=2,

    # Detection confidence.
    min_hand_detection_confidence=0.6,

    # Hand presence confidence.
    min_hand_presence_confidence=0.6,

    # Tracking confidence.
    min_tracking_confidence=0.6
)


# ==========================================
# CREATE DETECTOR
# ==========================================

print()
print("Loading MediaPipe Hand Landmarker...")


detector = (
    vision.HandLandmarker
    .create_from_options(
        options
    )
)


print("Hand Landmarker loaded!")


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print()
    print("ERROR: Could not open camera.")
    print()

    detector.close()

    sys.exit()


# ==========================================
# CAMERA SETTINGS
# ==========================================

# Request a reasonable resolution.

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ==========================================
# START MESSAGE
# ==========================================

print()
print("==========================================")
print("       SIGNFLOW HAND TRACKING TEST")
print("==========================================")
print()
print("Show one or two hands to the camera.")
print()
print("Q = Quit")
print()


# ==========================================
# TIMESTAMP
# ==========================================

frame_timestamp = 0


# ==========================================
# COLORS
# ==========================================

# BGR format for OpenCV.

LANDMARK_COLOR = (
    0,
    255,
    0
)

CONNECTION_COLOR = (
    255,
    200,
    0
)

TEXT_COLOR = (
    255,
    255,
    255
)


# ==========================================
# HAND CONNECTIONS
# ==========================================

# MediaPipe hand landmark connections.

HAND_CONNECTIONS = [
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
    (13, 17)
]


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    # ======================================
    # READ CAMERA
    # ======================================

    success, frame = camera.read()


    if not success:

        print(
            "ERROR: Could not read camera frame."
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
    # CONVERT BGR -> RGB
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ======================================
    # CREATE MEDIAPIPE IMAGE
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
    # HAND COUNT
    # ======================================

    hand_count = len(
        result.hand_landmarks
    )


    # ======================================
    # DRAW EACH HAND
    # ======================================

    for hand_index, hand_landmarks in enumerate(
        result.hand_landmarks
    ):

        # ==================================
        # GET HANDEDNESS
        # ==================================

        handedness = (
            result.handedness[
                hand_index
            ][0]
        )


        hand_name = (
            handedness.category_name
        )


        hand_score = (
            handedness.score
        )


        # ==================================
        # CONVERT LANDMARKS TO PIXELS
        # ==================================

        points = []


        for landmark in hand_landmarks:

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


            points.append(
                (x, y)
            )


        # ==================================
        # DRAW CONNECTIONS
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
                    CONNECTION_COLOR,
                    2
                )


        # ==================================
        # DRAW LANDMARKS
        # ==================================

        for point in points:

            cv2.circle(
                frame,
                point,
                5,
                LANDMARK_COLOR,
                -1
            )


        # ==================================
        # HAND LABEL
        # ==================================

        if len(points) > 0:

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
                    wrist_y - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                TEXT_COLOR,
                2
            )


    # ======================================
    # TOP INFORMATION
    # ======================================

    cv2.putText(
        frame,
        f"Hands detected: {hand_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        TEXT_COLOR,
        2
    )


    cv2.putText(
        frame,
        "Q = Quit",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        TEXT_COLOR,
        2
    )


    # ======================================
    # STATUS
    # ======================================

    if hand_count == 0:

        status = "NO HAND"

    elif hand_count == 1:

        status = "ONE HAND"

    else:

        status = "TWO HANDS"


    cv2.putText(
        frame,
        status,
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2
    )


    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "SignFlow - Hand Tracking Test",
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
print("Hand tracking test stopped.")