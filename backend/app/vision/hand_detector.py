import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from landmarks import flatten_landmarks
from normalization import normalize_landmarks


MODEL_PATH = "backend/app/vision/hand_landmarker.task"


def start_hand_detection():

    # Create MediaPipe model
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    detector = vision.HandLandmarker.create_from_options(options)

    # Start camera
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        detector.close()
        return

    print("Hand detection started.")
    print("Press Q to quit.")

    frame_timestamp = 0

    while True:

        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame.")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hands
        result = detector.detect_for_video(
            mp_image,
            frame_timestamp
        )

        frame_timestamp += 1

        # Process detected hands
        if result.hand_landmarks:

            for hand in result.hand_landmarks:

                # Get 63 landmark values
                landmark_data = flatten_landmarks(hand)

                # Normalize landmarks
                normalized_data = normalize_landmarks(
                    landmark_data
                )

                # Test normalization
                if len(normalized_data) == 63:
                    print(
                        "Normalized landmarks:",
                        len(normalized_data)
                    )

                # Get frame dimensions
                height, width, _ = frame.shape

                # Draw landmarks
                for landmark in hand:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

        # Show camera
        cv2.imshow(
            "SignFlow - Hand Detection",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up
    camera.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_hand_detection()