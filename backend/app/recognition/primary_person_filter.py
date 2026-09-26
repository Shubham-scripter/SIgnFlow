import math
from types import SimpleNamespace

import cv2
import mediapipe as mp


class PrimaryUserHandFilter:
    """
    Primary-person hand filter for SignFlow.

    The filter combines:
        1. MediaPipe Pose wrist position
        2. Hand-to-pose wrist distance
        3. Hand bounding-box size

    The goal is to keep hands belonging to the primary person while
    rejecting hands that are:
        - too far from the primary person's wrist
        - too small in the image (usually farther from camera)
        - associated with an unreliable primary wrist

    Important:
        Pose num_poses=1 means we track one primary pose.
        HandLandmarker remains responsible for detecting hands.
        This class only filters the detected hands.
    """

    def __init__(
        self,
        pose_model_path,
        max_wrist_distance=0.30,
        min_hand_size=0.075,
        min_wrist_visibility=0.25,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.pose_model_path = pose_model_path

        self.max_wrist_dist = float(max_wrist_distance)
        self.min_hand_size = float(min_hand_size)
        self.min_wrist_visibility = float(min_wrist_visibility)

        if not pose_model_path:
            raise ValueError(
                "Pose model path was not provided."
            )

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = (
            mp.tasks.vision.PoseLandmarkerOptions
        )
        RunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=pose_model_path
            ),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=(
                min_detection_confidence
            ),
            min_pose_presence_confidence=(
                min_detection_confidence
            ),
            min_tracking_confidence=(
                min_tracking_confidence
            ),
        )

        self.pose_landmarker = (
            PoseLandmarker.create_from_options(options)
        )

        self.last_status = {
            "left": "NOT DETECTED",
            "right": "NOT DETECTED",
            "pose": "READY",
        }

    # ---------------------------------------------------------
    # Basic geometry
    # ---------------------------------------------------------

    @staticmethod
    def _distance(pt1, pt2):
        """
        Euclidean distance between two normalized MediaPipe
        landmarks.
        """

        return math.hypot(
            pt1.x - pt2.x,
            pt1.y - pt2.y,
        )

    @staticmethod
    def _visibility(landmark):
        """
        Return MediaPipe visibility as a float.

        Some MediaPipe landmarks may not expose visibility.
        In that case use 1.0.
        """

        visibility = getattr(
            landmark,
            "visibility",
            1.0,
        )

        if visibility is None:
            visibility = 1.0

        try:
            return float(visibility)
        except (TypeError, ValueError):
            return 1.0

    @staticmethod
    def _hand_size(hand_landmarks):
        """
        Calculate normalized hand bounding-box size.

        Returns the largest dimension of the hand bounding box.

        Example:

            size = 0.30

        means the detected hand occupies roughly 30% of
        the image along its largest dimension.

        Smaller values generally correspond to hands farther
        from the camera.
        """

        if not hand_landmarks:
            return 0.0

        xs = [
            float(landmark.x)
            for landmark in hand_landmarks
        ]

        ys = [
            float(landmark.y)
            for landmark in hand_landmarks
        ]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        return max(width, height)

    @staticmethod
    def _hand_center(hand_landmarks):
        """
        Calculate the center of the detected hand.
        """

        if not hand_landmarks:
            return None

        x = sum(
            float(landmark.x)
            for landmark in hand_landmarks
        ) / len(hand_landmarks)

        y = sum(
            float(landmark.y)
            for landmark in hand_landmarks
        ) / len(hand_landmarks)

        return SimpleNamespace(
            x=x,
            y=y,
        )

    # ---------------------------------------------------------
    # Pose
    # ---------------------------------------------------------

    def _get_pose_wrists(
        self,
        frame_bgr,
        timestamp_ms,
    ):
        """
        Detect the primary person's pose and return:

            left wrist
            right wrist
        """

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        results = self.pose_landmarker.detect_for_video(
            mp_image,
            int(timestamp_ms),
        )

        if not results.pose_landmarks:
            return None, None

        # num_poses=1
        pose = results.pose_landmarks[0]

        if len(pose) <= 16:
            return None, None

        # MediaPipe Pose:
        #
        # 15 = left wrist
        # 16 = right wrist
        left_wrist = pose[15]
        right_wrist = pose[16]

        return left_wrist, right_wrist

    # ---------------------------------------------------------
    # Filtering
    # ---------------------------------------------------------

    def filter_result(
        self,
        frame_bgr,
        result,
        timestamp_ms,
    ):
        """
        Filter an existing MediaPipe HandLandmarker result.

        Returns an object exposing:

            result.hand_landmarks
            result.handedness

        so the rest of SignFlow can continue using the result
        exactly as before.
        """

        # -----------------------------------------------------
        # No detected hands
        # -----------------------------------------------------

        if not result.hand_landmarks:
            self.last_status = {
                "left": "NOT DETECTED",
                "right": "NOT DETECTED",
                "pose": "NO HANDS",
            }

            return SimpleNamespace(
                hand_landmarks=[],
                handedness=[],
            )

        # -----------------------------------------------------
        # Detect primary person's pose
        # -----------------------------------------------------

        (
            pose_left_wrist,
            pose_right_wrist,
        ) = self._get_pose_wrists(
            frame_bgr,
            timestamp_ms,
        )

        # -----------------------------------------------------
        # If primary pose is unavailable:
        #
        # STRICT MODE:
        # reject all hands.
        #
        # This prevents another person's hands from being
        # accidentally passed to the ISL classifier.
        # -----------------------------------------------------

        if (
            pose_left_wrist is None
            or pose_right_wrist is None
        ):
            self.last_status = {
                "left": "POSE UNAVAILABLE - REJECTED",
                "right": "POSE UNAVAILABLE - REJECTED",
                "pose": "UNAVAILABLE",
            }

            return SimpleNamespace(
                hand_landmarks=[],
                handedness=[],
            )

        # -----------------------------------------------------
        # Prepare output
        # -----------------------------------------------------

        filtered_landmarks = []
        filtered_handedness = []

        self.last_status = {
            "left": "NOT DETECTED",
            "right": "NOT DETECTED",
            "pose": "PRIMARY POSE",
        }

        # -----------------------------------------------------
        # Process every detected hand
        # -----------------------------------------------------

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            # Safety check
            if hand_index >= len(result.handedness):
                continue

            categories = result.handedness[hand_index]

            if not categories:
                continue

            handedness = categories[0].category_name

            if not hand_landmarks:
                continue

            # -------------------------------------------------
            # Determine corresponding primary-person wrist
            # -------------------------------------------------

            if handedness == "Left":

                pose_wrist = pose_left_wrist
                status_key = "left"

            elif handedness == "Right":

                pose_wrist = pose_right_wrist
                status_key = "right"

            else:

                print(
                    "[PRIMARY HAND FILTER] "
                    "Unknown handedness rejected"
                )

                continue

            # -------------------------------------------------
            # Pose wrist visibility
            # -------------------------------------------------

            visibility = self._visibility(
                pose_wrist
            )

            # We do NOT automatically accept the hand when
            # visibility is low.
            #
            # Instead we use a softer threshold of 0.25.
            #
            # If it is below that, reject it because we cannot
            # reliably associate the hand with the primary
            # person's body.
            # -------------------------------------------------

            if visibility < self.min_wrist_visibility:

                self.last_status[status_key] = (
                    "PRIMARY WRIST LOW VISIBILITY "
                    f"({visibility:.2f})"
                )

                print(
                    "[PRIMARY HAND FILTER]",
                    f"{handedness} hand rejected - "
                    "primary wrist visibility too low "
                    f"({visibility:.2f} < "
                    f"{self.min_wrist_visibility:.2f})",
                )

                continue

            # -------------------------------------------------
            # Hand wrist
            # -------------------------------------------------

            hand_wrist = hand_landmarks[0]

            # -------------------------------------------------
            # Distance from primary pose wrist
            # -------------------------------------------------

            wrist_distance = self._distance(
                pose_wrist,
                hand_wrist,
            )

            # -------------------------------------------------
            # Hand size
            # -------------------------------------------------

            hand_size = self._hand_size(
                hand_landmarks
            )

            # -------------------------------------------------
            # Hand center
            # -------------------------------------------------

            hand_center = self._hand_center(
                hand_landmarks
            )

            # -------------------------------------------------
            # Decision
            # -------------------------------------------------

            wrist_ok = (
                wrist_distance
                <= self.max_wrist_dist
            )

            size_ok = (
                hand_size
                >= self.min_hand_size
            )

            # -------------------------------------------------
            # Reject if wrist is too far
            # -------------------------------------------------

            if not wrist_ok:

                self.last_status[status_key] = (
                    f"REJECTED - WRIST "
                    f"{wrist_distance:.3f}"
                )

                print(
                    "[PRIMARY HAND FILTER]",
                    f"{handedness} hand rejected - "
                    f"wrist distance "
                    f"{wrist_distance:.3f} > "
                    f"{self.max_wrist_dist:.3f}",
                )

                continue

            # -------------------------------------------------
            # Reject if hand is too small
            #
            # This is the important filter for people farther
            # away from the camera.
            # -------------------------------------------------

            if not size_ok:

                self.last_status[status_key] = (
                    f"REJECTED - SMALL HAND "
                    f"{hand_size:.3f}"
                )

                print(
                    "[PRIMARY HAND FILTER]",
                    f"{handedness} hand rejected - "
                    f"hand too small/far: "
                    f"size={hand_size:.3f} < "
                    f"{self.min_hand_size:.3f}",
                )

                continue

            # -------------------------------------------------
            # ACCEPT
            # -------------------------------------------------

            filtered_landmarks.append(
                hand_landmarks
            )

            filtered_handedness.append(
                categories
            )

            self.last_status[status_key] = (
                f"OK "
                f"(wrist={wrist_distance:.3f}, "
                f"size={hand_size:.3f}, "
                f"visibility={visibility:.2f})"
            )

            print(
                "[PRIMARY HAND FILTER]",
                f"{handedness} hand accepted - "
                f"wrist={wrist_distance:.3f}, "
                f"size={hand_size:.3f}, "
                f"visibility={visibility:.2f}",
            )

        # -----------------------------------------------------
        # Return filtered result
        # -----------------------------------------------------

        return SimpleNamespace(
            hand_landmarks=filtered_landmarks,
            handedness=filtered_handedness,
        )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def close(self):

        if getattr(
            self,
            "pose_landmarker",
            None,
        ) is not None:

            self.pose_landmarker.close()

            self.pose_landmarker = None