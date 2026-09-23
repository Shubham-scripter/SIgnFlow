import numpy as np


def extract_isl_features(result):
    """
    Convert MediaPipe hand detection result into
    the exact 84-feature format expected by the
    pretrained ISL model.

    Left hand:
        21 landmarks × (x, y) = 42

    Right hand:
        21 landmarks × (x, y) = 42

    Total:
        84 position features
    """

    features = np.zeros(84, dtype=np.float32)

    if not result.hand_landmarks:
        return features

    for idx, hand_landmarks in enumerate(result.hand_landmarks):

        if idx >= 2:
            break

        # MediaPipe Tasks API uses handedness separately.
        handedness = result.handedness[idx][0].category_name

        if handedness == "Left":
            offset = 0
        else:
            offset = 42

        wrist = hand_landmarks[0]
        mcp = hand_landmarks[9]

        wrist_x = wrist.x
        wrist_y = wrist.y

        scale = np.sqrt(
            (mcp.x - wrist_x) ** 2 +
            (mcp.y - wrist_y) ** 2
        ) + 1e-6

        for i, landmark in enumerate(hand_landmarks):

            x = (landmark.x - wrist_x) / scale
            y = (landmark.y - wrist_y) / scale

            features[offset + i * 2] = x
            features[offset + i * 2 + 1] = y

    return features


def create_model_input(sequence):
    """
    Convert 30 frames of 84 features into
    the model's required 30 × 168 format.

    Positions:
        (30, 84)

    Velocities:
        (30, 84)

    Final:
        (30, 168)
    """

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.shape != (30, 84):
        raise ValueError(
            f"Expected (30, 84), got {sequence.shape}"
        )

    velocity = np.zeros_like(sequence)

    velocity[1:] = (
        sequence[1:] - sequence[:-1]
    )

    combined = np.concatenate(
        [sequence, velocity],
        axis=1
    )

    return combined