def extract_landmarks(hand_landmarks):
    """
    Convert MediaPipe hand landmarks
    into a list of x, y, z coordinates.
    """

    landmarks = []

    for landmark in hand_landmarks:
        landmarks.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return landmarks


def flatten_landmarks(hand_landmarks):
    """
    Convert landmarks into a single list.

    21 landmarks × 3 coordinates = 63 values.
    """

    data = []

    for landmark in hand_landmarks:
        data.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return data