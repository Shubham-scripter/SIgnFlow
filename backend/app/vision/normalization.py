def normalize_landmarks(landmarks):
    """
    Normalize hand landmarks relative to the wrist.

    landmarks:
        List containing 63 values
        21 landmarks × (x, y, z)
    """

    if len(landmarks) != 63:
        return []

    # Wrist = first landmark
    wrist_x = landmarks[0]
    wrist_y = landmarks[1]
    wrist_z = landmarks[2]

    normalized = []

    for i in range(0, 63, 3):

        x = landmarks[i]
        y = landmarks[i + 1]
        z = landmarks[i + 2]

        normalized.append(x - wrist_x)
        normalized.append(y - wrist_y)
        normalized.append(z - wrist_z)

    return normalized