import sys
import os
import cv2
import json
import zipfile
import tempfile

import numpy as np
import tensorflow as tf
import mediapipe as mp

from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT ISL FEATURES
# ============================================================

from backend.app.vision.isl_features import (
    extract_isl_features,
    create_model_input
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "temp_isl_model",
    "models",
    "isl_model.keras"
)

LABEL_PATH = os.path.join(
    PROJECT_ROOT,
    "temp_isl_model",
    "models",
    "index_to_label.json"
)

HAND_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "app",
    "vision",
    "hand_landmarker.task"
)


# ============================================================
# LOAD LABELS
# ============================================================

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    index_to_label = json.load(f)

index_to_label = {
    int(k): v
    for k, v in index_to_label.items()
}

print("Labels:")
print(index_to_label)


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(30, 168),
            name="input_layer"
        ),

        tf.keras.layers.Conv1D(
            64,
            3,
            activation="relu",
            name="conv1d"
        ),

        tf.keras.layers.BatchNormalization(
            name="batch_normalization"
        ),

        tf.keras.layers.MaxPooling1D(
            pool_size=2,
            name="max_pooling1d"
        ),

        tf.keras.layers.Dropout(
            0.3,
            name="dropout"
        ),

        tf.keras.layers.Conv1D(
            128,
            3,
            activation="relu",
            name="conv1d_1"
        ),

        tf.keras.layers.BatchNormalization(
            name="batch_normalization_1"
        ),

        tf.keras.layers.MaxPooling1D(
            pool_size=2,
            name="max_pooling1d_1"
        ),

        tf.keras.layers.Dropout(
            0.3,
            name="dropout_1"
        ),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64),
            name="bidirectional"
        ),

        tf.keras.layers.Dense(
            128,
            activation="relu",
            name="dense"
        ),

        tf.keras.layers.Dropout(
            0.3,
            name="dropout_2"
        ),

        tf.keras.layers.Dense(
            9,
            activation="softmax",
            name="dense_1"
        )
    ])

    return model


# ============================================================
# HELPER: LOAD VARIABLES FROM H5 GROUP
# ============================================================

def load_layer_variables(layer, variables_group):

    if "vars" not in variables_group:
        return

    variables = variables_group["vars"]

    for i, variable in enumerate(layer.weights):

        key = str(i)

        if key not in variables:
            print(
                f"Missing weight {key} "
                f"for {layer.name}"
            )
            continue

        value = variables[key][:]

        if tuple(variable.shape) != tuple(value.shape):

            print(
                f"Shape mismatch for {layer.name} "
                f"weight {key}: "
                f"model={variable.shape}, "
                f"file={value.shape}"
            )

            continue

        variable.assign(value)


# ============================================================
# LOAD BIDIRECTIONAL LSTM
# ============================================================

def load_bidirectional_weights(layer, group):

    print("Loading Bidirectional LSTM weights...")

    # --------------------------------------------------------
    # Forward LSTM
    # --------------------------------------------------------

    if "forward_layer" in group:

        forward_group = group["forward_layer"]

        if "cell" in forward_group:

            cell_group = forward_group["cell"]

            print("  Loading forward LSTM...")

            load_layer_variables(
                layer.forward_layer.cell,
                cell_group
            )


    # --------------------------------------------------------
    # Backward LSTM
    # --------------------------------------------------------

    if "backward_layer" in group:

        backward_group = group["backward_layer"]

        if "cell" in backward_group:

            cell_group = backward_group["cell"]

            print("  Loading backward LSTM...")

            load_layer_variables(
                layer.backward_layer.cell,
                cell_group
            )

    print("  Bidirectional LSTM loaded.")


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

def load_weights_from_keras(model, keras_path):

    print("\nLoading model weights...")

    with zipfile.ZipFile(
        keras_path,
        "r"
    ) as z:

        with tempfile.NamedTemporaryFile(
            suffix=".h5",
            delete=False
        ) as temp:

            temp.write(
                z.read("model.weights.h5")
            )

            weights_path = temp.name


    try:

        import h5py

        with h5py.File(
            weights_path,
            "r"
        ) as f:

            layers_group = f["layers"]


            # ------------------------------------------------
            # LOAD EACH MODEL LAYER
            # ------------------------------------------------

            for layer in model.layers:

                layer_name = layer.name

                if layer_name not in layers_group:

                    print(
                        f"Skipping: {layer_name}"
                    )

                    continue


                layer_group = layers_group[
                    layer_name
                ]


                # --------------------------------------------
                # BIDIRECTIONAL LSTM
                # --------------------------------------------

                if layer_name == "bidirectional":

                    load_bidirectional_weights(
                        layer,
                        layer_group
                    )

                    continue


                # --------------------------------------------
                # NORMAL LAYERS
                # --------------------------------------------

                if "vars" not in layer_group:

                    print(
                        f"No variables: {layer_name}"
                    )

                    continue


                variables = layer_group["vars"]

                print(
                    f"Loading {layer_name}: "
                    f"{len(layer.weights)} weights"
                )


                for i, variable in enumerate(
                    layer.weights
                ):

                    key = str(i)

                    if key not in variables:

                        print(
                            f"Missing weight {key} "
                            f"for {layer_name}"
                        )

                        continue


                    value = variables[key][:]


                    if tuple(variable.shape) != tuple(
                        value.shape
                    ):

                        print(
                            f"Shape mismatch for "
                            f"{layer_name} weight {key}"
                        )

                        print(
                            "Model:",
                            variable.shape
                        )

                        print(
                            "File:",
                            value.shape
                        )

                        continue


                    variable.assign(value)


    finally:

        try:
            os.remove(weights_path)
        except:
            pass


    print("\nAll available model weights loaded.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading ISL model...")

model = create_model()

load_weights_from_keras(
    model,
    MODEL_PATH
)

print("ISL model loaded!")


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

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


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    detector.close()

    sys.exit()


# ============================================================
# SEQUENCE BUFFER
# ============================================================

sequence = deque(
    maxlen=30
)

frame_timestamp = 0


# ============================================================
# START
# ============================================================

print()
print("========================================")
print("      SIGNFLOW ISL RECOGNITION")
print("========================================")
print()
print("Perform one of the supported gestures:")
print()
print("background")
print("help")
print("hey")
print("namaste")
print("please")
print("sorry")
print("thank_you")
print("water")
print("yes")
print()
print("Press Q to quit.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()


    if not success:

        print(
            "ERROR: Could not read camera."
        )

        break


    # --------------------------------------------------------
    # MIRROR CAMERA
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MEDIAPIPE IMAGE
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # HAND DETECTION
    # --------------------------------------------------------

    result = detector.detect_for_video(
        mp_image,
        frame_timestamp
    )

    frame_timestamp += 1


    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    features = extract_isl_features(
        result
    )


    # --------------------------------------------------------
    # ADD TO SEQUENCE
    # --------------------------------------------------------

    sequence.append(
        features
    )


    # --------------------------------------------------------
    # DRAW HAND LANDMARKS
    # --------------------------------------------------------

    if result.hand_landmarks:

        height, width, _ = frame.shape


        for hand in result.hand_landmarks:

            for landmark in hand:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )


                cv2.circle(

                    frame,

                    (x, y),

                    5,

                    (0, 255, 0),

                    -1
                )


    # --------------------------------------------------------
    # DEFAULT DISPLAY
    # --------------------------------------------------------

    label = "Waiting..."

    confidence = 0.0


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if len(sequence) == 30:

        sequence_array = np.array(
            sequence,
            dtype=np.float32
        )


        # --------------------------------------------
        # 30 × 84
        #        ↓
        # 30 × 168
        # --------------------------------------------

        model_input = create_model_input(
            sequence_array
        )


        # --------------------------------------------
        # Add batch dimension
        #
        # 30 × 168
        #      ↓
        # 1 × 30 × 168
        # --------------------------------------------

        model_input = np.expand_dims(
            model_input,
            axis=0
        )


        # --------------------------------------------
        # PREDICT
        # --------------------------------------------

        prediction = model.predict(
            model_input,
            verbose=0
        )[0]


        # --------------------------------------------
        # BEST CLASS
        # --------------------------------------------

        class_index = int(
            np.argmax(prediction)
        )


        confidence = float(
            prediction[class_index]
        )


        label = index_to_label.get(
            class_index,
            "Unknown"
        )


    # ========================================================
    # DISPLAY GESTURE
    # ========================================================

    cv2.putText(

        frame,

        f"Gesture: {label}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0, 255, 0),

        2
    )


    # ========================================================
    # DISPLAY CONFIDENCE
    # ========================================================

    cv2.putText(

        frame,

        f"Confidence: {confidence:.2f}",

        (20, 80),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2
    )


    # ========================================================
    # DISPLAY BUFFER
    # ========================================================

    cv2.putText(

        frame,

        f"Frames: {len(sequence)}/30",

        (20, 120),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(

        "SignFlow - ISL Recognition",

        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

detector.close()

cv2.destroyAllWindows()

print()
print("SignFlow ISL recognition stopped.")