import json
import zipfile
import io

import h5py
import numpy as np
import tensorflow as tf


MODEL_FILE = "temp_isl_model/models/isl_model.keras"


def load_weights_from_keras(model):
    """Load weights directly from the .keras H5 weight archive."""

    with zipfile.ZipFile(MODEL_FILE, "r") as z:
        weight_data = z.read("model.weights.h5")

    f = h5py.File(io.BytesIO(weight_data), "r")

    # Conv1D
    model.get_layer("conv1d").set_weights([
        f["layers/conv1d/vars/0"][:],
        f["layers/conv1d/vars/1"][:]
    ])

    model.get_layer("conv1d_1").set_weights([
        f["layers/conv1d_1/vars/0"][:],
        f["layers/conv1d_1/vars/1"][:]
    ])

    # Batch Normalization
    model.get_layer("batch_normalization").set_weights([
        f["layers/batch_normalization/vars/0"][:],
        f["layers/batch_normalization/vars/1"][:],
        f["layers/batch_normalization/vars/2"][:],
        f["layers/batch_normalization/vars/3"][:]
    ])

    model.get_layer("batch_normalization_1").set_weights([
        f["layers/batch_normalization_1/vars/0"][:],
        f["layers/batch_normalization_1/vars/1"][:],
        f["layers/batch_normalization_1/vars/2"][:],
        f["layers/batch_normalization_1/vars/3"][:]
    ])

    # Bidirectional LSTM
    bidirectional = model.get_layer("bidirectional")

    forward_lstm = bidirectional.forward_layer
    backward_lstm = bidirectional.backward_layer

    forward_lstm.set_weights([
        f["layers/bidirectional/forward_layer/cell/vars/0"][:],
        f["layers/bidirectional/forward_layer/cell/vars/1"][:],
        f["layers/bidirectional/forward_layer/cell/vars/2"][:]
    ])

    backward_lstm.set_weights([
        f["layers/bidirectional/backward_layer/cell/vars/0"][:],
        f["layers/bidirectional/backward_layer/cell/vars/1"][:],
        f["layers/bidirectional/backward_layer/cell/vars/2"][:]
    ])

    # Dense layers
    model.get_layer("dense").set_weights([
        f["layers/dense/vars/0"][:],
        f["layers/dense/vars/1"][:]
    ])

    model.get_layer("dense_1").set_weights([
        f["layers/dense_1/vars/0"][:],
        f["layers/dense_1/vars/1"][:]
    ])

    f.close()


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(30, 168),
            name="input_layer"
        ),

        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
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
            filters=128,
            kernel_size=3,
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


if __name__ == "__main__":

    print("Creating model...")
    model = create_model()

    print("Building model...")
    model.build((None, 30, 168))

    print("Loading trained weights...")
    load_weights_from_keras(model)

    print("SUCCESS!")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    # Test prediction
    dummy_input = np.zeros((1, 30, 168), dtype=np.float32)

    prediction = model.predict(
        dummy_input,
        verbose=0
    )

    print("Prediction shape:", prediction.shape)
    print("Prediction:", prediction)
    print("Predicted class:", np.argmax(prediction))