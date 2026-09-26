import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "backend",
    "models",
    "isl",
    "isl_static_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "backend",
    "models",
    "isl",
    "isl_training_data.npz"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "backend",
    "models",
    "isl",
    "isl_model_config.json"
)


print("=" * 70)
print("SIGNFLOW MODEL DIAGNOSTIC")
print("=" * 70)

# ------------------------------------------------------------
# CHECK FILES
# ------------------------------------------------------------

print("\n[FILES]")

for path in [MODEL_PATH, DATA_PATH, CONFIG_PATH]:
    print(path)

    if os.path.exists(path):
        print("  FOUND")
    else:
        print("  MISSING")


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\n[MODEL] Loading...")

model = joblib.load(MODEL_PATH)

print("Model type:", type(model).__name__)

if hasattr(model, "n_features_in_"):
    print(
        "Model expects features:",
        model.n_features_in_
    )

if hasattr(model, "classes_"):
    print(
        "Model classes:",
        list(model.classes_)
    )


# ------------------------------------------------------------
# LOAD TRAINING DATA
# ------------------------------------------------------------

print("\n[DATA] Loading training data...")

data = np.load(DATA_PATH)

print("NPZ keys:", list(data.keys()))

X = data["X"]
y = data["y"]

print("X shape:", X.shape)
print("y shape:", y.shape)

print("X dtype:", X.dtype)
print("y dtype:", y.dtype)

print(
    "X min:",
    float(np.min(X))
)

print(
    "X max:",
    float(np.max(X))
)

print(
    "X mean:",
    float(np.mean(X))
)

print(
    "X std:",
    float(np.std(X))
)


# ------------------------------------------------------------
# FEATURE COUNT CHECK
# ------------------------------------------------------------

print("\n[FEATURE CHECK]")

if X.ndim != 2:
    print("ERROR: X is not 2-dimensional.")
else:
    print(
        "Training feature count:",
        X.shape[1]
    )

    if hasattr(model, "n_features_in_"):
        print(
            "Model feature count:",
            model.n_features_in_
        )

        if X.shape[1] == model.n_features_in_:
            print("PASS: Feature counts match.")
        else:
            print("FAIL: Feature counts DO NOT match.")


# ------------------------------------------------------------
# MODEL PREDICTION ON TRAINING DATA
# ------------------------------------------------------------

print("\n[PREDICTION] Testing model on its own training data...")

predictions = model.predict(X)

accuracy = np.mean(predictions == y)

print(
    "Training accuracy:",
    f"{accuracy * 100:.2f}%"
)


# ------------------------------------------------------------
# CONFIDENCE TEST
# ------------------------------------------------------------

print("\n[CONFIDENCE] Testing Random Forest probabilities...")

probabilities = model.predict_proba(X)

max_confidence = np.max(
    probabilities,
    axis=1
)

print(
    "Average confidence:",
    f"{np.mean(max_confidence) * 100:.2f}%"
)

print(
    "Minimum confidence:",
    f"{np.min(max_confidence) * 100:.2f}%"
)

print(
    "Maximum confidence:",
    f"{np.max(max_confidence) * 100:.2f}%"
)


# ------------------------------------------------------------
# FIRST 20 EXAMPLES
# ------------------------------------------------------------

print("\n[FIRST 20 TRAINING EXAMPLES]")

for i in range(min(20, len(X))):

    predicted = predictions[i]
    actual = y[i]
    confidence = max_confidence[i]

    status = "OK" if predicted == actual else "WRONG"

    print(
        f"{i:03d} | "
        f"Actual: {str(actual):15s} | "
        f"Predicted: {str(predicted):15s} | "
        f"Confidence: {confidence * 100:6.2f}% | "
        f"{status}"
    )


# ------------------------------------------------------------
# CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\n[CLASS DISTRIBUTION]")

classes, counts = np.unique(
    y,
    return_counts=True
)

for class_name, count in zip(classes, counts):

    print(
        f"{str(class_name):20s} : {count}"
    )


print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)