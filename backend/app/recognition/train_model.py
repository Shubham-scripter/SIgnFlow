import os
import csv
import json
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


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

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "data",
    "isl"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "models",
    "isl"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ==========================================
# SETTINGS
# ==========================================

FEATURE_COUNT = 126

# Controls how tolerant the Unknown rejection is.
# Higher value = more tolerant.
REJECTION_MARGIN = 1.20


# ==========================================
# FIND DATASETS
# ==========================================

def find_datasets():

    if not os.path.exists(DATASET_DIR):

        print(
            "ERROR: Dataset folder not found:"
        )

        print(
            DATASET_DIR
        )

        return []

    files = []

    for filename in os.listdir(
        DATASET_DIR
    ):

        if filename.lower().endswith(
            ".csv"
        ):

            files.append(
                filename
            )

    files.sort()

    return files


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    X = []
    y = []

    dataset_files = find_datasets()

    if not dataset_files:

        return (
            np.empty(
                (
                    0,
                    FEATURE_COUNT
                ),
                dtype=np.float32
            ),
            np.array([])
        )


    print()
    print("Datasets found:")

    for filename in dataset_files:

        print(
            f"  {filename}"
        )


    print()


    for filename in dataset_files:

        # Filename becomes gesture name.
        #
        # namaste.csv -> namaste
        # hello.csv   -> hello
        # yes.csv     -> yes

        gesture_name = os.path.splitext(
            filename
        )[0]


        file_path = os.path.join(
            DATASET_DIR,
            filename
        )


        print(
            f"Loading: {filename}"
        )


        sample_count = 0


        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.reader(
                file
            )

            # Skip CSV header.
            next(
                reader,
                None
            )


            for row in reader:

                # Check feature count.

                if len(row) != FEATURE_COUNT:

                    print(
                        f"  Skipping invalid row "
                        f"({len(row)} features)"
                    )

                    continue


                try:

                    features = [
                        float(value)
                        for value in row
                    ]

                except ValueError:

                    print(
                        "  Skipping non-numeric row"
                    )

                    continue


                X.append(
                    features
                )

                y.append(
                    gesture_name
                )

                sample_count += 1


        print(
            f"  Samples: {sample_count}"
        )


    return (
        np.array(
            X,
            dtype=np.float32
        ),
        np.array(
            y
        )
    )


# ==========================================
# CALCULATE UNKNOWN REJECTION THRESHOLD
# ==========================================

def calculate_threshold(
    X,
    y
):

    print()
    print(
        "Calculating Unknown rejection threshold..."
    )


    nearest_distances = []


    for i in range(
        len(X)
    ):

        current = X[i]


        # Only compare the sample
        # with samples of the same gesture.

        same_class = X[
            y == y[i]
        ]


        # Need at least two samples
        # to calculate a distance.

        if len(
            same_class
        ) < 2:

            continue


        distances = np.linalg.norm(
            same_class - current,
            axis=1
        )


        # Remove the sample's distance
        # to itself.

        distances = distances[
            distances > 1e-6
        ]


        if len(
            distances
        ) == 0:

            continue


        nearest_distance = np.min(
            distances
        )


        nearest_distances.append(
            nearest_distance
        )


    if not nearest_distances:

        print(
            "WARNING: Could not calculate "
            "rejection threshold."
        )

        return 0.0


    nearest_distances = np.array(
        nearest_distances,
        dtype=np.float32
    )


    # 95th percentile of the
    # normal training variation.

    base_threshold = np.percentile(
        nearest_distances,
        95
    )


    # Add tolerance.

    threshold = (
        base_threshold *
        REJECTION_MARGIN
    )


    print()

    print(
        f"Minimum distance : "
        f"{nearest_distances.min():.4f}"
    )

    print(
        f"Average distance : "
        f"{nearest_distances.mean():.4f}"
    )

    print(
        f"95% distance     : "
        f"{base_threshold:.4f}"
    )

    print(
        f"Final threshold  : "
        f"{threshold:.4f}"
    )


    return float(
        threshold
    )


# ==========================================
# MAIN
# ==========================================

print()
print("========================================")
print("       SIGNFLOW MODEL TRAINING")
print("========================================")
print()


# ==========================================
# LOAD DATA
# ==========================================

X, y = load_dataset()


print()

print(
    "Dataset shape:",
    X.shape
)


if len(X) == 0:

    print()

    print(
        "ERROR: No training data found."
    )

    print(
        "Put gesture CSV files inside:"
    )

    print(
        DATASET_DIR
    )

    exit()


# ==========================================
# SHOW CLASS COUNTS
# ==========================================

classes = sorted(
    np.unique(
        y
    )
)


print()

print(
    "Samples per gesture:"
)


for gesture in classes:

    count = np.sum(
        y == gesture
    )

    print(
        f"{gesture}: {count}"
    )


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

if len(classes) >= 2:

    # Multiple gestures available.

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )


    print()

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )


else:

    # Only one gesture available.

    print()

    print(
        "Only one gesture found."
    )

    print(
        "Training using all available samples."
    )


    X_train = X

    X_test = None

    y_train = y

    y_test = None


# ==========================================
# TRAIN RANDOM FOREST
# ==========================================

print()

print(
    "Training Random Forest..."
)


model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print(
    "Training complete."
)


# ==========================================
# MODEL EVALUATION
# ==========================================

if (
    X_test is not None
    and len(
        np.unique(
            y_test
        )
    ) >= 2
):

    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    print()

    print(
        "========================================"
    )

    print(
        "           MODEL RESULTS"
    )

    print(
        "========================================"
    )


    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )


    print()

    print(
        "Classification Report:"
    )

    print()


    print(
        classification_report(
            y_test,
            predictions
        )
    )


else:

    print()

    print(
        "Classification accuracy is not"
    )

    print(
        "meaningful with only one gesture."
    )


# ==========================================
# CALCULATE UNKNOWN THRESHOLD
# ==========================================

threshold = calculate_threshold(
    X_train,
    y_train
)


# ==========================================
# SAVE RANDOM FOREST
# ==========================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "isl_static_model.pkl"
)


joblib.dump(
    model,
    MODEL_PATH
)


# ==========================================
# SAVE TRAINING DATA
# ==========================================

TRAINING_DATA_PATH = os.path.join(
    MODEL_DIR,
    "isl_training_data.npz"
)


np.savez(
    TRAINING_DATA_PATH,
    X=X_train,
    y=y_train
)


# ==========================================
# SAVE CONFIGURATION
# ==========================================

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "isl_model_config.json"
)


config = {
    "feature_count": FEATURE_COUNT,
    "threshold": threshold,
    "rejection_margin": REJECTION_MARGIN,
    "classes": classes
}


with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        config,
        file,
        indent=4
    )


# ==========================================
# FINAL OUTPUT
# ==========================================

print()

print(
    "========================================"
)

print(
    "          MODEL SAVED"
)

print(
    "========================================"
)


print()

print(
    "Random Forest:"
)

print(
    MODEL_PATH
)


print()

print(
    "Training data:"
)

print(
    TRAINING_DATA_PATH
)


print()

print(
    "Configuration:"
)

print(
    CONFIG_PATH
)


print()

print(
    "Gestures:"
)


for gesture in classes:

    print(
        f"  - {gesture}"
    )


print()

print(
    f"Unknown threshold: "
    f"{threshold:.4f}"
)


print()

print(
    "Training complete."
)