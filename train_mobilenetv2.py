import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------
# 1. Paths and reproducibility
# --------------------------------------------------
SEED = 42
tf.keras.utils.set_random_seed(SEED)

CSV_PATH = (
    r"C:\Users\kashu\OneDrive\Desktop"
    r"\AgriFreshNET Freshness and Shelf-Life Image Datase"
    r"\AgriFreshNET Freshness and Shelf-Life Image Datase"
    r"\Processed Data\NourishIQ_Dataset_Split\dataset_split.csv"
)

MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Original trained model: do not overwrite it
ORIGINAL_MODEL_PATH = os.path.join(
    MODEL_DIR, "nourishiq_mobilenetv2.keras"
)

# New fine-tuned model
FINETUNED_MODEL_PATH = os.path.join(
    MODEL_DIR, "nourishiq_mobilenetv2_finetuned.keras"
)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 8
AUTOTUNE = tf.data.AUTOTUNE

# --------------------------------------------------
# 2. Check original model and dataset
# --------------------------------------------------
if not os.path.isfile(ORIGINAL_MODEL_PATH):
    raise FileNotFoundError(
        f"Original trained model not found: {ORIGINAL_MODEL_PATH}"
    )

if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

required_columns = {"image_path", "class", "split"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_columns}")

df["split"] = df["split"].astype(str).str.strip().str.lower()
df["class"] = df["class"].astype(str).str.strip()

encoder = LabelEncoder()
encoder.fit(sorted(df["class"].unique()))

if len(encoder.classes_) != 24:
    raise ValueError(
        f"Expected 24 classes, found {len(encoder.classes_)}"
    )

# Keep class-label order consistent with the original experiment.
labels_path = os.path.join(MODEL_DIR, "mobilenetv2_class_labels.json")
if os.path.isfile(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        saved_labels = json.load(f)

    if list(saved_labels) != list(encoder.classes_):
        raise ValueError(
            "Class-label order differs from the saved experiment."
        )

df["label"] = encoder.transform(df["class"])

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

if min(len(train_df), len(val_df), len(test_df)) == 0:
    raise ValueError("Train, validation, or test split is empty.")

missing = df.loc[
    ~df["image_path"].map(os.path.isfile), "image_path"
]
if not missing.empty:
    raise FileNotFoundError(
        f"{len(missing)} image paths are missing."
    )

print("Dataset split counts:")
print(df["split"].value_counts())
print("Number of classes:", len(encoder.classes_))

# --------------------------------------------------
# 3. Build TensorFlow datasets
# --------------------------------------------------
def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def make_dataset(frame, training=False):
    paths = frame["image_path"].to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            len(frame),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(AUTOTUNE)


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# --------------------------------------------------
# 4. Load the trained MobileNetV2 model
# --------------------------------------------------
print("\nLoading the existing trained MobileNetV2 model...")

model = tf.keras.models.load_model(
    ORIGINAL_MODEL_PATH,
    custom_objects={"preprocess_input": preprocess_input},
)

print("Original model loaded successfully.")

# Find the pretrained MobileNetV2 backbone inside the saved model.
base_model = None

for layer in model.layers:
    if (
        isinstance(layer, tf.keras.Model)
        and "mobilenetv2" in layer.name.lower()
    ):
        base_model = layer
        break

if base_model is None:
    raise ValueError(
        "Could not find the nested MobileNetV2 backbone in the saved "
        "model. Original model remains unchanged."
    )

# --------------------------------------------------
# 5. Fine-tune only the last 30 backbone layers
# --------------------------------------------------
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Recompile after changing trainable layers.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --------------------------------------------------
# 6. Fine-tune using training and validation data
# --------------------------------------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        FINETUNED_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        os.path.join(RESULTS_DIR, "mobilenetv2_finetuned_history.csv")
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# --------------------------------------------------
# 7. Evaluate the best fine-tuned checkpoint
# --------------------------------------------------
if not os.path.isfile(FINETUNED_MODEL_PATH):
    raise FileNotFoundError(
        "Fine-tuned checkpoint was not saved. "
        "Check the training output before proceeding."
    )

best_model = tf.keras.models.load_model(
    FINETUNED_MODEL_PATH,
    custom_objects={"preprocess_input": preprocess_input},
)

test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)

y_true = test_df["label"].to_numpy()
probabilities = best_model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    digits=4,
    zero_division=0,
)

with open(
    os.path.join(
        RESULTS_DIR, "mobilenetv2_finetuned_classification_report.txt"
    ),
    "w",
    encoding="utf-8",
) as f:
    f.write(report)

np.savetxt(
    os.path.join(
        RESULTS_DIR, "mobilenetv2_finetuned_confusion_matrix.csv"
    ),
    confusion_matrix(y_true, y_pred),
    delimiter=",",
    fmt="%d",
)

with open(
    os.path.join(RESULTS_DIR, "mobilenetv2_finetuned_test_metrics.txt"),
    "w",
    encoding="utf-8",
) as f:
    f.write(f"Test accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test loss: {test_loss:.4f}\n")

print(
    "\nFine-tuned MobileNetV2 test accuracy:",
    round(test_accuracy * 100, 2),
    "%",
)
print("Fine-tuned MobileNetV2 test loss:", round(test_loss, 4))
print("\nClass-wise report:\n", report)
print("\nFine-tuned model saved:", FINETUNED_MODEL_PATH)
print("Original model preserved:", ORIGINAL_MODEL_PATH)
print("Results saved in:", RESULTS_DIR)