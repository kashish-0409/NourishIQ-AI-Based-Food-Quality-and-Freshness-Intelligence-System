
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------
# 1. Paths and reproducibility
# --------------------------------------------------
SEED = 42
tf.keras.utils.set_random_seed(SEED)

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "results" / "clean_dataset_split.csv"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "nourishiq_mobilenetv2_v2.keras"
LABELS_PATH = MODEL_DIR / "nourishiq_clean_class_labels.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
HEAD_EPOCHS = 10
FINETUNE_EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE

# --------------------------------------------------
# 2. Load clean dataset split
# --------------------------------------------------
if not CSV_PATH.is_file():
    raise FileNotFoundError(f"Clean split CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print("\nCSV columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

print("\nSplit counts:")
print(df["split"].value_counts())

print("\nDuplicate image paths:", df["image_path"].duplicated().sum())

required = {"image_path", "class", "split"}

if not required.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required}")

df["image_path"] = df["image_path"].astype(str).str.strip()
df["class"] = df["class"].astype(str).str.strip()
df["split"] = df["split"].astype(str).str.strip().str.lower()

encoder = LabelEncoder()
encoder.fit(df["class"])

if len(encoder.classes_) != 24:
    raise ValueError(f"Expected 24 classes, found {len(encoder.classes_)}")

df["label"] = encoder.transform(df["class"])

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

if min(len(train_df), len(val_df), len(test_df)) == 0:
    raise ValueError("Train, validation, or test split is empty.")

missing = df.loc[~df["image_path"].map(os.path.isfile), "image_path"]
if not missing.empty:
    raise FileNotFoundError(
        f"{len(missing)} image paths are missing. Check the CSV."
    )

print("Dataset:", CSV_PATH)
print("\nSplit counts:")
print(df["split"].value_counts())
print("\nClasses:", len(encoder.classes_))

# Save a separate class-label file; preserve the existing app labels.
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(encoder.classes_.tolist(), f, indent=2)

# --------------------------------------------------
# 3. TensorFlow data pipelines
# --------------------------------------------------
def load_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def make_dataset(frame, training=False):
    paths = frame["image_path"].to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            len(frame), seed=SEED, reshuffle_each_iteration=True
        )

    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(AUTOTUNE)


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# --------------------------------------------------
# 4. Fresh MobileNetV2 transfer-learning model
# --------------------------------------------------
# This starts with ImageNet weights, NOT the old NourishIQ model.
augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ],
    name="data_augmentation",
)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3),
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
x = augmentation(inputs)
x = tf.keras.layers.Lambda(preprocess_input, name="preprocessing")(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)
outputs = tf.keras.layers.Dense(
    len(encoder.classes_), activation="softmax", name="predictions"
)(x)

model = tf.keras.Model(inputs, outputs, name="NourishIQ_Clean_MobileNetV2")

def compile_model(learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
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
        str(RESULTS_DIR / "clean_mobilenetv2_history.csv")
    ),
]

# --------------------------------------------------
# 5. Train classification head
# --------------------------------------------------
compile_model(learning_rate=1e-3)

print("\nPhase 1: Training classification head")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks,
)

# --------------------------------------------------
# 6. Fine-tune the last 30 backbone layers
# --------------------------------------------------
print("\nPhase 2: Fine-tuning MobileNetV2")

base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

# Keep Batch Normalization layers frozen for stability.
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

compile_model(learning_rate=1e-5)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINETUNE_EPOCHS,
    callbacks=callbacks,
)

# --------------------------------------------------
# 7. Evaluate the best saved checkpoint ONCE on test set
# --------------------------------------------------
if not MODEL_PATH.is_file():
    raise FileNotFoundError("Best model checkpoint was not saved.")

best_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"preprocess_input": preprocess_input},
)

test_loss, _ = best_model.evaluate(test_ds, verbose=1)
probabilities = best_model.predict(test_ds, verbose=1)
y_true = test_df["label"].to_numpy()
y_pred = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
macro_precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)
macro_recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    digits=4,
    zero_division=0,
)

report_path = RESULTS_DIR / "clean_mobilenetv2_classification_report.txt"
report_path.write_text(report, encoding="utf-8")

cm = confusion_matrix(
    y_true, y_pred, labels=np.arange(len(encoder.classes_))
)
pd.DataFrame(
    cm, index=encoder.classes_, columns=encoder.classes_
).to_csv(RESULTS_DIR / "clean_mobilenetv2_confusion_matrix.csv")

metrics = {
    "test_samples": int(len(y_true)),
    "number_of_classes": int(len(encoder.classes_)),
    "test_accuracy": float(accuracy),
    "test_loss": float(test_loss),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "model": str(MODEL_PATH),
    "evaluation_split": str(CSV_PATH),
}

with open(
    RESULTS_DIR / "clean_mobilenetv2_test_metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metrics, f, indent=2)

with open(
    RESULTS_DIR / "clean_mobilenetv2_test_metrics.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(f"Test samples: {len(y_true)}\n")
    f.write(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    f.write(f"Test loss: {test_loss:.4f}\n")
    f.write(f"Macro precision: {macro_precision:.4f}\n")
    f.write(f"Macro recall: {macro_recall:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n")

print("\n========== CLEAN TEST RESULTS ==========")
print(f"Accuracy:       {accuracy * 100:.2f}%")
print(f"Macro Precision:{macro_precision:.4f}")
print(f"Macro Recall:   {macro_recall:.4f}")
print(f"Macro F1:       {macro_f1:.4f}")
print(f"Weighted F1:    {weighted_f1:.4f}")
print(f"Test loss:      {test_loss:.4f}")
print("\nClassification report:\n", report)
print("\nModel saved:", MODEL_PATH)
print("Labels saved:", LABELS_PATH)
print("Results saved in:", RESULTS_DIR)
