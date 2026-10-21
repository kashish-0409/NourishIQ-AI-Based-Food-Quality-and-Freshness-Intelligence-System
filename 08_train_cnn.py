
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# NOURISHIQ - BASELINE CNN TRAINING
# ==========================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

dataset_path = r"C:\Users\kashu\OneDrive\Desktop\AgriFreshNET Freshness and Shelf-Life Image Datase\AgriFreshNET Freshness and Shelf-Life Image Datase\Processed Data\Processed Data"

csv_path = os.path.join(
    os.path.dirname(dataset_path),
    "NourishIQ_Dataset_Split",
    "dataset_split.csv"
)

project_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(project_dir, "models")
results_dir = os.path.join(project_dir, "results")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

# ==========================================
# CHECK CSV AND IMAGE PATHS
# ==========================================

print("=" * 55)
print("NOURISHIQ BASELINE CNN")
print("=" * 55)

if not os.path.isdir(dataset_path):
    raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

df = pd.read_csv(csv_path)

required_columns = {"image_path", "class", "split"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_columns}")

if df["image_path"].duplicated().any():
    raise ValueError("Duplicate image paths found in CSV.")

if not df["split"].isin(["train", "validation", "test"]).all():
    raise ValueError("Unexpected split name found in CSV.")

missing_paths = [
    path for path in df["image_path"].tolist()
    if not os.path.isfile(path)
]

if missing_paths:
    raise FileNotFoundError(
        f"{len(missing_paths)} image paths are missing. "
        f"Example: {missing_paths[0]}"
    )

class_names = sorted(df["class"].unique().tolist())
class_to_index = {name: i for i, name in enumerate(class_names)}

print("Total images:", len(df))
print("Total classes:", len(class_names))
print("Split counts:\n", df["split"].value_counts())

# ==========================================
# VERIFY CLASS DISTRIBUTION
# ==========================================

distribution = pd.crosstab(df["class"], df["split"])

if len(class_names) != 24:
    raise ValueError(f"Expected 24 classes, found {len(class_names)}.")

if not {"train", "validation", "test"}.issubset(distribution.columns):
    raise ValueError("One or more dataset splits are missing.")

print("\nClass-wise split counts:")
print(distribution)

# ==========================================
# BUILD TF.DATA PIPELINE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

def load_image(path, label, training=False):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.08)
        image = tf.image.random_contrast(
            image, lower=0.9, upper=1.1
        )

    return image, label


def make_dataset(frame, training=False):
    paths = frame["image_path"].to_numpy()
    labels = frame["class"].map(class_to_index).to_numpy(
        dtype=np.int32
    )

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(frame),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        lambda path, label: load_image(path, label, training),
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset


train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# ==========================================
# BASELINE CNN MODEL
# ==========================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),

    tf.keras.layers.Conv2D(
        32, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        128, 3, padding="same", activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# SAVE BEST MODEL
# ==========================================

model_path = os.path.join(models_dir, "nourishiq_baseline.keras")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(
        os.path.join(results_dir, "training_history.csv")
    )
]

# ==========================================
# TRAIN
# ==========================================

print("\nStarting CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Load best checkpoint before evaluating
model = tf.keras.models.load_model(model_path)

# ==========================================
# PLOT TRAINING CURVES
# ==========================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("NourishIQ CNN Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(results_dir, "accuracy_curve.png"),
    dpi=300
)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("NourishIQ CNN Loss")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(results_dir, "loss_curve.png"),
    dpi=300
)
plt.close()

# ==========================================
# TEST SET EVALUATION
# ==========================================

print("\nEvaluating on the held-out test set...")

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

true_labels = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

probabilities = model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)

report = classification_report(
    true_labels,
    predicted_labels,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\nTEST ACCURACY:", round(float(test_accuracy), 4))
print("\nCLASSIFICATION REPORT\n")
print(report)

with open(
    os.path.join(results_dir, "classification_report.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(f"Test loss: {test_loss:.4f}\n")
    file.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
    file.write(report)

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=np.arange(len(class_names))
)

fig, ax = plt.subplots(figsize=(15, 13))
image = ax.imshow(cm, cmap="Blues")
ax.set_title("NourishIQ CNN - Confusion Matrix")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=90, fontsize=7)
ax.set_yticklabels(class_names, fontsize=7)
fig.colorbar(image, ax=ax)
fig.tight_layout()
fig.savefig(
    os.path.join(results_dir, "confusion_matrix.png"),
    dpi=300
)
plt.close(fig)

# ==========================================
# SAVE CLASS LABEL MAPPING
# ==========================================

label_mapping_path = os.path.join(
    models_dir, "class_labels.txt"
)

with open(label_mapping_path, "w", encoding="utf-8") as file:
    for index, name in enumerate(class_names):
        file.write(f"{index}\t{name}\n")

print("\nTRAINING AND EVALUATION COMPLETED")
print("Model:", model_path)
print("Results folder:", results_dir)