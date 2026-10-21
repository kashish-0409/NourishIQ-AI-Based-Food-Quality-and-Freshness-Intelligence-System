
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# NOURISHIQ - MODEL EVALUATION
# ==========================================

dataset_path = r"C:\Users\kashu\OneDrive\Desktop\AgriFreshNET Freshness and Shelf-Life Image Datase\AgriFreshNET Freshness and Shelf-Life Image Datase\Processed Data\Processed Data"

csv_path = os.path.join(
    os.path.dirname(dataset_path),
    "NourishIQ_Dataset_Split",
    "dataset_split.csv"
)

project_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    project_dir, "models", "nourishiq_baseline.keras"
)

results_dir = os.path.join(project_dir, "results")
os.makedirs(results_dir, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

print("=" * 50)
print("NOURISHIQ MODEL EVALUATION")
print("=" * 50)

# ==========================================
# CHECK FILES
# ==========================================

if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV not found: {csv_path}")

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Trained model not found: {model_path}")

df = pd.read_csv(csv_path)

test_df = df[df["split"] == "test"].copy()

if test_df.empty:
    raise ValueError("No test images found in CSV.")

missing_paths = [
    p for p in test_df["image_path"]
    if not os.path.isfile(p)
]

if missing_paths:
    raise FileNotFoundError(
        f"{len(missing_paths)} test images are missing. "
        f"Example: {missing_paths[0]}"
    )

# Must match the alphabetical class mapping used in training.
class_names = sorted(df["class"].unique().tolist())
class_to_index = {name: i for i, name in enumerate(class_names)}

print("Test images:", len(test_df))
print("Classes:", len(class_names))

# ==========================================
# LOAD SAVED MODEL
# ==========================================

model = tf.keras.models.load_model(model_path)

# ==========================================
# LOAD TEST IMAGES
# ==========================================

def load_test_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


test_paths = test_df["image_path"].to_numpy()

test_labels = test_df["class"].map(
    class_to_index
).to_numpy(dtype=np.int32)

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)

test_ds = test_ds.map(
    load_test_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==========================================
# EVALUATE
# ==========================================

test_loss, model_accuracy = model.evaluate(
    test_ds, verbose=1
)

true_labels = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

probabilities = model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(true_labels, predicted_labels)

report_dict = classification_report(
    true_labels,
    predicted_labels,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_text = classification_report(
    true_labels,
    predicted_labels,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\nTEST LOSS:", round(float(test_loss), 4))
print("TEST ACCURACY:", round(float(accuracy), 4))
print("\nCLASSIFICATION REPORT\n")
print(report_text)

# ==========================================
# SAVE METRICS
# ==========================================

metrics_path = os.path.join(
    results_dir, "evaluation_metrics.txt"
)

with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(f"Test images: {len(test_df)}\n")
    f.write(f"Test loss: {test_loss:.4f}\n")
    f.write(f"Test accuracy: {accuracy:.4f}\n\n")
    f.write(report_text)

report_csv_path = os.path.join(
    results_dir, "evaluation_report.csv"
)

pd.DataFrame(report_dict).transpose().to_csv(
    report_csv_path, index=True
)

# ==========================================
# SAVE CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=np.arange(len(class_names))
)

fig, ax = plt.subplots(figsize=(16, 14))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=True,
    values_format="d"
)

ax.set_title("NourishIQ - Test Confusion Matrix")
fig.tight_layout()

cm_path = os.path.join(
    results_dir, "evaluation_confusion_matrix.png"
)

fig.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nEVALUATION COMPLETED")
print("Metrics:", metrics_path)
print("Report CSV:", report_csv_path)
print("Confusion matrix:", cm_path)
