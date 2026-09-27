# ==========================================
# NourishIQ - Preprocessing Configuration
# ==========================================

import os
import csv

# ------------------------------------------
# DATASET SPLIT CSV
# ------------------------------------------

csv_path = r"C:\Users\kashu\OneDrive\Desktop\AgriFreshNET Freshness and Shelf-Life Image Datase\AgriFreshNET Freshness and Shelf-Life Image Datase\Processed Data\NourishIQ_Dataset_Split\dataset_split.csv"


# ------------------------------------------
# CURRENT IMAGE INFORMATION
# ------------------------------------------

ORIGINAL_IMAGE_SIZE = (512, 512)

# Model input size will be finalized
# according to the CNN / transfer-learning model.
MODEL_INPUT_SIZE = None


# ------------------------------------------
# NORMALIZATION
# ------------------------------------------

NORMALIZATION_METHOD = "Rescale pixel values from 0-255 to 0-1"


# ------------------------------------------
# DATA AUGMENTATION
# ------------------------------------------

AUGMENTATION_APPLIED_TO = "Training set only"

AUGMENTATION_PLAN = [
    "Horizontal flip",
    "Small rotation",
    "Small zoom",
    "Small translation"
]


# ------------------------------------------
# VERIFY CSV
# ------------------------------------------

if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"Dataset split CSV not found:\n{csv_path}"
    )


with open(
    csv_path,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    rows = list(reader)


# ------------------------------------------
# VERIFY IMAGE PATHS
# ------------------------------------------

missing_images = []

for row in rows:

    image_path = row["image_path"]

    if not os.path.isfile(image_path):
        missing_images.append(image_path)


# ------------------------------------------
# DISPLAY CONFIGURATION
# ------------------------------------------

print("==========================================")
print("NOURISHIQ PREPROCESSING CONFIGURATION")
print("==========================================")

print("\nDataset:")
print("Total images :", len(rows))
print("Original size:", ORIGINAL_IMAGE_SIZE)

print("\nModel Input:")
print("Input size   :", "To be finalized with model")

print("\nNormalization:")
print(NORMALIZATION_METHOD)

print("\nAugmentation:")
print("Applied to   :", AUGMENTATION_APPLIED_TO)

for item in AUGMENTATION_PLAN:
    print("-", item)

print("\nImage Path Check:")
print("Missing images:", len(missing_images))


# ------------------------------------------
# FINAL STATUS
# ------------------------------------------

print("\n==========================================")

if len(missing_images) == 0:

    print("PREPROCESSING DATA CHECK: PASS")

else:

    print("PREPROCESSING DATA CHECK: CHECK REQUIRED")

print("==========================================")