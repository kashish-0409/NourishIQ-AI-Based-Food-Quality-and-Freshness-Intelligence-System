import os
import random
import csv

# -----------------------------------
# DATASET PATH
# -----------------------------------

dataset_path = r"C:\Users\kashu\OneDrive\Desktop\AgriFreshNET Freshness and Shelf-Life Image Datase\AgriFreshNET Freshness and Shelf-Life Image Datase\Processed Data\Processed Data"

output_folder = os.path.join(
    os.path.dirname(dataset_path),
    "NourishIQ_Dataset_Split"
)

# -----------------------------------
# SETTINGS
# -----------------------------------

random.seed(42)

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

# -----------------------------------
# CHECK DATASET
# -----------------------------------

if not os.path.exists(dataset_path):
    raise FileNotFoundError(
        "Dataset folder not found."
    )

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# -----------------------------------
# FIND CLASSES
# -----------------------------------

class_names = sorted([
    folder
    for folder in os.listdir(dataset_path)
    if os.path.isdir(
        os.path.join(dataset_path, folder)
    )
])

print(f"Classes found: {len(class_names)}")
print()

# -----------------------------------
# SPLIT DATA
# -----------------------------------

all_data = []

total_train = 0
total_validation = 0
total_test = 0

for class_name in class_names:

    class_path = os.path.join(
        dataset_path,
        class_name
    )

    images = [
        file
        for file in os.listdir(class_path)
        if file.lower().endswith(image_extensions)
        and os.path.isfile(
            os.path.join(class_path, file)
        )
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * 0.70)
    validation_count = int(total * 0.15)

    train_images = images[:train_count]

    validation_images = images[
        train_count:
        train_count + validation_count
    ]

    test_images = images[
        train_count + validation_count:
    ]

    # -----------------------------------
    # ADD TO MANIFEST
    # -----------------------------------

    for file in train_images:

        all_data.append([
            os.path.join(class_path, file),
            class_name,
            "train"
        ])

    for file in validation_images:

        all_data.append([
            os.path.join(class_path, file),
            class_name,
            "validation"
        ])

    for file in test_images:

        all_data.append([
            os.path.join(class_path, file),
            class_name,
            "test"
        ])

    total_train += len(train_images)
    total_validation += len(validation_images)
    total_test += len(test_images)

    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Validation={len(validation_images)}, "
        f"Test={len(test_images)}"
    )

# -----------------------------------
# SAVE CSV
# -----------------------------------

csv_path = os.path.join(
    output_folder,
    "dataset_split.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "image_path",
        "class",
        "split"
    ])

    writer.writerows(all_data)

# -----------------------------------
# FINAL SUMMARY
# -----------------------------------

print("\n===================================")
print("DATASET SPLIT COMPLETED")
print("===================================")

print(f"Total classes : {len(class_names)}")
print(f"Train images  : {total_train}")
print(f"Validation    : {total_validation}")
print(f"Test images   : {total_test}")
print(
    f"Total images  : "
    f"{total_train + total_validation + total_test}"
)

print("\nCSV file created:")
print(csv_path)