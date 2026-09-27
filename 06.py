import os
import csv
from collections import Counter, defaultdict

# -----------------------------------
# CSV PATH
# -----------------------------------

csv_path = r"C:\Users\kashu\OneDrive\Desktop\AgriFreshNET Freshness and Shelf-Life Image Datase\AgriFreshNET Freshness and Shelf-Life Image Datase\Processed Data\NourishIQ_Dataset_Split\dataset_split.csv"

# -----------------------------------
# CHECK CSV EXISTS
# -----------------------------------

if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"CSV file not found:\n{csv_path}"
    )

# -----------------------------------
# READ CSV
# -----------------------------------

rows = []

with open(
    csv_path,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:
        rows.append(row)

# -----------------------------------
# BASIC COUNTS
# -----------------------------------

total_images = len(rows)

split_counts = Counter(
    row["split"]
    for row in rows
)

class_counts = Counter(
    row["class"]
    for row in rows
)

# -----------------------------------
# PATH CHECK
# -----------------------------------

missing_paths = []

for row in rows:

    image_path = row["image_path"]

    if not os.path.isfile(image_path):
        missing_paths.append(image_path)

# -----------------------------------
# CLASS + SPLIT COUNTS
# -----------------------------------

class_split_counts = defaultdict(
    Counter
)

for row in rows:

    class_name = row["class"]
    split = row["split"]

    class_split_counts[
        class_name
    ][split] += 1

# -----------------------------------
# UNEXPECTED SPLITS
# -----------------------------------

valid_splits = {
    "train",
    "validation",
    "test"
}

unexpected_splits = [
    split
    for split in split_counts
    if split not in valid_splits
]

# -----------------------------------
# DISPLAY RESULTS
# -----------------------------------

print("===================================")
print("NOURISHIQ DATASET SPLIT VERIFICATION")
print("===================================\n")

print("TOTAL IMAGES")
print("-------------")
print(f"Total entries : {total_images}")

print("\nSPLIT DISTRIBUTION")
print("------------------")
print(f"Train       : {split_counts['train']}")
print(f"Validation  : {split_counts['validation']}")
print(f"Test        : {split_counts['test']}")

print("\nCLASS COUNT")
print("-----------")
print(f"Total classes : {len(class_counts)}")

print("\nCLASS-WISE SPLIT")
print("----------------")

for class_name in sorted(class_split_counts):

    counts = class_split_counts[class_name]

    print(
        f"{class_name}: "
        f"Train={counts['train']}, "
        f"Validation={counts['validation']}, "
        f"Test={counts['test']}"
    )

# -----------------------------------
# PATH VERIFICATION
# -----------------------------------

print("\nIMAGE PATH VERIFICATION")
print("-----------------------")

print(
    f"Missing image paths : "
    f"{len(missing_paths)}"
)

if missing_paths:

    print("\nFirst missing paths:")

    for path in missing_paths[:10]:
        print(path)

else:

    print("All image paths are valid.")

# -----------------------------------
# SPLIT VERIFICATION
# -----------------------------------

print("\nSPLIT CHECK")
print("-----------")

if (
    split_counts["train"] == 9912
    and
    split_counts["validation"] == 2112
    and
    split_counts["test"] == 2136
):

    print("Train/Validation/Test counts: PASS")

else:

    print("Train/Validation/Test counts: CHECK REQUIRED")

# -----------------------------------
# CLASS VERIFICATION
# -----------------------------------

expected_class_count = 24

if len(class_counts) == expected_class_count:
    print("Number of classes: PASS")
else:
    print("Number of classes: CHECK REQUIRED")

# -----------------------------------
# EACH CLASS VERIFICATION
# -----------------------------------

class_error = False

for class_name in sorted(class_split_counts):

    counts = class_split_counts[class_name]

    if (
        counts["train"] != 413
        or
        counts["validation"] != 88
        or
        counts["test"] != 89
    ):

        class_error = True

        print(
            f"Problem in class: {class_name}"
        )

if not class_error:
    print("Class-wise split counts: PASS")

# -----------------------------------
# FINAL STATUS
# -----------------------------------

print("\n===================================")

if (
    total_images == 14160
    and
    len(class_counts) == 24
    and
    split_counts["train"] == 9912
    and
    split_counts["validation"] == 2112
    and
    split_counts["test"] == 2136
    and
    len(missing_paths) == 0
    and
    len(unexpected_splits) == 0
    and
    not class_error
):

    print("FINAL VERIFICATION: PASS")

else:

    print("FINAL VERIFICATION: CHECK REQUIRED")

print("===================================")