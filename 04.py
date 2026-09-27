import os
from PIL import Image
from collections import Counter

# Dataset path
dataset_path = r"C:\Users\kashu\OneDrive\Desktop\AgriFreshNET Freshness and Shelf-Life Image Datase\AgriFreshNET Freshness and Shelf-Life Image Datase\Processed Data\Processed Data"

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

total_images = 0
readable_images = 0
corrupt_images = 0

format_counts = Counter()
dimension_counts = Counter()

corrupt_files = []


# -----------------------------------
# CHECK ALL IMAGES
# -----------------------------------

for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):

        if not file.lower().endswith(image_extensions):
            continue

        total_images += 1

        file_path = os.path.join(folder_path, file)

        # File format
        extension = os.path.splitext(file)[1].lower()
        format_counts[extension] += 1

        # Try opening image
        try:

            with Image.open(file_path) as img:

                # Verify image integrity
                img.verify()

            # Open again because verify() closes the image
            with Image.open(file_path) as img:

                width, height = img.size

                dimension_counts[(width, height)] += 1

            readable_images += 1

        except Exception:

            corrupt_images += 1
            corrupt_files.append(file_path)


# -----------------------------------
# RESULTS
# -----------------------------------

print("\n==============================")
print("IMAGE QUALITY CHECK")
print("==============================")

print("\nTotal images:", total_images)

print("Readable images:", readable_images)

print("Corrupt / unreadable images:", corrupt_images)


print("\n==============================")
print("FILE FORMAT DISTRIBUTION")
print("==============================")

for fmt, count in format_counts.items():

    print(f"{fmt}: {count}")


print("\n==============================")
print("IMAGE DIMENSIONS")
print("==============================")

for dimension, count in dimension_counts.most_common():

    print(
        f"{dimension[0]} x {dimension[1]} : {count} images"
    )


print("\n==============================")
print("CORRUPT FILES")
print("==============================")

if corrupt_files:

    for file in corrupt_files:
        print(file)

else:

    print("No corrupt or unreadable images found.")