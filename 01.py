import os
from collections import defaultdict

dataset_path = r"C:\Users\kashu\OneDrive\Desktop\AgriFreshNET Freshness and Shelf-Life Image Datase\AgriFreshNET Freshness and Shelf-Life Image Datase\Processed Data\Processed Data"

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

class_counts = {}
total_images = 0

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        count = 0

        for file in os.listdir(folder_path):
            if file.lower().endswith(image_extensions):
                count += 1

        class_counts[folder] = count
        total_images += count

print("TOTAL IMAGES:", total_images)
print("\nCLASS-WISE IMAGE COUNT:\n")

for folder, count in sorted(class_counts.items()):
    print(f"{folder}: {count}")