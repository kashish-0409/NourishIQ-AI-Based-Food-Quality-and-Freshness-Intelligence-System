import os
import matplotlib.pyplot as plt
from collections import Counter

dataset_path = r"C:\Users\kashu\OneDrive\Desktop\AgriFreshNET Freshness and Shelf-Life Image Datase\AgriFreshNET Freshness and Shelf-Life Image Datase\Processed Data\Processed Data"

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

class_counts = {}

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        count = 0

        for file in os.listdir(folder_path):
            if file.lower().endswith(image_extensions):
                count += 1

        class_counts[folder] = count


# -----------------------------
# 1. CLASS-WISE DISTRIBUTION
# -----------------------------

classes = list(class_counts.keys())
counts = list(class_counts.values())

plt.figure(figsize=(14, 7))
plt.bar(classes, counts, color="skyblue")
for i, value in enumerate(counts):
    plt.text(i, value + 5, str(value), ha='center', fontsize=8)
plt.xlabel("Food Freshness Class")
plt.ylabel("Number of Images")
plt.title("Class-wise Image Distribution")

plt.xticks(rotation=90)
plt.tight_layout()

plt.savefig("class_distribution.png", dpi=300)
plt.show()


# -----------------------------
# 2. FRESHNESS DISTRIBUTION
# -----------------------------

freshness_counts = {
    "Fresh": 0,
    "Semi Fresh": 0,
    "Rotten": 0
}

for folder, count in class_counts.items():

    name = folder.lower()

    if name.startswith("fresh "):
        freshness_counts["Fresh"] += count

    elif name.startswith("semi"):
        freshness_counts["Semi Fresh"] += count

    elif name.startswith("rotten"):
        freshness_counts["Rotten"] += count


plt.figure(figsize=(7, 5))
plt.bar(
    freshness_counts.keys(),
    freshness_counts.values(),
    color=["lightgreen", "khaki", "lightcoral"]
)

plt.xlabel("Freshness Stage")
plt.ylabel("Number of Images")
plt.title("Freshness Stage Distribution")

plt.tight_layout()

plt.savefig("freshness_distribution.png", dpi=300)
plt.show()


# -----------------------------
# 3. FOOD-WISE DISTRIBUTION
# -----------------------------

food_counts = {
    "Banana": 0,
    "Bittermelon": 0,
    "Cucumber": 0,
    "Eggplant": 0,
    "Orange": 0,
    "Papaya": 0,
    "Pineapple": 0,
    "Tomato": 0
}

for folder, count in class_counts.items():

    name = folder.lower()

    for food in food_counts:

        if food.lower() in name:
            food_counts[food] += count
            break


food_colors = [
    "#F4D35E",  # Banana
    "#7CB342",  # Bittermelon
    "#8BC34A",  # Cucumber
    "#9575CD",  # Eggplant
    "#F4A261",  # Orange
    "#F6A57A",  # Papaya
    "#D4A72C",  # Pineapple
    "#E76F51"   # Tomato
]

plt.figure(figsize=(10, 6))

plt.bar(
    food_counts.keys(),
    food_counts.values(),
    color=food_colors
)

plt.xlabel("Food Type")
plt.ylabel("Number of Images")
plt.title("Food-wise Image Distribution")

plt.tight_layout()

plt.savefig("food_distribution.png", dpi=300)
plt.show()


# -----------------------------
# PRINT RESULTS
# -----------------------------

print("\nFRESHNESS DISTRIBUTION:")
for key, value in freshness_counts.items():
    print(key, ":", value)

print("\nFOOD-WISE DISTRIBUTION:")
for key, value in food_counts.items():
    print(key, ":", value)