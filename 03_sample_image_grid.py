import os
import matplotlib.pyplot as plt
from PIL import Image

# Dataset path
dataset_path = r"C:\Users\kashu\OneDrive\Desktop\AgriFreshNET Freshness and Shelf-Life Image Datase\AgriFreshNET Freshness and Shelf-Life Image Datase\Processed Data\Processed Data"

# Food types
foods = [
    "Banana",
    "Bittermelon",
    "Cucumber",
    "Eggplant",
    "Orange",
    "Papaya",
    "Pineapple",
    "Tomato"
]

# Freshness stages
stages = [
    "Fresh",
    "Semi Fresh",
    "Rotten"
]


# -----------------------------------
# FIND CORRECT FOLDER
# -----------------------------------

def find_folder(food, stage):

    for folder in os.listdir(dataset_path):

        folder_lower = folder.lower()

        # Make underscores and spaces equivalent
        folder_normalized = folder_lower.replace("_", " ")
        stage_normalized = stage.lower().replace("_", " ")

        if (
            food.lower() in folder_normalized
            and stage_normalized in folder_normalized
        ):
            return os.path.join(dataset_path, folder)

    return None


# -----------------------------------
# CREATE FIGURE
# -----------------------------------

fig, axes = plt.subplots(
    nrows=8,
    ncols=3,
    figsize=(11, 22)
)


# -----------------------------------
# ADD IMAGES
# -----------------------------------

for i, food in enumerate(foods):

    for j, stage in enumerate(stages):

        ax = axes[i, j]

        folder_path = find_folder(food, stage)

        if folder_path is None:

            ax.text(
                0.5,
                0.5,
                "Image not found",
                ha="center",
                va="center",
                fontsize=10
            )

            ax.set_xlabel(
                f"{food}\n{stage}",
                fontsize=10
            )

            ax.set_xticks([])
            ax.set_yticks([])

            continue


        # Get image files
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            )
        ])


        if len(image_files) == 0:

            ax.text(
                0.5,
                0.5,
                "No image",
                ha="center",
                va="center"
            )

            ax.set_xticks([])
            ax.set_yticks([])

            continue


        # Select first image
        image_path = os.path.join(
            folder_path,
            image_files[0]
        )


        try:

            image = Image.open(image_path)

            ax.imshow(image)

            # Remove axis ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Remove borders
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Name BELOW image
            ax.set_xlabel(
                f"{food} — {stage}",
                fontsize=10,
                labelpad=8
            )

        except Exception:

            ax.text(
                0.5,
                0.5,
                "Unreadable",
                ha="center",
                va="center"
            )

            ax.set_xticks([])
            ax.set_yticks([])


# -----------------------------------
# MAIN TITLE
# -----------------------------------

fig.suptitle(
    "Representative Images Across Food Types and Freshness Stages",
    fontsize=17,
    fontweight="bold",
    y=0.995
)


# -----------------------------------
# SPACING
# -----------------------------------

plt.subplots_adjust(
    left=0.08,
    right=0.92,
    top=0.96,
    bottom=0.025,
    wspace=0.28,
    hspace=0.85
)


# -----------------------------------
# SAVE
# -----------------------------------

plt.savefig(
    "sample_image_grid.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()