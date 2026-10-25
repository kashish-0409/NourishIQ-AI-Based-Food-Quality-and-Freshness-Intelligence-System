
import os
import re
import hashlib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

SOURCE_CSV = (
    r"C:\Users\kashu\OneDrive\Desktop"
    r"\AgriFreshNET Freshness and Shelf-Life Image Datase"
    r"\AgriFreshNET Freshness and Shelf-Life Image Datase"
    r"\Processed Data\NourishIQ_Dataset_Split\dataset_split.csv"
)

RESULTS_DIR = (
    r"C:\Users\kashu\OneDrive\Desktop"
    r"\AgriFreshNET Freshness and Shelf-Life Image Datase"
    r"\results"
)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical_name(path):
    name = Path(path).name.lower()
    # aug_233_IMG_xxx.jpg -> img_xxx.jpg
    while re.match(r"^aug_\d+_", name):
        name = re.sub(r"^aug_\d+_", "", name, count=1)
    return name

def main():
    df = pd.read_csv(SOURCE_CSV)

    required = {"image_path", "class", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    df["image_path"] = df["image_path"].astype(str).str.strip()
    df["class"] = df["class"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip()

    hashes = []
    families = []
    missing = []

    for i, row in df.iterrows():
        path = row["image_path"]
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(SOURCE_CSV), path)

        if not os.path.isfile(path):
            hashes.append(None)
            families.append(f"MISSING_{i}")
            missing.append(path)
            continue

        try:
            hashes.append(sha256_file(path))
            families.append(canonical_name(path))
        except OSError:
            hashes.append(None)
            families.append(f"UNREADABLE_{i}")
            missing.append(path)

    df["sha256"] = hashes
    df["family"] = families

    # Union rows that share either a normalized filename or exact file hash.
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for column in ["family", "sha256"]:
        seen = {}
        for i, value in enumerate(df[column]):
            if pd.isna(value) or str(value).startswith(("MISSING_", "UNREADABLE_")):
                continue
            if value in seen:
                union(i, seen[value])
            else:
                seen[value] = i

    df["group_id"] = [find(i) for i in range(len(df))]

    # Exclude groups whose labels disagree; do not guess the correct label.
    label_counts = df.groupby("group_id")["class"].nunique()
    conflict_groups = set(label_counts[label_counts > 1].index)

    review = df[df["group_id"].isin(conflict_groups)].copy()
    clean = df[~df["group_id"].isin(conflict_groups)].copy()

    # Files that could not be read must not enter the clean split.
    unreadable = clean[clean["sha256"].isna()].copy()
    clean = clean[clean["sha256"].notna()].copy()

    if clean.empty:
        raise ValueError("No clean rows remain after review filtering.")

    # Split by group so related files cannot cross splits.
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    trainval_idx, test_idx = next(
        first.split(clean, groups=clean["group_id"])
    )

    trainval = clean.iloc[trainval_idx].copy()
    test = clean.iloc[test_idx].copy()

    second = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
    train_idx, val_idx = next(
        second.split(trainval, groups=trainval["group_id"])
    )

    train = trainval.iloc[train_idx].copy()
    val = trainval.iloc[val_idx].copy()

    train["split"] = "train"
    val["split"] = "validation"
    test["split"] = "test"

    output = pd.concat([train, val, test], ignore_index=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "clean_dataset_split.csv")
    review_path = os.path.join(RESULTS_DIR, "label_conflicts_for_review.csv")
    unreadable_path = os.path.join(RESULTS_DIR, "unreadable_images_for_review.csv")

    output[["image_path", "class", "split"]].to_csv(output_path, index=False)
    review.to_csv(review_path, index=False)
    unreadable.to_csv(unreadable_path, index=False)

    print("\nCLEAN SPLIT CREATED")
    print("Original dataset was not modified.")
    print("Clean rows:", len(output))
    print("Conflicting-label rows excluded:", len(review))
    print("Missing/unreadable rows excluded:", len(unreadable))
    print("\nSplit counts:")
    print(output["split"].value_counts())
    print("\nClass counts per split:")
    print(pd.crosstab(output["class"], output["split"]))
    print("\nSaved:", output_path)
    print("Review conflicts:", review_path)
    print("Review unreadable:", unreadable_path)

if __name__ == "__main__":
    main()
