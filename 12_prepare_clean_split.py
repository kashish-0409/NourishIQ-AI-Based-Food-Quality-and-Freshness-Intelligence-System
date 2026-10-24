
import os
import hashlib
import pandas as pd
from collections import defaultdict

CSV_PATH = (
    r"C:\Users\kashu\OneDrive\Desktop"
    r"\AgriFreshNET Freshness and Shelf-Life Image Datase"
    r"\AgriFreshNET Freshness and Shelf-Life Image Datase"
    r"\Processed Data\NourishIQ_Dataset_Split\dataset_split.csv"
)

OUTPUT_DIR = "results"
OUTPUT_CSV = os.path.join(
    OUTPUT_DIR, "dataset_split_review.csv"
)
REVIEW_CSV = os.path.join(
    OUTPUT_DIR, "duplicate_label_conflicts.csv"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    df = pd.read_csv(CSV_PATH)

    required = {"image_path", "class", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV needs columns: {required}")

    hashes = []
    missing = []

    for _, row in df.iterrows():
        path = str(row["image_path"]).strip()

        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(CSV_PATH), path)

        if not os.path.isfile(path):
            hashes.append(None)
            missing.append(path)
            continue

        hashes.append(sha256_file(path))

    df["sha256"] = hashes
    df["review_status"] = "ok"

    # Group identical file contents.
    valid = df[df["sha256"].notna()]
    label_counts = valid.groupby("sha256")["class"].nunique()
    split_counts = valid.groupby("sha256")["split"].nunique()

    conflict_hashes = set(label_counts[label_counts > 1].index)
    leakage_hashes = set(split_counts[split_counts > 1].index)

    df.loc[df["sha256"].isin(conflict_hashes),
           "review_status"] = "REVIEW_LABEL_CONFLICT"

    df.loc[
        df["sha256"].isin(leakage_hashes)
        & ~df["sha256"].isin(conflict_hashes),
        "review_status"
    ] = "REVIEW_CROSS_SPLIT_DUPLICATE"

    # Keep this as a review copy only. Do not train on it yet.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    review = df[df["sha256"].isin(
        conflict_hashes | leakage_hashes
    )].copy()
    review.to_csv(REVIEW_CSV, index=False)

    print("Review files created successfully.")
    print(f"Total CSV rows: {len(df)}")
    print(f"Missing image paths: {len(missing)}")
    print(f"Label-conflict groups: {len(conflict_hashes)}")
    print(f"Cross-split duplicate groups: {len(leakage_hashes)}")
    print(f"Review CSV: {OUTPUT_CSV}")
    print(f"Flagged rows: {REVIEW_CSV}")
    print("Original dataset and models were not modified.")
    print("Do NOT train on the review CSV yet.")


if __name__ == "__main__":
    main()
