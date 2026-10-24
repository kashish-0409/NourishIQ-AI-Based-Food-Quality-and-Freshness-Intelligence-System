
import os
import hashlib
from collections import defaultdict

import pandas as pd

CSV_PATH = (
    r"C:\Users\kashu\OneDrive\Desktop"
    r"\AgriFreshNET Freshness and Shelf-Life Image Datase"
    r"\AgriFreshNET Freshness and Shelf-Life Image Datase"
    r"\Processed Data\NourishIQ_Dataset_Split\dataset_split.csv"
)

OUTPUT_PATH = os.path.join(
    "results", "duplicate_group_analysis.csv"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    print("Analyzing duplicate images...")
    df = pd.read_csv(CSV_PATH)

    required = {"image_path", "class", "split"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain these columns: {required}"
        )

    groups = defaultdict(list)
    missing = []

    for index, row in df.iterrows():
        path = str(row["image_path"]).strip()

        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(CSV_PATH), path)

        if not os.path.isfile(path):
            missing.append(path)
            continue

        try:
            digest = sha256_file(path)
        except OSError:
            missing.append(path)
            continue

        groups[digest].append({
            "csv_row": index + 1,
            "image_path": path,
            "class": str(row["class"]),
            "split": str(row["split"]),
        })

    output_rows = []

    for digest, records in groups.items():
        if len(records) < 2:
            continue

        labels = sorted({r["class"] for r in records})
        splits = sorted({r["split"] for r in records})

        for record in records:
            output_rows.append({
                "sha256": digest,
                "duplicate_count": len(records),
                "number_of_distinct_labels": len(labels),
                "all_labels": " | ".join(labels),
                "all_splits": " | ".join(splits),
                "cross_split_duplicate": len(splits) > 1,
                "label_inconsistency_candidate": len(labels) > 1,
                "csv_row": record["csv_row"],
                "class": record["class"],
                "split": record["split"],
                "image_path": record["image_path"],
            })

    result = pd.DataFrame(output_rows)

    os.makedirs("results", exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    cross_split_groups = (
        result.loc[result["cross_split_duplicate"], "sha256"].nunique()
        if not result.empty else 0
    )

    inconsistent_groups = (
        result.loc[
            result["label_inconsistency_candidate"], "sha256"
        ].nunique()
        if not result.empty else 0
    )

    print("\nDUPLICATE ANALYSIS SUMMARY")
    print("=" * 35)
    print(f"CSV rows: {len(df)}")
    print(f"Missing/unreadable files: {len(missing)}")
    print(f"Duplicate groups: {result['sha256'].nunique() if not result.empty else 0}")
    print(f"Groups spanning multiple splits: {cross_split_groups}")
    print(f"Groups with multiple labels: {inconsistent_groups}")
    print(f"Detailed report: {OUTPUT_PATH}")

    if missing:
        print("\nFirst missing/unreadable paths:")
        for path in missing[:10]:
            print(path)

    print("\nAnalysis complete. No source files were modified.")


if __name__ == "__main__":
    main()
