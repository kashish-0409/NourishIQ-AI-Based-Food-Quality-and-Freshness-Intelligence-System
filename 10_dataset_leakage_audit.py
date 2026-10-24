
import os
import hashlib
import pandas as pd
from collections import defaultdict

# Existing dataset CSV
CSV_PATH = (
    r"C:\Users\kashu\OneDrive\Desktop"
    r"\AgriFreshNET Freshness and Shelf-Life Image Datase"
    r"\AgriFreshNET Freshness and Shelf-Life Image Datase"
    r"\Processed Data\NourishIQ_Dataset_Split\dataset_split.csv"
)

OUTPUT_PATH = os.path.join("results", "dataset_leakage_audit.txt")


def sha256_file(file_path):
    """Return SHA-256 hash of a file's contents."""
    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    print("Starting dataset leakage audit...")
    print("Reading dataset CSV...")

    df = pd.read_csv(CSV_PATH)

    required_columns = {"image_path", "split"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {required_columns}"
        )

    print(f"CSV rows: {len(df)}")

    hashes = defaultdict(list)
    missing_files = []
    checked_files = 0

    for index, row in df.iterrows():
        image_path = str(row["image_path"]).strip()
        split = str(row["split"]).strip()

        # Support paths stored relative to the CSV folder.
        if not os.path.isabs(image_path):
            image_path = os.path.join(
                os.path.dirname(CSV_PATH), image_path
            )

        if not os.path.isfile(image_path):
            missing_files.append(image_path)
            continue

        try:
            file_hash = sha256_file(image_path)
            hashes[file_hash].append({
                "path": image_path,
                "split": split,
                "row": index + 1
            })
            checked_files += 1
        except OSError as error:
            print(f"Could not read: {image_path} ({error})")
            missing_files.append(image_path)

        if checked_files and checked_files % 1000 == 0:
            print(f"Checked {checked_files} images...")

    cross_split_duplicates = []
    same_split_duplicates = []

    for file_hash, records in hashes.items():
        if len(records) < 2:
            continue

        splits = {record["split"] for record in records}

        if len(splits) > 1:
            cross_split_duplicates.append((file_hash, records))
        else:
            same_split_duplicates.append((file_hash, records))

    report = []
    report.append("NOURISHIQ DATASET LEAKAGE AUDIT")
    report.append("=" * 40)
    report.append(f"CSV rows: {len(df)}")
    report.append(f"Files successfully checked: {checked_files}")
    report.append(f"Missing/unreadable files: {len(missing_files)}")
    report.append(
        f"Exact duplicate groups across splits: "
        f"{len(cross_split_duplicates)}"
    )
    report.append(
        f"Exact duplicate groups within one split: "
        f"{len(same_split_duplicates)}"
    )
    report.append("")
    report.append(
        "CROSS-SPLIT DUPLICATES "
        "(same file contents found in different splits)"
    )

    if not cross_split_duplicates:
        report.append("No exact cross-split duplicates found.")

    for group_number, (file_hash, records) in enumerate(
        cross_split_duplicates, start=1
    ):
        report.append(f"\nGroup {group_number} | SHA-256: {file_hash}")
        for record in records:
            report.append(
                f"  Split: {record['split']} | "
                f"CSV row: {record['row']} | "
                f"Path: {record['path']}"
            )

    report.append("\nMISSING OR UNREADABLE FILES")
    for path in missing_files:
        report.append(path)

    report_text = "\n".join(report)

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        file.write(report_text)

    print("\n" + report_text)
    print(f"\nReport saved to: {OUTPUT_PATH}")
    print("\nAudit complete. Original dataset and models were not changed.")


if __name__ == "__main__":
    main()
