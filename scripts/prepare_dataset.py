"""
Regenerates data/image_manifest.csv and data/clinical_features.csv from the
raw unified_oai_dataset.csv + dataset/{train,val,test} image folders.

Re-created because the original outputs existed in a prior project folder
that wasn't carried over; only the raw CSV and images were available here.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "unified_oai_dataset.csv"
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_DIR = PROJECT_ROOT / "data"

# Columns identified in unified_oai_dataset.csv that give image/split/grade info.
ID_COLS = ["ID_STR", "SIDE_CODE", "KL_GRADE_DIR", "SPLIT", "IMAGE_NAME"]

# 12 clinical columns: age, BMI, WOMAC pain (R/L), WOMAC function (R/L),
# family history (hip/knee), JSN grade (R/L), osteophyte grade (R/L).
CLINICAL_COLS = [
    "V00AGE",
    "P01BMI",
    "V00WOMKPR", "V00WOMKPL",
    "V00WOMADLR", "V00WOMADLL",
    "P01FAMHR", "P01FAMKR",
    "P01SVRKJSL", "P01SVLKJSL",
    "P01SVRKOST", "P01SVLKOST",
]


def main():
    usecols = ID_COLS + CLINICAL_COLS
    df = pd.read_csv(RAW_CSV, usecols=usecols, dtype={"ID_STR": str})

    df = df.rename(columns={"ID_STR": "ID", "KL_GRADE_DIR": "KL_GRADE"})

    # auto_test is a near-duplicate of test (811/828 patients overlap) and is dropped.
    n_before = len(df)
    df = df[df["SPLIT"] != "auto_test"].copy()
    print(f"Dropped {n_before - len(df)} auto_test rows.")

    # Rebuild IMAGE_PATH relative to this project instead of the old Colab path.
    df["IMAGE_PATH"] = (
        "dataset/" + df["SPLIT"] + "/" + df["KL_GRADE"].astype(str) + "/" + df["IMAGE_NAME"]
    )

    missing = [
        p for p in df["IMAGE_PATH"] if not (PROJECT_ROOT / p).exists()
    ]
    if missing:
        print(f"WARNING: {len(missing)} image(s) referenced in the CSV are missing on disk, e.g.:")
        for p in missing[:5]:
            print(" ", p)
    else:
        print(f"All {len(df)} image paths verified on disk.")

    overlap_train_val = set(df.loc[df.SPLIT == "train", "ID"]) & set(df.loc[df.SPLIT == "val", "ID"])
    overlap_train_test = set(df.loc[df.SPLIT == "train", "ID"]) & set(df.loc[df.SPLIT == "test", "ID"])
    overlap_val_test = set(df.loc[df.SPLIT == "val", "ID"]) & set(df.loc[df.SPLIT == "test", "ID"])
    assert not overlap_train_val, f"train/val patient overlap: {len(overlap_train_val)}"
    assert not overlap_train_test, f"train/test patient overlap: {len(overlap_train_test)}"
    assert not overlap_val_test, f"val/test patient overlap: {len(overlap_val_test)}"
    print("No patient overlap between train/val/test.")

    DATA_DIR.mkdir(exist_ok=True)

    manifest_cols = ["ID", "SIDE_CODE", "SPLIT", "IMAGE_NAME", "IMAGE_PATH", "KL_GRADE"]
    manifest = df[manifest_cols]
    manifest.to_csv(DATA_DIR / "image_manifest.csv", index=False)
    print(f"Wrote {DATA_DIR / 'image_manifest.csv'} ({len(manifest)} rows)")

    clinical = df[manifest_cols + CLINICAL_COLS]
    clinical.to_csv(DATA_DIR / "clinical_features.csv", index=False)
    print(f"Wrote {DATA_DIR / 'clinical_features.csv'} ({len(clinical)} rows)")

    print("\nSplit counts:")
    print(df.groupby("SPLIT")["ID"].count())
    print("\nGrade distribution (train):")
    print(df.loc[df.SPLIT == "train", "KL_GRADE"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()
