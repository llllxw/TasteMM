from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from sklearn.model_selection import StratifiedKFold


LABEL_NAMES = {
    1: "bitter",
    2: "sweet",
    3: "umami",
    4: "salty",
    5: "sour",
    6: "tasteless",
}

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "compound.csv"
DEFAULT_OUTPUT = HERE / "manifests"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_uid(row_index: int, smiles: str, label: int) -> str:
    payload = f"{row_index}|{smiles}|{label}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def reproduce_tastemm_splits(df: pd.DataFrame) -> pd.DataFrame:
    labels = df["Label"].astype(int).to_numpy()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows: list[dict] = []

    for fold, (train_idx, val_test_idx) in enumerate(outer.split(df, labels)):
        held_labels = labels[val_test_idx]
        val_rel = test_rel = None
        selected_attempt = None
        for attempt in range(100):
            inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=42 + attempt)
            val_rel, test_rel = next(inner.split(val_test_idx, held_labels))
            if len(set(held_labels[val_rel])) == 6 and len(set(held_labels[test_rel])) == 6:
                selected_attempt = attempt
                break
        if selected_attempt is None or val_rel is None or test_rel is None:
            raise RuntimeError(f"Could not create a six-class validation/test split for fold {fold}.")

        partitions = {
            "train": train_idx,
            "val": val_test_idx[val_rel],
            "test": val_test_idx[test_rel],
        }
        for partition, indices in partitions.items():
            for idx in indices:
                row = df.iloc[int(idx)]
                rows.append(
                    {
                        "row_index": int(idx),
                        "sample_uid": sample_uid(int(idx), str(row["SMILES"]), int(row["Label"])),
                        "fold": fold,
                        "split_id": f"fold{fold}_seed42",
                        "partition": partition,
                        "inner_attempt": selected_attempt,
                        "source_id": row["ID"],
                        "name": row["Name"],
                        "smiles": row["SMILES"],
                        "true_label": int(row["Label"]),
                        "true_label_name": LABEL_NAMES[int(row["Label"])],
                    }
                )
    return pd.DataFrame(rows)


def canonical_structure_overlap(manifest: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    base = manifest.sort_values("row_index").drop_duplicates("row_index")
    canonical_by_row = {}
    for row in base.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.smiles))
        if mol is None:
            raise ValueError(f"Invalid SMILES in manifest row {row.row_index}: {row.smiles!r}")
        canonical_by_row[int(row.row_index)] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    result = {}
    for fold in range(5):
        fold_rows = manifest[manifest["fold"] == fold]
        train_rows = fold_rows[fold_rows["partition"] == "train"]["row_index"].astype(int)
        test_rows = fold_rows[fold_rows["partition"] == "test"]["row_index"].astype(int)
        train_structures = {canonical_by_row[index] for index in train_rows}
        test_canonical = [canonical_by_row[index] for index in test_rows]
        overlapping_rows = sum(value in train_structures for value in test_canonical)
        result[str(fold)] = {
            "test_rows": int(len(test_rows)),
            "test_rows_with_structure_in_train": int(overlapping_rows),
            "fraction_test_rows_with_structure_in_train": float(overlapping_rows / len(test_rows)),
            "unique_test_structures_also_in_train": int(len(set(test_canonical).intersection(train_structures))),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the exact five matched TasteMM benchmark splits.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    required = {"ID", "Name", "SMILES", "Label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(df) != 12706:
        raise ValueError(f"Expected the final 12,706-row dataset, found {len(df):,} rows.")
    observed_labels = set(df["Label"].astype(int).unique())
    if observed_labels != set(LABEL_NAMES):
        raise ValueError(f"Expected labels 1-6, found {sorted(observed_labels)}")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = reproduce_tastemm_splits(df)
    manifest.to_csv(args.output / "six_class_split_manifest.csv", index=False, encoding="utf-8-sig")

    test_manifest = manifest[manifest["partition"] == "test"].copy()
    test_manifest.to_csv(args.output / "six_class_test_manifest.csv", index=False, encoding="utf-8-sig")

    class_counts = df["Label"].astype(int).value_counts().sort_index()
    audit = {
        "source_file": args.data.name,
        "source_file_sha256": file_sha256(args.data),
        "split_manifest_sha256": file_sha256(args.output / "six_class_split_manifest.csv"),
        "n_rows": int(len(df)),
        "n_unique_ids": int(df["ID"].nunique(dropna=False)),
        "n_unique_smiles": int(df["SMILES"].nunique(dropna=False)),
        "missing_id_rows": int(df["ID"].isna().sum()),
        "duplicate_nonmissing_id_rows_beyond_first": int(df["ID"].dropna().duplicated().sum()),
        "duplicate_smiles_rows_beyond_first": int(len(df) - df["SMILES"].nunique(dropna=False)),
        "class_counts": {LABEL_NAMES[int(k)]: int(v) for k, v in class_counts.items()},
        "split_method": "Five matched stratified 80/10/10 row-level train/validation/test splits with random_state=42",
        "test_rows_per_fold": {
            str(k): int(v)
            for k, v in test_manifest.groupby("fold").size().to_dict().items()
        },
        "canonical_structure_overlap": canonical_structure_overlap(manifest),
        "warning": "Repeated structures may cross folds because the original TasteMM split was row-level.",
    }
    with open(args.output / "dataset_and_split_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)

    print(f"Saved manifest: {args.output / 'six_class_split_manifest.csv'}")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
