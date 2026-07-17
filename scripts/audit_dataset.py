"""Validate the released CSV and export human-readable audit tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

LABEL_NAMES = {1: "bitter", 2: "sweet", 3: "umami", 4: "salty", 5: "sour", 6: "tasteless"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/compound.csv")
    parser.add_argument("--output-dir", default="data/audit")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = ["ID", "Name", "SMILES", "Label"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    missing_id = df["ID"].isna() | df["ID"].astype(str).str.strip().eq("")
    df["Label"] = pd.to_numeric(df["Label"], errors="raise").astype(int)
    if not set(df["Label"]).issubset(LABEL_NAMES):
        raise ValueError(f"Unexpected labels: {sorted(set(df['Label']) - set(LABEL_NAMES))}")

    invalid_smiles = []
    canonical_smiles = []
    for index, smiles in enumerate(df["SMILES"].astype(str)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles.append(index)
            canonical_smiles.append("")
        else:
            canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
    df["Canonical_SMILES"] = canonical_smiles

    class_distribution = (
        df.groupby("Label", as_index=False)
        .size()
        .rename(columns={"size": "Count"})
    )
    class_distribution["Taste"] = class_distribution["Label"].map(LABEL_NAMES)
    class_distribution["Fraction"] = class_distribution["Count"] / len(df)
    class_distribution = class_distribution[["Label", "Taste", "Count", "Fraction"]]
    class_distribution.to_csv(output_dir / "class_distribution.csv", index=False, encoding="utf-8-sig")

    conflict_keys = (
        df.groupby("Canonical_SMILES")["Label"].nunique().loc[lambda series: series > 1].index
    )
    conflicts = df[df["Canonical_SMILES"].isin(conflict_keys)].sort_values(["Canonical_SMILES", "Label", "ID"])
    conflicts.to_csv(output_dir / "cross_label_structures.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_file": str(input_path),
        "rows": int(len(df)),
        "columns": required,
        "invalid_smiles_rows": int(len(invalid_smiles)),
        "missing_id_rows": int(missing_id.sum()),
        "duplicate_nonmissing_id_rows": int(df.loc[~missing_id].duplicated("ID", keep=False).sum()),
        "duplicate_exact_smiles_rows": int(df.duplicated("SMILES", keep=False).sum()),
        "unique_canonical_smiles": int(df["Canonical_SMILES"].nunique()),
        "canonical_structures_with_multiple_labels": int(len(conflict_keys)),
        "rows_in_cross_label_structures": int(len(conflicts)),
        "class_counts": {LABEL_NAMES[int(row.Label)]: int(row.Count) for row in class_distribution.itertuples()},
        "interpretation_note": (
            "This audit describes the released final table. It cannot recover how many records were excluded "
            "during source harmonization unless the pre-curation master table and exclusion log are also available. "
            "Missing source IDs are not counted as duplicate identifiers; row_index is the stable modeling key."
        ),
    }
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
