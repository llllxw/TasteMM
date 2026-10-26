# Released TasteMM benchmark table

`compound.csv` is the current curated compound-level table for the six-class TasteMM project. It is UTF-8 CSV with a byte-order mark and can be opened directly in Excel, R, Python, or a text editor.

## Dataset update on 2026-10-26

The table was replaced with the author's updated `dataset.csv` from the second revision. Its GB18030 text encoding was converted to UTF-8 and line endings were normalized to LF; all CSV field values and row order were preserved. Relative to the previous table, 7,904 SMILES fields and 13 compound names changed. IDs, labels, row order, and class counts are unchanged.

- Original supplied file SHA-256: `dff36db3ae0476bda1280370f8199bd45c459e4d785c781c70231a4aed09b0b5`.
- Repository CSV SHA-256: `13c556b7d5a4e81906a29578a787a6a852a598dd658200716b0ca90d3916f6a9`.
- The files under `audit/` were regenerated for this update using RDKit 2024.03.5.

Existing benchmark manifests and desktop model packages were produced before this dataset update. This replacement does not retrain those models or recompute benchmark results. The earlier dataset remains available in Git history at commit `41d333bc0ff6ac332a85e680395c641a7d13c9f7`; use that snapshot when reproducing artifacts tied to the earlier table. Structure-dependent sample identifiers and manifests must be regenerated before running new benchmarks against the updated table.

## Columns

| Column | Meaning |
| --- | --- |
| `ID` | Source/curation identifier retained when available; it is not a unique key and 3,201 rows are missing it |
| `Name` | Compound name retained during curation |
| `SMILES` | Molecular structure string used to construct all three model inputs |
| `Label` | Integer taste class; see `label_mapping.csv` |

The final table contains 12,706 rows: 3,005 bitter, 2,635 sweet, 328 umami, 28 salty, 1,700 sour, and 5,010 tasteless records. Because salty contains only 28 records, salty-specific estimates should be interpreted cautiously.

## Audit files

Run `python scripts/audit_dataset.py` from the repository root to regenerate:

- `audit/class_distribution.csv`: class counts and fractions;
- `audit/cross_label_structures.csv`: canonical structures associated with more than one label in the released table;
- `audit/dataset_audit.json`: machine-readable row, validity, duplicate, and conflict counts.

The released final table does not contain the pre-curation records or a row-level exclusion log. It therefore cannot, by itself, establish the number of ambiguous records removed during harmonization. This limitation is stated explicitly so that an exclusion count is not inferred from the final table.

The stable key for the released CSV is its zero-based `row_index`; benchmark manifests additionally provide a deterministic `sample_uid`. Neither `ID` nor `SMILES` should be used as a unique key.

The current canonical-SMILES audit identifies 110 structures associated with more than one label (251 rows). Accordingly, “single-label” describes the learning target attached to each row; it should not be interpreted as proof that every chemical structure has only one possible taste. The complete rows are reported in `audit/cross_label_structures.csv`.
