# Released TasteMM benchmark table

`compound.csv` is the final compound-level table used for the six-class TasteMM experiments. It is plain UTF-8 CSV and can be opened directly in Excel, R, Python, or a text editor.

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

The current canonical-SMILES audit identifies 115 structures associated with more than one label (282 rows). Accordingly, “single-label” describes the learning target attached to each row; it should not be interpreted as proof that every chemical structure has only one possible taste. The complete rows are reported in `audit/cross_label_structures.csv`.
