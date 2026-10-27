# Released TasteMM benchmark table

`compound.csv` is the current curated compound-level table for the six-class TasteMM project. It is UTF-8 CSV with a byte-order mark and can be opened directly in Excel, R, Python, or a text editor.

## Current dataset update on 2026-10-27

`compound.csv` now contains the author's revised 12,706-row single-label `dataset.csv`. Class counts remain unchanged. This revision replaces excluded cross-label-conflict and composition/identity records with new source-annotated entries and retains the evidence-based citric-acid label correction. All CSV values and row order match the supplied file; only CRLF line endings were normalized to LF for this repository.

- Supplied file SHA-256: `65c0180c938f341a547fcd65ead209bddf7edba9dbec45f8b2efccb884b9e0c4`.
- Repository CSV SHA-256: `4b3fc2b850254cd1769a807941629b55e512da4bb7e09691d7ebd8a7dd68871b`.
- Audit regenerated with RDKit 2024.03.5: 0 invalid SMILES; 0 canonical structures associated with multiple labels (0 affected rows).
- Same-label duplicates remain: 10,885 distinct canonical structures across 12,706 records. This update is not a claim of complete structural deduplication or independent experimental confirmation of every database annotation.
- The previous repository data remain available in Git history at commit `11c0b7e`. This update does not upload the separate multilabel evaluation set, retrain models, or recompute existing benchmark manifests/results. Regenerate structure-dependent preprocessing and splits before training on this revision.

## Previous dataset update on 2026-10-26

The table was replaced with the author's updated `dataset.csv` from the second revision. Its GB18030 text encoding was converted to UTF-8 and line endings were normalized to LF; all CSV field values and row order were preserved. Relative to the previous table, 7,904 SMILES fields and 13 compound names changed. IDs, labels, row order, and class counts are unchanged.

- Original supplied file SHA-256: `dff36db3ae0476bda1280370f8199bd45c459e4d785c781c70231a4aed09b0b5`.
- Repository CSV SHA-256: `13c556b7d5a4e81906a29578a787a6a852a598dd658200716b0ca90d3916f6a9`.
- The files under `audit/` were regenerated for this update using RDKit 2024.03.5.

Existing benchmark manifests and desktop model packages were produced before this dataset update. This replacement does not retrain those models or recompute benchmark results. The earlier dataset remains available in Git history at commit `41d333bc0ff6ac332a85e680395c641a7d13c9f7`; use that snapshot when reproducing artifacts tied to the earlier table. Structure-dependent sample identifiers and manifests must be regenerated before running new benchmarks against the updated table.

## Columns

| Column | Meaning |
| --- | --- |
| `ID` | Source/curation identifier retained when available; it is not a unique key and 3,101 rows are missing it |
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

The current canonical-isomeric-SMILES audit identifies 0 structures associated with more than one label (0 rows). The file `audit/cross_label_structures.csv` therefore contains only its header. This within-table check does not establish that a compound cannot have multiple tastes under other experimental conditions; same-label structural duplicates remain.
