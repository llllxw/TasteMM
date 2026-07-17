from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "compound.csv"


@dataclass(frozen=True)
class ScopeTask:
    key: str
    filename_prefix: str
    task_id: str
    class_names: tuple[str, ...]
    source_to_scope: dict[int, int]
    include_labels: tuple[int, ...] | None = None


TASKS = (
    ScopeTask(
        key="scope3", filename_prefix="tastemolnet_scope3",
        task_id="scope3_bitter_sweet_tasteless_10650",
        class_names=("bitter", "sweet", "tasteless"),
        source_to_scope={1: 1, 2: 2, 6: 3}, include_labels=(1, 2, 6),
    ),
    ScopeTask(
        key="scope4", filename_prefix="virtuous_scope4",
        task_id="scope4_bitter_sweet_umami_other_12706",
        class_names=("bitter", "sweet", "umami", "other"),
        source_to_scope={1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 4},
    ),
    ScopeTask(
        key="scope5", filename_prefix="fart_scope5",
        task_id="scope5_bitter_sweet_sour_umami_undefined_12706",
        class_names=("bitter", "sweet", "sour", "umami", "undefined"),
        source_to_scope={1: 1, 2: 2, 5: 3, 3: 4, 4: 5, 6: 5},
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uid(task_key: str, source_row_index: int, smiles: str, source_label: int) -> str:
    raw = f"{task_key}|{source_row_index}|{smiles}|{source_label}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def build_task(source: pd.DataFrame, task: ScopeTask) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scope = source.copy()
    if task.include_labels is not None:
        scope = scope[scope["Label"].isin(task.include_labels)].copy()
    scope = scope.reset_index(drop=True)
    scope["scope_row_index"] = np.arange(len(scope))
    scope["scope_label"] = scope["Label"].map(task.source_to_scope).astype(int)
    scope["scope_label_name"] = scope["scope_label"].map(
        {index + 1: name for index, name in enumerate(task.class_names)}
    )
    scope["sample_uid"] = [
        uid(task.key, int(row.source_row_index), str(row.SMILES), int(row.Label))
        for row in scope.itertuples(index=False)
    ]

    rows: list[dict] = []
    y = scope["scope_label"].to_numpy()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, held_idx) in enumerate(outer.split(scope, y)):
        held_y = y[held_idx]
        inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        val_rel, test_rel = next(inner.split(held_idx, held_y))
        for partition, indices in {
            "train": train_idx, "val": held_idx[val_rel], "test": held_idx[test_rel]
        }.items():
            for idx in indices:
                row = scope.iloc[int(idx)]
                rows.append({
                    "scope_row_index": int(row["scope_row_index"]),
                    "source_row_index": int(row["source_row_index"]),
                    "sample_uid": row["sample_uid"],
                    "fold": fold,
                    "split_id": f"{task.key}_fold{fold}_seed42",
                    "partition": partition,
                    "source_id": row["ID"],
                    "name": row["Name"],
                    "smiles": row["SMILES"],
                    "source_label": int(row["Label"]),
                    "true_label": int(row["scope_label"]),
                    "true_label_name": row["scope_label_name"],
                })
    manifest = pd.DataFrame(rows)
    audit = {
        "task": task.task_id,
        "n_rows": int(len(scope)),
        "class_counts": scope["scope_label_name"].value_counts().to_dict(),
        "source_to_scope_label": {str(key): int(value) for key, value in task.source_to_scope.items()},
        "scope_label_mapping": {str(index + 1): name for index, name in enumerate(task.class_names)},
        "test_rows_per_fold": {
            str(key): int(value)
            for key, value in manifest[manifest["partition"] == "test"].groupby("fold").size().items()
        },
        "warning": "This reproduces the matched 80/10/10 row-level split logic; repeated structures are not grouped.",
    }
    return scope, manifest, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Create all original-label-range scope-matched tasks.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=HERE / "scope_matched" / "manifests")
    parser.add_argument("--tasks", nargs="+", choices=[task.key for task in TASKS], default=[task.key for task in TASKS])
    args = parser.parse_args()

    source = pd.read_csv(args.data).reset_index(names="source_row_index")
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [task for task in TASKS if task.key in args.tasks]
    for task in selected:
        scope, manifest, audit = build_task(source, task)
        prefix = task.filename_prefix
        manifest.to_csv(args.output / f"{prefix}_split_manifest.csv", index=False, encoding="utf-8-sig")
        manifest[manifest["partition"] == "test"].to_csv(
            args.output / f"{prefix}_test_manifest.csv", index=False, encoding="utf-8-sig"
        )
        server_csv = scope[["ID", "Name", "SMILES", "scope_label", "source_row_index", "sample_uid"]].rename(
            columns={"scope_label": "Label"}
        )
        server_csv.to_csv(args.output / f"{prefix}_compound.csv", index=False, encoding="utf-8-sig")
        audit["source_file"] = args.data.name
        audit["source_file_sha256"] = file_sha256(args.data)
        audit["split_manifest_sha256"] = file_sha256(args.output / f"{prefix}_split_manifest.csv")
        (args.output / f"{prefix}_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
