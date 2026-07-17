from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_utils import save_classification_fold_outputs


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ImportTask:
    key: str
    manifest_name: str
    output_name: str
    class_names: tuple[str, ...]
    task_id: str


TASKS = {
    task.key: task for task in (
        ImportTask("scope3", "tastemolnet_scope3_split_manifest.csv", "tastemm_scope3",
                   ("bitter", "sweet", "tasteless"), "scope3_bitter_sweet_tasteless_10650"),
        ImportTask("scope4", "virtuous_scope4_split_manifest.csv", "tastemm_scope4",
                   ("bitter", "sweet", "umami", "other"), "scope4_bitter_sweet_umami_other_12706"),
        ImportTask("scope5", "fart_scope5_split_manifest.csv", "tastemm_scope5",
                   ("bitter", "sweet", "sour", "umami", "undefined"),
                   "scope5_bitter_sweet_sour_umami_undefined_12706"),
    )
}


def main(default_task: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Import TasteMM outputs into a scope-matched comparison.")
    parser.add_argument("--task", choices=sorted(TASKS), default=default_task or "scope3")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    task = TASKS[args.task]
    manifest_path = args.manifest or HERE / "scope_matched" / "manifests" / task.manifest_name
    output = args.output or HERE / "scope_matched" / "outputs" / task.output_name
    manifest = pd.read_csv(manifest_path)

    for fold in range(5):
        completed = [path for path in args.run_root.glob(f"fold{fold}_seed42_*") if (path / "result.json").exists()]
        if len(completed) != 1:
            raise RuntimeError(f"Expected one completed TasteMM {task.key} run for fold {fold}, found {len(completed)}")
        artifacts = completed[0] / "artifacts"
        logits = np.load(artifacts / "test_logits.npy")
        labels = np.load(artifacts / "test_labels.npy").astype(int)
        row_indices = np.load(artifacts / "test_row_indices.npy").astype(int)
        if logits.shape[1] != len(task.class_names):
            raise ValueError(f"Fold {fold} logits shape {logits.shape} is not a {len(task.class_names)}-class run.")
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        test = manifest[(manifest["fold"] == fold) & (manifest["partition"] == "test")].reset_index(drop=True)
        expected_labels = test["true_label"].astype(int).to_numpy() - 1
        if not np.array_equal(labels, expected_labels):
            raise ValueError(f"Fold {fold}: label order does not match the frozen {task.key} manifest.")
        if not np.array_equal(row_indices, test["scope_row_index"].astype(int).to_numpy()):
            raise ValueError(f"Fold {fold}: row_index order does not match the frozen {task.key} manifest.")
        save_classification_fold_outputs(
            output, "TasteMM", fold, test, probabilities, list(task.class_names), task.task_id
        )
        print(f"Imported TasteMM {task.key} fold {fold}")


if __name__ == "__main__":
    main()
