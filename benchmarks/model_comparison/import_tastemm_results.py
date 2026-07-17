from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_utils import save_fold_outputs


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Import existing TasteMM predictions into the common benchmark schema.")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifests" / "six_class_split_manifest.csv")
    parser.add_argument("--run-root", type=Path, required=True, help="TasteMM five-fold run directory.")
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "tastemm")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    for fold in range(5):
        candidates = sorted(args.run_root.glob(f"fold{fold}_seed42_*"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one TasteMM run for fold {fold}, found {len(candidates)}")
        artifact_dir = candidates[0] / "artifacts"
        logits = np.load(artifact_dir / "test_logits.npy")
        logits = logits - logits.max(axis=1, keepdims=True)
        y_prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        saved_labels = np.load(artifact_dir / "test_labels.npy").astype(int)
        row_index_path = artifact_dir / "test_row_indices.npy"
        if not row_index_path.exists():
            raise FileNotFoundError(
                f"{row_index_path} is required for auditable per-molecule alignment. "
                "Run reevaluate_run.py for checkpoints produced by an older version."
            )
        saved_row_indices = np.load(row_index_path).astype(int)

        test_manifest = manifest[(manifest["fold"] == fold) & (manifest["partition"] == "test")].copy()
        test_manifest = test_manifest.reset_index(drop=True)
        expected_labels = test_manifest["true_label"].astype(int).to_numpy() - 1
        if y_prob.shape != (len(test_manifest), 6):
            raise ValueError(f"Fold {fold}: probability shape {y_prob.shape} does not match {(len(test_manifest), 6)}")
        if not np.array_equal(saved_labels, expected_labels):
            raise ValueError(
                f"Fold {fold}: saved TasteMM label order does not match the reproduced split manifest. "
                "Stop rather than silently misalign predictions."
            )
        expected_row_indices = test_manifest["row_index"].astype(int).to_numpy()
        if not np.array_equal(saved_row_indices, expected_row_indices):
            raise ValueError(f"Fold {fold}: TasteMM row_index order does not match the frozen manifest.")
        save_fold_outputs(args.output, "TasteMM", fold, test_manifest, y_prob)
        print(f"Imported TasteMM fold {fold}: n={len(test_manifest)}")


if __name__ == "__main__":
    main()
