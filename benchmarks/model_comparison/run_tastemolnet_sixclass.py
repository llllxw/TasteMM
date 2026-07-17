from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import RDLogger
from xgboost import XGBClassifier

from benchmark_utils import maccs_fingerprint, ordered_structure_hash, save_fold_outputs, validate_folds


HERE = Path(__file__).resolve().parent
MODEL_NAME = "TasteMolNet"


def feature_matrix(manifest: pd.DataFrame, cache_dir: Path) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "tastemolnet_maccs_features_12706.npy"
    metadata_path = cache_path.with_suffix(".metadata.json")
    expected_hash = ordered_structure_hash(manifest, "row_index", "smiles")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if cache_path.exists() and metadata.get("ordered_structure_sha256") == expected_hash:
        features = np.load(cache_path)
    else:
        base_rows = (
            manifest.sort_values("row_index")
            .drop_duplicates("row_index", keep="first")
            .reset_index(drop=True)
        )
        if not np.array_equal(base_rows["row_index"].to_numpy(), np.arange(12706)):
            raise ValueError("Manifest row indices are not the expected continuous range 0..12705.")
        features = np.vstack([maccs_fingerprint(s) for s in base_rows["smiles"]])
        np.save(cache_path, features)
        metadata_path.write_text(json.dumps({
            "ordered_structure_sha256": expected_hash,
            "shape": list(features.shape),
            "feature": "RDKit MACCS keys",
        }, indent=2), encoding="utf-8")
    if features.shape[0] != 12706:
        raise ValueError(f"Unexpected cached feature shape: {features.shape}")
    return features


def build_model(seed: int, n_jobs: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=6,
        eval_metric="mlogloss",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=n_jobs,
        random_state=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TasteMolNet task adaptation on frozen six-class TasteMM splits.")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifests" / "six_class_split_manifest.csv")
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "tastemolnet")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.folds = validate_folds(args.folds)

    RDLogger.DisableLog("rdApp.*")
    manifest = pd.read_csv(args.manifest)
    features = feature_matrix(manifest, HERE / "cache")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "unified_6class_12706",
        "model": MODEL_NAME,
        "method_source_doi": "10.1016/j.jfca.2026.108888",
        "feature": "MACCS keys",
        "classifier": "XGBoost",
        "split_manifest": str(args.manifest),
        "folds": args.folds,
        "label_encoding_for_training": "source labels 1-6 converted to 0-5",
        "task_adaptation": "TasteMolNet is retrained with six output classes for the common benchmark.",
        "note": "Fixed hyperparameters; no test-set tuning.",
        "implementation_boundary": "Independent task adaptation; the original web service was not called.",
    }
    with open(args.output / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    for fold in args.folds:
        pred_path = args.output / f"fold{fold}_predictions.csv"
        metric_path = args.output / f"fold{fold}_metrics.csv"
        if args.resume and pred_path.exists() and metric_path.exists():
            print(f"[SKIP] {MODEL_NAME} fold {fold}")
            continue

        train_rows = manifest[(manifest["fold"] == fold) & (manifest["partition"] == "train")]
        test_rows = manifest[(manifest["fold"] == fold) & (manifest["partition"] == "test")].reset_index(drop=True)
        train_indices = train_rows["row_index"].astype(int).to_numpy()
        test_indices = test_rows["row_index"].astype(int).to_numpy()
        y_train = train_rows["true_label"].astype(int).to_numpy() - 1

        model = build_model(seed=42 + fold, n_jobs=args.n_jobs)
        print(f"[RUN] {MODEL_NAME} fold={fold} train={len(train_indices)} test={len(test_indices)}")
        model.fit(features[train_indices], y_train)
        y_prob = model.predict_proba(features[test_indices])
        if y_prob.shape[1] != 6:
            raise ValueError(f"{MODEL_NAME} fold {fold} produced {y_prob.shape[1]} probability columns")

        save_fold_outputs(args.output, MODEL_NAME, fold, test_rows, y_prob)
        joblib.dump(model, args.output / f"fold{fold}_model.joblib", compress=3)
        print(f"[DONE] {MODEL_NAME} fold={fold}")


if __name__ == "__main__":
    main()
