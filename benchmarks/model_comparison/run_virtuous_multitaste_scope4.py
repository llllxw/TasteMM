from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from rdkit import RDLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

from benchmark_utils import save_classification_fold_outputs, validate_folds
from run_virtuous_multitaste_sixclass import PAPER_15_FEATURES, descriptor_matrix, oversample


HERE = Path(__file__).resolve().parent
MODEL_NAME = "Virtuous MultiTaste"
CLASS_NAMES = ["bitter", "sweet", "umami", "other"]
TASK = "scope4_bitter_sweet_umami_other_12706"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Virtuous MultiTaste on its four-class scope-matched task.")
    parser.add_argument("--manifest", type=Path, default=HERE / "scope_matched" / "manifests" / "virtuous_scope4_split_manifest.csv")
    parser.add_argument("--output", type=Path, default=HERE / "scope_matched" / "outputs" / "virtuous_multitaste_scope4")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.folds = validate_folds(args.folds)
    RDLogger.DisableLog("rdApp.*")

    manifest = pd.read_csv(args.manifest)
    features = descriptor_matrix(
        manifest, HERE / "scope_matched" / "cache" / "virtuous_scope4_paper15.csv",
        index_column="scope_row_index", expected_rows=12706,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "task": TASK, "model": MODEL_NAME, "classes": CLASS_NAMES, "features": PAPER_15_FEATURES,
        "method_source_doi": "10.1038/s41538-024-00287-6",
        "preprocessing": "KNN imputation k=20 and MinMax scaling fitted on training only",
        "imbalance": "random oversampling of training only",
        "classifier": "RandomForestClassifier(n_estimators=95)",
        "implementation_boundary": "Independent fixed-model adaptation; the InSyBio evolutionary search is not rerun.",
    }
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    for fold in args.folds:
        pred_path = args.output / f"fold{fold}_predictions.csv"
        metric_path = args.output / f"fold{fold}_metrics.csv"
        if args.resume and pred_path.exists() and metric_path.exists():
            print(f"[SKIP] fold {fold}")
            continue
        train = manifest[(manifest["fold"] == fold) & (manifest["partition"] == "train")]
        test = manifest[(manifest["fold"] == fold) & (manifest["partition"] == "test")].reset_index(drop=True)
        train_idx = train["scope_row_index"].astype(int).to_numpy()
        test_idx = test["scope_row_index"].astype(int).to_numpy()
        y_train = train["true_label"].astype(int).to_numpy() - 1
        imputer = KNNImputer(n_neighbors=20)
        x_train = imputer.fit_transform(features.iloc[train_idx])
        x_test = imputer.transform(features.iloc[test_idx])
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        x_train, y_train = oversample(x_train, y_train, seed=42 + fold)
        model = RandomForestClassifier(n_estimators=95, random_state=42 + fold, n_jobs=args.n_jobs)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        save_classification_fold_outputs(args.output, MODEL_NAME, fold, test, probabilities, CLASS_NAMES, TASK)
        joblib.dump({"imputer": imputer, "scaler": scaler, "model": model}, args.output / f"fold{fold}_model.joblib", compress=3)
        print(f"[DONE] fold={fold}")


if __name__ == "__main__":
    main()
