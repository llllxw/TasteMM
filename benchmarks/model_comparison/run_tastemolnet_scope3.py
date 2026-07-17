from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import RDLogger
from xgboost import XGBClassifier

from benchmark_utils import maccs_fingerprint, save_classification_fold_outputs, validate_folds


HERE = Path(__file__).resolve().parent
CLASS_NAMES = ["bitter", "sweet", "tasteless"]
MODEL_NAME = "TasteMolNet"


def compute_features(base: pd.DataFrame, cache: Path) -> np.ndarray:
    digest = hashlib.sha256()
    for row in base[["scope_row_index", "smiles"]].itertuples(index=False):
        digest.update(f"{int(row.scope_row_index)}\t{row.smiles}\n".encode("utf-8"))
    expected_hash = digest.hexdigest()
    metadata_path = cache.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if cache.exists() and metadata.get("ordered_structure_sha256") == expected_hash:
        return np.load(cache)
    values = []
    for smiles in base["smiles"]:
        values.append(maccs_fingerprint(smiles))
    features = np.vstack(values)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, features)
    metadata_path.write_text(json.dumps({
        "ordered_structure_sha256": expected_hash,
        "shape": list(features.shape),
        "feature": "RDKit MACCS keys",
    }, indent=2), encoding="utf-8")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TasteMolNet on the 10,650-row scope-matched task.")
    parser.add_argument("--manifest", type=Path, default=HERE / "scope_matched" / "manifests" / "tastemolnet_scope3_split_manifest.csv")
    parser.add_argument("--output", type=Path, default=HERE / "scope_matched" / "outputs" / "tastemolnet")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.folds = validate_folds(args.folds)
    RDLogger.DisableLog("rdApp.*")

    manifest = pd.read_csv(args.manifest)
    base = manifest.sort_values("scope_row_index").drop_duplicates("scope_row_index")
    if not np.array_equal(base["scope_row_index"].to_numpy(), np.arange(10650)):
        raise ValueError("Scope manifest is incomplete or out of order.")
    features = compute_features(base, HERE / "scope_matched" / "cache" / "maccs_scope3_10650.npy")
    args.output.mkdir(parents=True, exist_ok=True)

    config = {
        "task": "scope3_bitter_sweet_tasteless_10650",
        "model": MODEL_NAME,
        "method_source_doi": "10.1016/j.jfca.2026.108888",
        "hyperparameters": "Fixed paper-family parameters selected in the earlier seed-42 reproduction; no test tuning.",
        "implementation_boundary": "Independent task adaptation; the original web service was not called.",
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

        model = XGBClassifier(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
            colsample_bytree=1.0, min_child_weight=1, reg_alpha=0.0, reg_lambda=1.0,
            tree_method="hist", n_jobs=args.n_jobs, random_state=42 + fold,
        )
        print(f"[RUN] fold={fold} train={len(train)} test={len(test)}")
        model.fit(features[train_idx], y_train)
        y_prob = model.predict_proba(features[test_idx])
        save_classification_fold_outputs(
            args.output, MODEL_NAME, fold, test, y_prob, CLASS_NAMES,
            "scope3_bitter_sweet_tasteless_10650",
        )
        joblib.dump(model, args.output / f"fold{fold}_model.joblib", compress=3)
        print(f"[DONE] fold={fold}")


if __name__ == "__main__":
    main()
