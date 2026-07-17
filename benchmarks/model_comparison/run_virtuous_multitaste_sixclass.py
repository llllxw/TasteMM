from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem, RDLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

from benchmark_utils import ordered_structure_hash, save_fold_outputs, validate_folds


HERE = Path(__file__).resolve().parent
MODEL_NAME = "Virtuous MultiTaste"
PAPER_15_FEATURES = [
    "ATSC0c", "ATSC0se", "AATS0i", "ATSC1p", "AATSC2se",
    "AATSC0m", "AATSC1Z", "AATSC2are", "AATSC1pe", "SpDiam_A",
    "ATSC1c", "ATSC1se", "ATSC1Z", "ATSC1m", "ATSC4s",
]


def descriptor_matrix(
    manifest: pd.DataFrame, cache_path: Path, index_column: str = "row_index", expected_rows: int = 12706
) -> pd.DataFrame:
    metadata_path = cache_path.with_suffix(".metadata.json")
    expected_hash = ordered_structure_hash(manifest, index_column, "smiles")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if cache_path.exists() and metadata.get("ordered_structure_sha256") == expected_hash:
        cached = pd.read_csv(cache_path)
        if cached.shape == (expected_rows, 15):
            return cached

    base = manifest.sort_values(index_column).drop_duplicates(index_column)
    if not np.array_equal(base[index_column].to_numpy(), np.arange(expected_rows)):
        raise ValueError(f"Manifest does not contain {index_column}=0..{expected_rows - 1} exactly once.")

    full_calc = Calculator(descriptors, ignore_3D=True)
    by_name = {str(item): item for item in full_calc.descriptors}
    missing = [name for name in PAPER_15_FEATURES if name not in by_name]
    if missing:
        raise RuntimeError(f"Mordred installation is missing published descriptors: {missing}")
    calc = Calculator([by_name[name] for name in PAPER_15_FEATURES], ignore_3D=True)
    mols = [Chem.MolFromSmiles(str(smiles)) for smiles in base["smiles"]]
    if any(mol is None for mol in mols):
        raise ValueError("Invalid SMILES encountered while computing Mordred descriptors.")
    values = calc.pandas(mols, nproc=1, quiet=False)
    values.columns = [str(column) for column in values.columns]
    values = values[PAPER_15_FEATURES].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    values.to_csv(cache_path, index=False)
    metadata_path.write_text(json.dumps({
        "ordered_structure_sha256": expected_hash,
        "shape": list(values.shape),
        "descriptors": PAPER_15_FEATURES,
    }, indent=2), encoding="utf-8")
    return values


def oversample(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = int(counts.max())
    all_indices = []
    for cls, count in zip(classes, counts):
        idx = np.flatnonzero(y == cls)
        if count < target:
            idx = np.concatenate([idx, rng.choice(idx, size=target - int(count), replace=True)])
        all_indices.append(idx)
    indices = np.concatenate(all_indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Virtuous MultiTaste task adaptation on frozen TasteMM splits.")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifests" / "six_class_split_manifest.csv")
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "virtuous_multitaste")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.folds = validate_folds(args.folds)
    RDLogger.DisableLog("rdApp.*")

    manifest = pd.read_csv(args.manifest)
    raw_features = descriptor_matrix(manifest, HERE / "cache" / "virtuous_paper15_12706.csv")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "unified_6class_12706",
        "model": MODEL_NAME,
        "method_source_doi": "10.1038/s41538-024-00287-6",
        "features": PAPER_15_FEATURES,
        "preprocessing": "KNN imputation k=20 and MinMax scaling fitted on training partition only",
        "imbalance": "random oversampling of training partition only",
        "classifier": "RandomForestClassifier(n_estimators=95), matching the published final classifier size",
        "task_adaptation": "Virtuous MultiTaste is retrained with six output classes for the common benchmark.",
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
        train_idx = train["row_index"].astype(int).to_numpy()
        test_idx = test["row_index"].astype(int).to_numpy()
        y_train = train["true_label"].astype(int).to_numpy() - 1

        imputer = KNNImputer(n_neighbors=20)
        X_train = imputer.fit_transform(raw_features.iloc[train_idx])
        X_test = imputer.transform(raw_features.iloc[test_idx])
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_resampled, y_resampled = oversample(X_train, y_train, seed=42 + fold)

        model = RandomForestClassifier(n_estimators=95, random_state=42 + fold, n_jobs=args.n_jobs)
        print(f"[RUN] fold={fold} train={len(train)} resampled={len(y_resampled)} test={len(test)}")
        model.fit(X_resampled, y_resampled)
        y_prob = model.predict_proba(X_test)
        if y_prob.shape[1] != 6:
            raise ValueError(f"Fold {fold} produced {y_prob.shape[1]} probability columns")
        save_fold_outputs(args.output, MODEL_NAME, fold, test, y_prob)
        joblib.dump({"imputer": imputer, "scaler": scaler, "model": model}, args.output / f"fold{fold}_model.joblib", compress=3)
        print(f"[DONE] fold={fold}")


if __name__ == "__main__":
    main()
