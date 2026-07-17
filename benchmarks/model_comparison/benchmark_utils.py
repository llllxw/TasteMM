from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


LABEL_NAMES = ["bitter", "sweet", "umami", "salty", "sour", "tasteless"]
LABEL_IDS = np.arange(1, 7)
PROB_COLUMNS = [f"prob_{name}" for name in LABEL_NAMES]


def ordered_structure_hash(frame: pd.DataFrame, index_column: str, smiles_column: str) -> str:
    ordered = frame.sort_values(index_column).drop_duplicates(index_column, keep="first")
    digest = hashlib.sha256()
    for row in ordered[[index_column, smiles_column]].itertuples(index=False):
        digest.update(f"{int(row[0])}\t{row[1]}\n".encode("utf-8"))
    return digest.hexdigest()


def maccs_fingerprint(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    fp = MACCSkeys.GenMACCSKeys(mol)
    array = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


def validate_folds(folds: list[int]) -> list[int]:
    values = [int(value) for value in folds]
    if not values or len(set(values)) != len(values) or any(value not in range(5) for value in values):
        raise ValueError("folds must contain unique values selected from 0,1,2,3,4")
    return values


def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return classification_metrics(y_true, y_pred, y_prob, num_classes=6)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    label_ids = np.arange(1, num_classes + 1)
    y_bin = label_binarize(y_true, classes=label_ids)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_auroc": float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")),
        "macro_auprc": float(average_precision_score(y_bin, y_prob, average="macro")),
    }


def per_taste_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for class_id, class_name in zip(LABEL_IDS, LABEL_NAMES):
        true_binary = (np.asarray(y_true) == class_id).astype(int)
        pred_binary = (np.asarray(y_pred) == class_id).astype(int)
        rows.append(
            {
                "taste": class_name,
                "support": int(true_binary.sum()),
                "ovr_auroc": float(roc_auc_score(true_binary, y_prob[:, class_id - 1])),
                "ovr_auprc": float(average_precision_score(true_binary, y_prob[:, class_id - 1])),
                "precision": float(precision_score(true_binary, pred_binary, zero_division=0)),
                "recall": float(recall_score(true_binary, pred_binary, zero_division=0)),
                "f1": float(f1_score(true_binary, pred_binary, zero_division=0)),
            }
        )
    return rows


def save_fold_outputs(
    output_dir: Path,
    model_name: str,
    fold: int,
    manifest_test: pd.DataFrame,
    y_prob: np.ndarray,
) -> None:
    save_classification_fold_outputs(
        output_dir, model_name, fold, manifest_test, y_prob, LABEL_NAMES, "unified_6class_12706"
    )


def save_classification_fold_outputs(
    output_dir: Path,
    model_name: str,
    fold: int,
    manifest_test: pd.DataFrame,
    y_prob: np.ndarray,
    label_names: list[str],
    task: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "sample_uid", "split_id", "fold", "partition", "source_id",
        "name", "smiles", "true_label", "true_label_name",
    }
    missing = required.difference(manifest_test.columns)
    if missing:
        raise ValueError(f"Test manifest is missing columns: {sorted(missing)}")
    if not ({"row_index", "scope_row_index"} & set(manifest_test.columns)):
        raise ValueError("Test manifest must contain row_index or scope_row_index.")
    if manifest_test.empty or set(manifest_test["partition"].astype(str)) != {"test"}:
        raise ValueError("save_fold_outputs requires a non-empty test partition only.")
    if set(manifest_test["fold"].astype(int)) != {int(fold)}:
        raise ValueError(f"Manifest rows do not belong exclusively to fold {fold}.")
    split_ids = manifest_test["split_id"].astype(str).unique().tolist()
    if len(split_ids) != 1:
        raise ValueError(f"Fold {fold} must have exactly one split_id, found {split_ids}.")
    if manifest_test["sample_uid"].astype(str).duplicated().any():
        raise ValueError(f"Fold {fold} contains duplicate sample_uid values.")
    y_prob = np.asarray(y_prob, dtype=float)
    label_ids = np.arange(1, len(label_names) + 1)
    if y_prob.shape != (len(manifest_test), len(label_names)):
        raise ValueError(f"Probability shape {y_prob.shape} != {(len(manifest_test), len(label_names))}.")
    if not np.isfinite(y_prob).all() or np.any(y_prob < 0) or np.any(y_prob > 1):
        raise ValueError("Probabilities must be finite values in [0, 1].")
    if not np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to one.")
    y_true = manifest_test["true_label"].astype(int).to_numpy()
    if set(y_true) != set(label_ids):
        raise ValueError(f"Fold {fold} test set must contain every class 1..{len(label_names)}.")
    y_pred = np.asarray(y_prob).argmax(axis=1) + 1

    pred = manifest_test[
        [column for column in (
            "row_index", "scope_row_index", "source_row_index", "sample_uid", "split_id", "fold",
            "source_id", "name", "smiles", "true_label", "true_label_name"
        ) if column in manifest_test.columns]
    ].copy()
    pred["model"] = model_name
    pred["pred_label"] = y_pred
    pred["pred_label_name"] = [label_names[value - 1] for value in y_pred]
    for idx, label_name in enumerate(label_names):
        pred[f"prob_{label_name}"] = y_prob[:, idx]
    pred.to_csv(output_dir / f"fold{fold}_predictions.csv", index=False, encoding="utf-8-sig")

    metrics = classification_metrics(y_true, y_pred, y_prob, len(label_names))
    metric_rows = [
        {
            "task": task,
            "model": model_name,
            "split_id": split_ids[0],
            "fold": fold,
            "metric": metric,
            "value": value,
            "n_test": len(y_true),
        }
        for metric, value in metrics.items()
    ]
    pd.DataFrame(metric_rows).to_csv(output_dir / f"fold{fold}_metrics.csv", index=False)
