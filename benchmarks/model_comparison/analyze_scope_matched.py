from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from benchmark_utils import classification_metrics


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AnalysisTask:
    key: str
    task_id: str
    manifest_name: str
    class_names: tuple[str, ...]
    comparator: str
    output_dirs: tuple[str, str]


TASKS = {
    task.key: task for task in (
        AnalysisTask("scope3", "scope3_bitter_sweet_tasteless_10650", "tastemolnet_scope3_split_manifest.csv",
                     ("bitter", "sweet", "tasteless"), "TasteMolNet", ("tastemm_scope3", "tastemolnet")),
        AnalysisTask("scope4", "scope4_bitter_sweet_umami_other_12706", "virtuous_scope4_split_manifest.csv",
                     ("bitter", "sweet", "umami", "other"), "Virtuous MultiTaste",
                     ("tastemm_scope4", "virtuous_multitaste_scope4")),
        AnalysisTask("scope5", "scope5_bitter_sweet_sour_umami_undefined_12706", "fart_scope5_split_manifest.csv",
                     ("bitter", "sweet", "sour", "umami", "undefined"), "FART",
                     ("tastemm_scope5", "fart_scope5")),
    )
}


def main(default_task: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate and analyze one two-model scope-matched task.")
    parser.add_argument("--task", choices=sorted(TASKS), default=default_task or "scope3")
    parser.add_argument("--outputs", type=Path, default=HERE / "scope_matched" / "outputs")
    parser.add_argument("--analysis", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    task = TASKS[args.task]
    analysis = args.analysis or HERE / "scope_matched" / "analysis" / task.key
    manifest_path = args.manifest or HERE / "scope_matched" / "manifests" / task.manifest_name
    analysis.mkdir(parents=True, exist_ok=True)

    metric_files, prediction_files = [], []
    for directory in task.output_dirs:
        metric_files.extend(sorted((args.outputs / directory).glob("fold*_metrics.csv")))
        prediction_files.extend(sorted((args.outputs / directory).glob("fold*_predictions.csv")))
    if len(metric_files) != 10 or len(prediction_files) != 10:
        raise FileNotFoundError(
            f"{task.key} requires 5 metric and prediction files from each of {task.output_dirs}; "
            f"found {len(metric_files)} and {len(prediction_files)}."
        )
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    predictions = pd.concat([pd.read_csv(path) for path in prediction_files], ignore_index=True)
    expected_models = ["TasteMM", task.comparator]
    expected_splits = [f"{task.key}_fold{fold}_seed42" for fold in range(5)]
    if set(metrics["task"].astype(str)) != {task.task_id}:
        raise ValueError(f"Unexpected task IDs in {task.key} metrics.")
    if sorted(metrics["model"].astype(str).unique()) != sorted(expected_models):
        raise ValueError(f"{task.key} requires exactly {expected_models}.")
    if metrics.duplicated(["model", "split_id", "metric"]).any():
        raise ValueError("Duplicate scope metric rows found.")
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["partition"] == "test"].copy()
    probability_columns = [f"prob_{name}" for name in task.class_names]
    if predictions.duplicated(["model", "split_id", "sample_uid"]).any():
        raise ValueError("Duplicate scope prediction rows found.")
    all_probabilities = predictions[probability_columns].to_numpy(float)
    if not np.isfinite(all_probabilities).all() or np.any(all_probabilities < 0) or np.any(all_probabilities > 1):
        raise ValueError("Scope probabilities must be finite and in [0,1].")
    if not np.allclose(all_probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Scope probability rows do not sum to one.")

    for model in expected_models:
        model_metrics = metrics[metrics["model"] == model]
        if sorted(model_metrics["split_id"].unique()) != expected_splits:
            raise ValueError(f"{model} does not cover the same five {task.key} splits.")
        for split_id in expected_splits:
            expected = manifest[manifest["split_id"] == split_id].reset_index(drop=True)
            split_metrics = model_metrics[model_metrics["split_id"] == split_id]
            expected_fold = int(expected["fold"].iloc[0])
            if set(split_metrics["fold"].astype(int)) != {expected_fold}:
                raise ValueError(f"{model}/{split_id} metric fold does not match the manifest.")
            if set(split_metrics["n_test"].astype(int)) != {len(expected)}:
                raise ValueError(f"{model}/{split_id} metric n_test does not match the manifest.")
            observed = predictions[(predictions["model"] == model) & (predictions["split_id"] == split_id)].reset_index(drop=True)
            if len(observed) != len(expected):
                raise ValueError(f"{model}/{split_id} row count mismatch.")
            for column in ("scope_row_index", "sample_uid", "true_label", "fold"):
                if not np.array_equal(observed[column].to_numpy(), expected[column].to_numpy()):
                    raise ValueError(f"{model}/{split_id} {column} does not align with the frozen manifest.")
            probabilities = observed[probability_columns].to_numpy(float)
            predicted = probabilities.argmax(axis=1) + 1
            if not np.array_equal(predicted, observed["pred_label"].astype(int).to_numpy()):
                raise ValueError(f"{model}/{split_id} pred_label disagrees with probabilities.")
            recomputed = classification_metrics(
                expected["true_label"].astype(int).to_numpy(), predicted, probabilities, len(task.class_names)
            )
            reported = split_metrics.set_index("metric")["value"]
            if set(reported.index) != set(recomputed):
                raise ValueError(f"{model}/{split_id} metric set is incomplete or unexpected.")
            for metric_name, value in recomputed.items():
                if not np.isclose(float(reported[metric_name]), value, atol=1e-10, rtol=1e-8):
                    raise ValueError(f"{model}/{split_id} reported {metric_name} does not match predictions.")

    metrics.to_csv(analysis / "all_fold_metrics_long.csv", index=False)
    summary = metrics.groupby(["task", "model", "metric"], as_index=False).agg(
        mean=("value", "mean"), sd=("value", "std"), n_folds=("value", "size")
    )
    summary.to_csv(analysis / "metrics_mean_sd.csv", index=False)
    paired_rows = []
    for metric_name in ("macro_auroc", "macro_auprc"):
        pivot = metrics[metrics["metric"] == metric_name].pivot(index="split_id", columns="model", values="value")
        if pivot.shape != (5, 2) or pivot.isna().any().any():
            raise ValueError(f"Incomplete paired values for {metric_name}.")
        differences = pivot["TasteMM"] - pivot[task.comparator]
        sd = float(differences.std(ddof=1))
        se = sd / np.sqrt(len(differences))
        critical = float(stats.t.ppf(0.975, len(differences) - 1))
        paired_rows.append({
            "metric": metric_name, "model_a": "TasteMM", "model_b": task.comparator,
            "n_paired_runs": len(differences), "mean_difference_a_minus_b": float(differences.mean()),
            "paired_sd": sd, "ci95_low": float(differences.mean() - critical * se),
            "ci95_high": float(differences.mean() + critical * se),
            "paired_t_p_unadjusted": float(stats.ttest_rel(pivot["TasteMM"], pivot[task.comparator]).pvalue),
        })
    paired = pd.DataFrame(paired_rows)
    paired["paired_t_p_holm"] = multipletests(paired["paired_t_p_unadjusted"], method="holm")[1]
    paired["reject_holm_0_05"] = paired["paired_t_p_holm"] < 0.05
    paired.to_csv(analysis / "paired_effects.csv", index=False)
    audit = {
        "task": task.task_id, "models": expected_models, "validation": "PASS",
        "statistical_unit": "matched run score", "multiplicity": "Holm correction across AUROC and AUPRC",
        "prohibition": "Do not combine scores from different output spaces.",
    }
    (analysis / "analysis_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
