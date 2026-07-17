from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

from benchmark_utils import LABEL_NAMES, PROB_COLUMNS, overall_metrics, per_taste_metrics


HERE = Path(__file__).resolve().parent
PRIMARY_METRICS = ["macro_auroc", "macro_auprc"]
EXPECTED_MODELS = ["TasteMM", "TasteMolNet", "Virtuous MultiTaste", "FART"]
EXPECTED_SPLITS = [f"fold{fold}_seed42" for fold in range(5)]
EXPECTED_TASK = "unified_6class_12706"


def read_outputs(outputs: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames = [pd.read_csv(path) for path in sorted(outputs.glob("*/fold*_metrics.csv"))]
    prediction_frames = [pd.read_csv(path) for path in sorted(outputs.glob("*/fold*_predictions.csv"))]
    if not metric_frames:
        raise FileNotFoundError(f"No fold metric files found below {outputs}")
    if not prediction_frames:
        raise FileNotFoundError(f"No fold prediction files found below {outputs}")
    return pd.concat(metric_frames, ignore_index=True), pd.concat(prediction_frames, ignore_index=True)


def validate_paired_inputs(metrics: pd.DataFrame, predictions: pd.DataFrame, manifest_path: Path) -> dict:
    metric_required = {"task", "model", "split_id", "fold", "metric", "value", "n_test"}
    prediction_required = {
        "row_index", "sample_uid", "split_id", "fold", "true_label", "model", "pred_label", *PROB_COLUMNS,
    }
    for name, frame, required in (
        ("metrics", metrics, metric_required), ("predictions", predictions, prediction_required)
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} files are missing columns: {sorted(missing)}")
    if set(metrics["task"].astype(str)) != {EXPECTED_TASK}:
        raise ValueError(f"Metrics contain an unexpected task: {sorted(metrics['task'].astype(str).unique())}")
    models = sorted(metrics["model"].astype(str).unique())
    if models != sorted(EXPECTED_MODELS):
        raise ValueError(f"Expected exactly {EXPECTED_MODELS}; found {models}. Analysis is stopped.")
    if sorted(predictions["model"].astype(str).unique()) != sorted(EXPECTED_MODELS):
        raise ValueError("Metric and prediction model sets are not the same complete four-model set.")
    if metrics.duplicated(["model", "split_id", "metric"]).any():
        raise ValueError("Duplicate model/split/metric rows were found.")
    if predictions.duplicated(["model", "split_id", "sample_uid"]).any():
        raise ValueError("Duplicate model/split/sample_uid predictions were found.")
    if not np.isfinite(metrics["value"].astype(float)).all():
        raise ValueError("Metric values contain NaN or infinity.")

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["partition"].astype(str) == "test"].copy()
    if manifest.duplicated(["split_id", "sample_uid"]).any():
        raise ValueError("Frozen test manifest contains duplicate split/sample_uid rows.")
    manifest_splits = sorted(manifest["split_id"].astype(str).unique())
    if manifest_splits != EXPECTED_SPLITS:
        raise ValueError(f"Frozen manifest split IDs are {manifest_splits}, expected {EXPECTED_SPLITS}.")

    probability_array = predictions[PROB_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probability_array).all() or np.any(probability_array < 0) or np.any(probability_array > 1):
        raise ValueError("Prediction probabilities must be finite and in [0, 1].")
    if not np.allclose(probability_array.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Prediction probability rows do not sum to one.")
    expected_metric_names = set(overall_metrics(
        manifest[manifest["split_id"] == EXPECTED_SPLITS[0]]["true_label"].astype(int).to_numpy(),
        manifest[manifest["split_id"] == EXPECTED_SPLITS[0]]["true_label"].astype(int).to_numpy(),
        np.eye(6)[manifest[manifest["split_id"] == EXPECTED_SPLITS[0]]["true_label"].astype(int).to_numpy() - 1],
    ))

    for model in EXPECTED_MODELS:
        model_metrics = metrics[metrics["model"].astype(str) == model]
        if sorted(model_metrics["split_id"].astype(str).unique()) != EXPECTED_SPLITS:
            raise ValueError(f"{model} does not have exactly the five frozen split IDs.")
        if set(model_metrics["metric"].astype(str)) != expected_metric_names:
            raise ValueError(f"{model} metric set is incomplete or unexpected.")
        for split_id in EXPECTED_SPLITS:
            expected = manifest[manifest["split_id"].astype(str) == split_id].reset_index(drop=True)
            split_metrics = model_metrics[model_metrics["split_id"].astype(str) == split_id]
            expected_fold = int(expected["fold"].iloc[0])
            if set(split_metrics["fold"].astype(int)) != {expected_fold}:
                raise ValueError(f"{model}/{split_id}: metric fold does not match the manifest.")
            if set(split_metrics["n_test"].astype(int)) != {len(expected)}:
                raise ValueError(f"{model}/{split_id}: metric n_test does not match the manifest.")
            observed = predictions[
                (predictions["model"].astype(str) == model)
                & (predictions["split_id"].astype(str) == split_id)
            ].reset_index(drop=True)
            if len(observed) != len(expected):
                raise ValueError(f"{model}/{split_id}: {len(observed)} predictions != {len(expected)} manifest rows.")
            for column in ("sample_uid", "row_index", "true_label", "fold"):
                if not np.array_equal(observed[column].to_numpy(), expected[column].to_numpy()):
                    raise ValueError(f"{model}/{split_id}: {column} order does not match the frozen manifest.")
            probs = observed[PROB_COLUMNS].to_numpy(dtype=float)
            predicted = probs.argmax(axis=1) + 1
            if not np.array_equal(predicted, observed["pred_label"].astype(int).to_numpy()):
                raise ValueError(f"{model}/{split_id}: pred_label is inconsistent with argmax probabilities.")
            recomputed = overall_metrics(expected["true_label"].astype(int).to_numpy(), predicted, probs)
            reported = split_metrics.set_index("metric")["value"]
            for metric_name, value in recomputed.items():
                if not np.isclose(float(reported[metric_name]), value, atol=1e-10, rtol=1e-8):
                    raise ValueError(f"{model}/{split_id}: reported {metric_name} does not match predictions.")
    return {
        "validation": "PASS",
        "expected_models": EXPECTED_MODELS,
        "expected_splits": EXPECTED_SPLITS,
        "test_rows_per_split": manifest.groupby("split_id").size().astype(int).to_dict(),
    }


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["task", "model", "metric"], as_index=False)
        .agg(mean=("value", "mean"), sd=("value", "std"), n_folds=("value", "size"))
        .sort_values(["metric", "mean"], ascending=[True, False])
    )


def blocked_anova_and_tukey(metrics: pd.DataFrame, output: Path) -> None:
    anova_rows: list[dict] = []
    tukey_rows: list[dict] = []
    paired_rows: list[dict] = []

    for metric_name in PRIMARY_METRICS:
        data = metrics[metrics["metric"] == metric_name].copy()
        counts = data.groupby("model")["split_id"].nunique()
        eligible = counts[counts == 5].index
        data = data[data["model"].isin(eligible)].copy()
        if data["model"].nunique() < 2:
            continue

        fitted = ols("value ~ C(model) + C(split_id)", data=data).fit()
        raw_table = anova_lm(fitted, typ=2)
        table = raw_table.reset_index().rename(columns={"index": "term"})
        for _, row in table.iterrows():
            anova_rows.append({"metric": metric_name, **row.to_dict()})

        # Tukey HSD for a randomized complete block design: folds are blocks,
        # models are treatments, and the ANOVA residual supplies the error term.
        model_means = data.groupby("model")["value"].mean()
        n_blocks = data["split_id"].nunique()
        n_models = data["model"].nunique()
        mse = float(raw_table.loc["Residual", "sum_sq"] / raw_table.loc["Residual", "df"])
        df_error = float(raw_table.loc["Residual", "df"])
        se_hsd = float(np.sqrt(mse / n_blocks))
        q_critical = float(stats.studentized_range.ppf(0.95, n_models, df_error))
        for model_a, model_b in combinations(sorted(model_means.index), 2):
            difference = float(model_means[model_a] - model_means[model_b])
            q_stat = abs(difference) / se_hsd
            p_adjusted = float(stats.studentized_range.sf(q_stat, n_models, df_error))
            tukey_rows.append(
                {
                    "metric": metric_name,
                    "model_a": model_a,
                    "model_b": model_b,
                    "mean_difference_a_minus_b": difference,
                    "q_statistic": q_stat,
                    "p_tukey_adjusted": p_adjusted,
                    "simultaneous_ci95_low": difference - q_critical * se_hsd,
                    "simultaneous_ci95_high": difference + q_critical * se_hsd,
                    "reject_0_05": p_adjusted < 0.05,
                    "n_blocks": n_blocks,
                    "anova_error_df": df_error,
                }
            )

        pivot = data.pivot(index="split_id", columns="model", values="value").dropna()
        for model_a, model_b in combinations(pivot.columns, 2):
            differences = pivot[model_a] - pivot[model_b]
            n = len(differences)
            mean_diff = float(differences.mean())
            sd_diff = float(differences.std(ddof=1)) if n > 1 else float("nan")
            se = sd_diff / np.sqrt(n) if n > 1 else float("nan")
            critical = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
            paired_rows.append(
                {
                    "metric": metric_name,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_paired_folds": n,
                    "mean_difference_a_minus_b": mean_diff,
                    "paired_sd": sd_diff,
                    "ci95_low": mean_diff - critical * se,
                    "ci95_high": mean_diff + critical * se,
                    "paired_t_p_unadjusted": float(stats.ttest_rel(pivot[model_a], pivot[model_b]).pvalue),
                    "standardized_paired_effect_dz": mean_diff / sd_diff if sd_diff > 0 else float("nan"),
                }
            )

    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired["paired_t_p_holm"] = np.nan
        for _, idx in paired.groupby("metric").groups.items():
            idx = list(idx)
            paired.loc[idx, "paired_t_p_holm"] = multipletests(
                paired.loc[idx, "paired_t_p_unadjusted"].to_numpy(), method="holm"
            )[1]
        paired["reject_holm_0_05"] = paired["paired_t_p_holm"] < 0.05
    pd.DataFrame(anova_rows).to_csv(output / "primary_blocked_anova.csv", index=False)
    pd.DataFrame(tukey_rows).to_csv(output / "primary_tukey_hsd.csv", index=False)
    paired.to_csv(output / "primary_paired_effects.csv", index=False)


def build_per_taste(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, split_id, fold), group in predictions.groupby(["model", "split_id", "fold"]):
        y_true = group["true_label"].astype(int).to_numpy()
        y_pred = group["pred_label"].astype(int).to_numpy()
        y_prob = group[PROB_COLUMNS].to_numpy(dtype=float)
        for result in per_taste_metrics(y_true, y_pred, y_prob):
            for metric in ["ovr_auroc", "ovr_auprc", "precision", "recall", "f1"]:
                rows.append(
                    {
                        "task": "unified_6class_12706",
                        "model": model,
                        "split_id": split_id,
                        "fold": int(fold),
                        "taste": result["taste"],
                        "support": result["support"],
                        "metric": metric,
                        "value": result[metric],
                    }
                )
    return pd.DataFrame(rows)


def per_taste_tukey(per_taste: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for taste in LABEL_NAMES:
        for metric_name in ["ovr_auroc", "ovr_auprc"]:
            data = per_taste[(per_taste["taste"] == taste) & (per_taste["metric"] == metric_name)]
            counts = data.groupby("model")["split_id"].nunique()
            eligible = counts[counts == 5].index
            data = data[data["model"].isin(eligible)]
            if data["model"].nunique() < 2:
                continue
            fitted = ols("value ~ C(model) + C(split_id)", data=data).fit()
            raw_table = anova_lm(fitted, typ=2)
            means = data.groupby("model")["value"].mean()
            n_blocks = data["split_id"].nunique()
            n_models = data["model"].nunique()
            mse = float(raw_table.loc["Residual", "sum_sq"] / raw_table.loc["Residual", "df"])
            df_error = float(raw_table.loc["Residual", "df"])
            se_hsd = float(np.sqrt(mse / n_blocks))
            q_critical = float(stats.studentized_range.ppf(0.95, n_models, df_error))
            for model_a, model_b in combinations(sorted(means.index), 2):
                difference = float(means[model_a] - means[model_b])
                q_stat = abs(difference) / se_hsd
                rows.append(
                    {
                        "taste": taste,
                        "metric": metric_name,
                        "model_a": model_a,
                        "model_b": model_b,
                        "mean_difference_a_minus_b": difference,
                        "q_statistic": q_stat,
                        "p_tukey_within_taste": float(stats.studentized_range.sf(q_stat, n_models, df_error)),
                        "simultaneous_ci95_low": difference - q_critical * se_hsd,
                        "simultaneous_ci95_high": difference + q_critical * se_hsd,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # A second layer of multiplicity control across the six tastes, separately
    # for each metric and model pair.
    result["p_holm_across_tastes"] = np.nan
    for _, idx in result.groupby(["metric", "model_a", "model_b"]).groups.items():
        idx = list(idx)
        result.loc[idx, "p_holm_across_tastes"] = multipletests(
            result.loc[idx, "p_tukey_within_taste"].to_numpy(), method="holm"
        )[1]
    result["reject_holm_0_05"] = result["p_holm_across_tastes"] < 0.05
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate unified benchmark results and run reviewer-requested statistics.")
    parser.add_argument("--outputs", type=Path, default=HERE / "outputs")
    parser.add_argument("--analysis", type=Path, default=HERE / "analysis")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifests" / "six_class_split_manifest.csv")
    args = parser.parse_args()
    args.analysis.mkdir(parents=True, exist_ok=True)

    metrics, predictions = read_outputs(args.outputs)
    validation_audit = validate_paired_inputs(metrics, predictions, args.manifest)
    metrics.to_csv(args.analysis / "all_fold_metrics_long.csv", index=False)
    summary = summarize_metrics(metrics)
    summary.to_csv(args.analysis / "overall_metrics_mean_sd.csv", index=False)

    blocked_anova_and_tukey(metrics, args.analysis)

    per_taste = build_per_taste(predictions)
    per_taste.to_csv(args.analysis / "per_taste_fold_metrics_long.csv", index=False)
    per_taste_summary = (
        per_taste.groupby(["model", "taste", "metric"], as_index=False)
        .agg(mean=("value", "mean"), sd=("value", "std"), n_folds=("value", "size"), mean_support=("support", "mean"))
    )
    per_taste_summary.to_csv(args.analysis / "per_taste_metrics_mean_sd.csv", index=False)
    per_taste_tukey(per_taste).to_csv(args.analysis / "per_taste_tukey_hsd.csv", index=False)

    audit = {
        **validation_audit,
        "models_found": sorted(metrics["model"].unique().tolist()),
        "folds_per_model": {
            model: int(count)
            for model, count in metrics[metrics["metric"] == "macro_auroc"].groupby("model")["fold"].nunique().items()
        },
        "primary_metrics": PRIMARY_METRICS,
        "statistical_unit": "fold score; identical split_id is the pairing/blocking unit",
        "tukey_note": "Tukey HSD uses the fold-blocked ANOVA residual (randomized complete block design); paired effect estimates are also reported.",
        "salty_note": "Salty per-taste inference is exploratory because only 28 samples exist in the full dataset.",
    }
    with open(args.analysis / "analysis_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
