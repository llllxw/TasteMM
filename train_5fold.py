from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys

from tools import PROCESSED_DIR, training_run_name


def get_args():
    parser = argparse.ArgumentParser(description="Run five matched 80/10/10 experiments for TasteMM.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--train_script", type=str, default="train.py")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--graph_aux_weight", type=float, default=0.2)
    parser.add_argument("--graph_warmup_epochs", type=int, default=30)
    parser.add_argument("--graph_warmup_lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="runs")
    parser.add_argument("--processed_dir", type=str, default=PROCESSED_DIR)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--summary_csv", type=str, default="results/baseline_5fold_summary.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow_partial_summary", action="store_true")
    return parser.parse_args()


def parse_folds(folds_str: str):
    folds = [int(part.strip()) for part in folds_str.split(",") if part.strip()]
    if not folds or len(set(folds)) != len(folds) or any(fold not in range(5) for fold in folds):
        raise ValueError("--folds must contain unique values selected from 0,1,2,3,4.")
    return folds


def run_config_for_name(args) -> dict:
    return {
        "skip_pretrain": bool(args.skip_pretrain),
        "graph_aux_weight": float(args.graph_aux_weight),
        "graph_warmup_epochs": int(args.graph_warmup_epochs),
        "graph_warmup_lr": float(args.graph_warmup_lr),
        "num_classes": int(args.num_classes),
    }


def run_one_fold(fold: int, args) -> str:
    run_name = training_run_name(fold, args.seed, run_config_for_name(args))
    run_dir = os.path.join(args.out_dir, run_name)
    result_json = os.path.join(run_dir, "result.json")
    if args.resume and os.path.exists(result_json):
        result = load_json(result_json)
        expected = {
            "fold": fold,
            "seed": args.seed,
            "skip_pretrain": bool(args.skip_pretrain),
            "graph_aux_weight": float(args.graph_aux_weight),
            "graph_warmup_epochs": int(args.graph_warmup_epochs),
            "graph_warmup_lr": float(args.graph_warmup_lr),
            "num_classes": int(args.num_classes),
            "processed_dir": os.path.abspath(args.processed_dir),
        }
        observed = result.get("config", {})
        mismatches = {key: (observed.get(key), value) for key, value in expected.items() if observed.get(key) != value}
        if mismatches:
            raise RuntimeError(f"Refusing to resume fold {fold} with a mismatched configuration: {mismatches}")
        print(f"[SKIP] fold={fold} already finished: {run_dir}")
        return run_dir

    cmd = [
        args.python,
        args.train_script,
        "--fold", str(fold),
        "--seed", str(args.seed),
        "--graph_aux_weight", str(args.graph_aux_weight),
        "--graph_warmup_epochs", str(args.graph_warmup_epochs),
        "--graph_warmup_lr", str(args.graph_warmup_lr),
        "--out_dir", args.out_dir,
        "--processed_dir", args.processed_dir,
        "--num_classes", str(args.num_classes),
    ]
    if args.skip_pretrain:
        cmd.append("--skip_pretrain")
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return run_dir


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values)


def sample_sd(values):
    values = [float(value) for value in values]
    if len(values) < 2:
        return float("nan")
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def aggregate_results(run_dirs, summary_csv: str, expected_folds, allow_partial=False):
    rows = []
    for run_dir in run_dirs:
        result_path = os.path.join(run_dir, "result.json")
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Missing requested run result: {result_path}")
        result = load_json(result_path)
        cfg = result["config"]
        metrics = result["metrics"]
        confidence = result["confidence"]["ts"]
        selective = result["confidence"].get("margin", {}).get("selective", {})
        rows.append({
            "run_name": os.path.basename(run_dir),
            "fold": cfg["fold"],
            "seed": cfg["seed"],
            "skip_pretrain": cfg["skip_pretrain"],
            "graph_aux_weight": cfg["graph_aux_weight"],
            "graph_warmup_epochs": cfg["graph_warmup_epochs"],
            "graph_warmup_lr": cfg["graph_warmup_lr"],
            "num_classes": cfg["num_classes"],
            "n_folds": 1,
            "overall_accuracy": metrics["overall_accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_f1": metrics["macro_f1"],
            "macro_auroc": metrics["macro_auroc"],
            "ts_temperature": result["temperature_T"],
            "ts_ece15": confidence["ece15"],
            "ts_brier": confidence["brier"],
            "ts_brier_top1": confidence.get("brier_top1", float("nan")),
            "margin_aurc": selective.get("aurc", float("nan")),
            "margin_eaurc": selective.get("eaurc", float("nan")),
            "margin_risk80": selective.get("risk@80cov", float("nan")),
            "margin_risk90": selective.get("risk@90cov", float("nan")),
        })

    observed_folds = [int(row["fold"]) for row in rows]
    if sorted(observed_folds) != sorted(expected_folds):
        raise RuntimeError(f"Summary folds {observed_folds} do not match requested folds {expected_folds}.")
    if not allow_partial and set(expected_folds) != set(range(5)):
        raise RuntimeError("A publication summary requires folds 0..4; use --allow_partial_summary only for diagnostics.")
    config_keys = ["seed", "skip_pretrain", "graph_aux_weight", "graph_warmup_epochs", "graph_warmup_lr", "num_classes"]
    for key in config_keys:
        if len({str(row[key]) for row in rows}) != 1:
            raise RuntimeError(f"Cannot aggregate runs with different {key} values.")

    mean_row = {
        "run_name": "mean",
        "fold": "all",
        "seed": rows[0]["seed"],
        "skip_pretrain": rows[0]["skip_pretrain"],
        "graph_aux_weight": rows[0]["graph_aux_weight"],
        "graph_warmup_epochs": rows[0]["graph_warmup_epochs"],
        "graph_warmup_lr": rows[0]["graph_warmup_lr"],
        "num_classes": rows[0]["num_classes"],
        "n_folds": len(rows),
    }
    numeric_keys = [
        "overall_accuracy",
        "weighted_f1",
        "macro_f1",
        "macro_auroc",
        "ts_temperature",
        "ts_ece15",
        "ts_brier",
        "ts_brier_top1",
        "margin_aurc",
        "margin_eaurc",
        "margin_risk80",
        "margin_risk90",
    ]
    for key in numeric_keys:
        mean_row[key] = mean(row[key] for row in rows)
    rows.append(mean_row)

    fold_rows = rows[:-1]
    sd_row = {
        "run_name": "sd",
        "fold": "all",
        "seed": rows[0]["seed"],
        "skip_pretrain": rows[0]["skip_pretrain"],
        "graph_aux_weight": rows[0]["graph_aux_weight"],
        "graph_warmup_epochs": rows[0]["graph_warmup_epochs"],
        "graph_warmup_lr": rows[0]["graph_warmup_lr"],
        "num_classes": rows[0]["num_classes"],
        "n_folds": len(fold_rows),
    }
    for key in numeric_keys:
        sd_row[key] = sample_sd(row[key] for row in fold_rows)
    rows.append(sd_row)

    summary_dir = os.path.dirname(summary_csv)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] summary saved to: {summary_csv}")


def main():
    args = get_args()
    folds = parse_folds(args.folds)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.train_script):
        args.train_script = os.path.join(script_dir, args.train_script)
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(script_dir, args.out_dir)
    if not os.path.isabs(args.processed_dir):
        args.processed_dir = os.path.join(script_dir, args.processed_dir)
    if not os.path.isabs(args.summary_csv):
        args.summary_csv = os.path.join(script_dir, args.summary_csv)

    run_dirs = [run_one_fold(fold, args) for fold in folds]
    aggregate_results(run_dirs, args.summary_csv, folds, allow_partial=args.allow_partial_summary)


if __name__ == "__main__":
    main()
