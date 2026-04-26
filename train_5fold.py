from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys


def get_args():
    parser = argparse.ArgumentParser(description="Run 5-fold CV for the final baseline.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--train_script", type=str, default="train.py")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--graph_aux_weight", type=float, default=0.2)
    parser.add_argument("--graph_warmup_epochs", type=int, default=30)
    parser.add_argument("--graph_warmup_lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="runs")
    parser.add_argument("--summary_csv", type=str, default="results/baseline_5fold_summary.csv")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_folds(folds_str: str):
    return [int(part.strip()) for part in folds_str.split(",") if part.strip()]


def build_model_tag(args) -> str:
    pre_flag = int(not args.skip_pretrain)
    gaux = str(args.graph_aux_weight).replace(".", "p")
    return f"baseline_gatv2_rich_pre{pre_flag}_gaux{gaux}_gwarm{args.graph_warmup_epochs}"


def build_run_name(fold: int, seed: int, args) -> str:
    return f"fold{fold}_seed{seed}_{build_model_tag(args)}"


def run_one_fold(fold: int, args) -> str:
    run_name = build_run_name(fold, args.seed, args)
    run_dir = os.path.join(args.out_dir, run_name)
    result_json = os.path.join(run_dir, "result.json")
    if args.resume and os.path.exists(result_json):
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
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def aggregate_results(run_dirs, summary_csv: str):
    rows = []
    for run_dir in run_dirs:
        result_path = os.path.join(run_dir, "result.json")
        if not os.path.exists(result_path):
            continue
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

    if not rows:
        print("[WARN] no finished runs found")
        return

    mean_row = {
        "run_name": "mean",
        "fold": "all",
        "seed": rows[0]["seed"],
        "skip_pretrain": rows[0]["skip_pretrain"],
        "graph_aux_weight": rows[0]["graph_aux_weight"],
        "graph_warmup_epochs": rows[0]["graph_warmup_epochs"],
        "graph_warmup_lr": rows[0]["graph_warmup_lr"],
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

    sd_row = {
        "run_name": "sd",
        "fold": "all",
        "seed": rows[0]["seed"],
        "skip_pretrain": rows[0]["skip_pretrain"],
        "graph_aux_weight": rows[0]["graph_aux_weight"],
        "graph_warmup_epochs": rows[0]["graph_warmup_epochs"],
        "graph_warmup_lr": rows[0]["graph_warmup_lr"],
    }
    fold_rows = rows[:-1]
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
    if not os.path.isabs(args.summary_csv):
        args.summary_csv = os.path.join(script_dir, args.summary_csv)

    run_dirs = [run_one_fold(fold, args) for fold in folds]
    aggregate_results(run_dirs, args.summary_csv)


if __name__ == "__main__":
    main()
