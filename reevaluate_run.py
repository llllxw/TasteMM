"""Re-export auditable artifacts from an existing TasteMM checkpoint without retraining."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from model import TasteBaselineModel
from tools import CONTRAST_DIM, EMBED_DIM, GAT_DIM, PROCESSED_DIR, setup_seed
from train import (
    evaluate_confbest, load_fold_data, save_run_artifacts, save_test_prediction_csv, sha256_file,
)


def resolve_checkpoint(run_dir: str, payload: dict, explicit: str) -> str:
    recorded = payload.get("checkpoints", {}).get("best_classify_model", "")
    candidates = [explicit] if explicit else []
    if recorded:
        candidates.append(recorded if os.path.isabs(recorded) else os.path.join(run_dir, recorded))
    candidates.append(os.path.join(run_dir, "checkpoints", "best_classify_model.pth"))
    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(f"No checkpoint found. Checked: {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate an existing TasteMM checkpoint and export current artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--processed-dir", default=PROCESSED_DIR)
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    result_path = os.path.join(run_dir, "result.json")
    with open(result_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload["config"]
    fold, seed = int(config["fold"]), int(config.get("seed", 42))
    num_classes = int(config.get("num_classes", 6))
    setup_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, train_data = load_fold_data(fold, os.path.abspath(args.processed_dir))
    model = TasteBaselineModel(
        embed_dim=EMBED_DIM,
        num_graph_features=int(train_data[0].x.size(-1)),
        edge_attr_dim=int(train_data[0].edge_attr.size(-1)),
        gat_dim=GAT_DIM,
        contrast_dim=CONTRAST_DIM,
        graph_aux_hidden_dim=128,
        num_classes=num_classes,
    ).to(device)
    checkpoint = resolve_checkpoint(run_dir, payload, args.checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    metrics, confidence, arrays = evaluate_confbest(model, val_loader, test_loader, device)
    relative_checkpoint = os.path.relpath(checkpoint, run_dir).replace("\\", "/")
    payload["metrics"] = metrics
    payload["confidence"] = confidence
    payload["temperature_T"] = confidence["ts"]["T"]
    payload["checkpoints"] = {"best_classify_model": relative_checkpoint, "sha256": sha256_file(checkpoint)}
    payload["reevaluation"] = {
        "timestamp": int(time.time()),
        "processed_dir": os.path.abspath(args.processed_dir),
        "note": "Existing checkpoint evaluated without parameter updates using the current artifact schema.",
    }
    save_run_artifacts(run_dir, payload, arrays)
    save_test_prediction_csv(run_dir, arrays, num_classes)
    print(f"[DONE] Re-evaluated fold {fold} without retraining: {run_dir}")


if __name__ == "__main__":
    main()
