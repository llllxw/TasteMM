from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass

os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from calibration_metrics import (
    brier_score,
    brier_top1,
    classwise_brier_score,
    classwise_ece_top1,
    ece_top1,
    reliability_bins_top1,
    selective_metrics,
    true_class_conditioned_brier_top1,
    true_class_conditioned_selective_metrics,
)
from confidence_methods import conf_margin_from_logits, softmax_np, temperature_scale_logits
from loss import SupConHardLoss
from metric import calculate_metrics
from model import TasteBaselineModel
from tools import (
    BATCH_SIZE,
    CONTRAST_DIM,
    EMBED_DIM,
    FINETUNE_EPOCHS,
    GAT_DIM,
    PATIENCE,
    PRETRAIN_EPOCHS,
    PROCESSED_DIR,
    TEMPERATURE,
    setup_seed,
    training_config_tag,
    training_run_name,
)


@dataclass
class RunConfig:
    fold: int
    seed: int
    skip_pretrain: bool
    graph_aux_weight: float
    graph_warmup_epochs: int
    graph_warmup_lr: float
    out_dir: str
    num_classes: int
    processed_dir: str


def to_py(obj):
    if isinstance(obj, np.generic):
        return to_py(obj.item())
    if isinstance(obj, np.ndarray):
        return to_py(obj.tolist())
    if torch.is_tensor(obj):
        return to_py(obj.detach().cpu().numpy().tolist())
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(item) for item in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trusted_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was exposed.
        return torch.load(path, map_location="cpu")


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def save_run_artifacts(run_dir: str, payload_json: dict, arrays: dict) -> None:
    os.makedirs(run_dir, exist_ok=True)
    art_dir = os.path.join(run_dir, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(to_py(payload_json), f, ensure_ascii=False, indent=2, allow_nan=False)
    for name, arr in arrays.items():
        np.save(os.path.join(art_dir, f"{name}.npy"), arr)


def save_test_prediction_csv(run_dir: str, arrays: dict, num_classes: int) -> None:
    prediction_frame = {
        "row_index": arrays["test_row_indices"],
        "true_label_zero_based": arrays["test_labels"],
        "pred_label_zero_based": arrays["test_probs_raw"].argmax(axis=1),
    }
    for class_index in range(num_classes):
        prediction_frame[f"prob_class_{class_index}"] = arrays["test_probs_raw"][:, class_index]
    pd.DataFrame(prediction_frame).to_csv(
        os.path.join(run_dir, "artifacts", "test_predictions.csv"), index=False, encoding="utf-8-sig"
    )


def get_args():
    parser = argparse.ArgumentParser(description="Train the final baseline model.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--graph_aux_weight", type=float, default=0.2)
    parser.add_argument("--graph_warmup_epochs", type=int, default=30)
    parser.add_argument("--graph_warmup_lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="runs")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--processed_dir", type=str, default=PROCESSED_DIR)
    return parser.parse_args()


def build_run_config(args) -> RunConfig:
    if args.fold not in range(5):
        raise ValueError("--fold must be one of 0,1,2,3,4.")
    if args.num_classes < 2:
        raise ValueError("--num_classes must be at least 2.")
    if args.graph_aux_weight < 0:
        raise ValueError("graph_aux_weight must be non-negative.")
    if args.graph_warmup_epochs < 0:
        raise ValueError("graph_warmup_epochs must be non-negative.")
    if args.graph_warmup_lr <= 0:
        raise ValueError("graph_warmup_lr must be positive.")
    return RunConfig(
        fold=int(args.fold),
        seed=int(args.seed),
        skip_pretrain=bool(args.skip_pretrain),
        graph_aux_weight=float(args.graph_aux_weight),
        graph_warmup_epochs=int(args.graph_warmup_epochs),
        graph_warmup_lr=float(args.graph_warmup_lr),
        out_dir=str(args.out_dir),
        num_classes=int(args.num_classes),
        processed_dir=os.path.abspath(args.processed_dir),
    )


def build_model_tag(config: RunConfig) -> str:
    return training_config_tag(asdict(config))


def load_fold_data(fold: int, processed_dir: str):
    fold_dir = os.path.join(processed_dir, f"fold_{fold}")
    train_data = trusted_torch_load(os.path.join(fold_dir, "train_pyg.pt"))
    val_data = trusted_torch_load(os.path.join(fold_dir, "val_pyg.pt"))
    test_data = trusted_torch_load(os.path.join(fold_dir, "test_pyg.pt"))
    for split_name, dataset in (("train", train_data), ("validation", val_data), ("test", test_data)):
        if not dataset:
            raise ValueError(f"Fold {fold} {split_name} processed dataset is empty.")
        if not all(hasattr(item, "row_index") for item in dataset):
            raise RuntimeError("Processed samples lack row_index. Regenerate them with the current data_process.py.")
    manifest_path = os.path.join(fold_dir, "split_manifest.csv")
    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
        for split_name, dataset in (("train", train_data), ("validation", val_data), ("test", test_data)):
            expected = manifest.loc[manifest["split"] == split_name, "row_index"].astype(int).to_numpy()
            observed = np.asarray([int(item.row_index.reshape(-1)[0].item()) for item in dataset])
            if not np.array_equal(observed, expected):
                raise ValueError(f"Fold {fold} {split_name} PyG order does not match split_manifest.csv.")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader, train_data


def pretrain_contrastive(model, train_loader, optimizer, criterion, device, epochs: int):
    losses = []
    for _ in tqdm(range(epochs), desc="Contrastive pretrain"):
        model.train()
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            labels = data.y.reshape(-1)
            embeds = model(data, mode="contrastive")
            loss = criterion(embeds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        losses.append(total_loss / len(train_loader.dataset))
    return model, losses


def set_graph_warmup_requires_grad(model) -> None:
    trainable_prefixes = ("gat", "graph_")
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(trainable_prefixes)


def enable_all_requires_grad(model) -> None:
    for param in model.parameters():
        param.requires_grad = True


def make_class_weights(train_data, num_classes: int, device):
    train_labels = [int(d.y.item()) for d in train_data]
    class_counts = np.bincount(train_labels, minlength=num_classes)
    if len(class_counts) != num_classes or np.any(class_counts == 0):
        raise ValueError(f"Training partition must contain all {num_classes} classes; counts={class_counts.tolist()}")
    weights = [len(train_labels) / (num_classes * max(int(count), 1)) for count in class_counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def graph_only_warmup(model, train_loader, val_loader, device, epochs, lr, train_data, num_classes):
    set_graph_warmup_requires_grad(model)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=make_class_weights(train_data, num_classes, device))

    best_val_f1 = -1.0
    best_state = None
    warmup_losses = []
    warmup_val_f1 = []

    for _ in tqdm(range(epochs), desc="Graph warmup"):
        model.train()
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            logits = model(data, mode="graph_aux")
            labels = data.y.reshape(-1)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        warmup_losses.append(total_loss / len(train_loader.dataset))

        model.eval()
        val_logits = []
        val_labels = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                val_logits.append(model(data, mode="graph_aux").cpu().numpy())
                val_labels.append(data.y.reshape(-1).cpu().numpy())
        val_logits = np.concatenate(val_logits, axis=0)
        val_labels = np.concatenate(val_labels, axis=0).astype(int)
        val_f1 = float(calculate_metrics(val_labels, val_logits)["weighted_f1"])
        warmup_val_f1.append(val_f1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    enable_all_requires_grad(model)
    return model, warmup_losses, warmup_val_f1


def finetune_classification(model, train_loader, val_loader, optimizer, device, epochs, patience, train_data, num_classes, graph_aux_weight):
    criterion = nn.CrossEntropyLoss(weight=make_class_weights(train_data, num_classes, device))
    best_weighted_f1 = -1.0
    not_improved = 0
    best_state = None
    train_losses = []
    graph_aux_losses = []
    val_f1_history = []

    for _ in tqdm(range(epochs), desc="Classification finetune"):
        model.train()
        total_loss = 0.0
        total_graph_aux_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            labels = data.y.reshape(-1)
            logits = model(data, mode="classify")
            main_loss = criterion(logits, labels)
            # The reported model used a separate stochastic forward pass for the graph auxiliary
            # head. Keeping it explicit preserves its independent dropout mask and checkpoint behavior.
            graph_aux_logits = model(data, mode="graph_aux")
            aux_loss = criterion(graph_aux_logits, labels)
            loss = main_loss + graph_aux_weight * aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            total_graph_aux_loss += aux_loss.item() * data.num_graphs

        train_losses.append(total_loss / len(train_loader.dataset))
        graph_aux_losses.append(total_graph_aux_loss / len(train_loader.dataset))

        model.eval()
        val_logits = []
        val_labels = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                val_logits.append(model(data, mode="classify").cpu().numpy())
                val_labels.append(data.y.reshape(-1).cpu().numpy())
        val_logits = np.concatenate(val_logits, axis=0)
        val_labels = np.concatenate(val_labels, axis=0).astype(int)
        val_f1 = float(calculate_metrics(val_labels, val_logits)["weighted_f1"])
        val_f1_history.append(val_f1)

        if val_f1 > best_weighted_f1:
            best_weighted_f1 = val_f1
            not_improved = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            not_improved += 1
            if not_improved >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, graph_aux_losses, val_f1_history


def collect_outputs(model, loader, device, include_features=False):
    model.eval()
    logits_list = []
    labels_list = []
    row_indices = []
    fused_list = []
    graph_list = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            fused, graph = model.encode(data)
            logits_list.append(model.class_head(fused).cpu().numpy())
            labels_list.append(data.y.reshape(-1).cpu().numpy())
            if not hasattr(data, "row_index"):
                raise RuntimeError("Processed samples lack row_index. Regenerate them with data_process.py.")
            row_indices.append(data.row_index.reshape(-1).cpu().numpy())
            if include_features:
                fused_list.append(fused.cpu().numpy())
                graph_list.append(graph.cpu().numpy())
    outputs = {
        "logits": np.concatenate(logits_list, axis=0),
        "labels": np.concatenate(labels_list, axis=0).astype(int),
        "row_indices": np.concatenate(row_indices, axis=0).astype(int),
    }
    if include_features:
        outputs["fused_embeddings"] = np.concatenate(fused_list, axis=0)
        outputs["graph_embeddings"] = np.concatenate(graph_list, axis=0)
    return outputs


def fit_temperature_gridsearch(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    temperatures = np.linspace(0.5, 5.0, 46)
    best_t = 1.0
    best_nll = 1e18
    y = torch.from_numpy(val_labels).long()
    for temp in temperatures:
        scaled = torch.from_numpy(val_logits / float(temp)).float()
        nll = nn.CrossEntropyLoss()(scaled, y).item()
        if nll < best_nll:
            best_nll = nll
            best_t = float(temp)
    return best_t


def evaluate_confbest(model, val_loader, test_loader, device):
    val_outputs = collect_outputs(model, val_loader, device)
    val_logits, val_labels = val_outputs["logits"], val_outputs["labels"]
    temperature = fit_temperature_gridsearch(val_logits, val_labels)

    test_outputs = collect_outputs(model, test_loader, device, include_features=True)
    test_logits, test_labels = test_outputs["logits"], test_outputs["labels"]
    test_probs_raw = softmax_np(test_logits)
    test_logits_ts = temperature_scale_logits(test_logits, temperature)
    test_probs_ts = softmax_np(test_logits_ts)
    test_pred = test_probs_ts.argmax(axis=1)
    test_conf_margin = conf_margin_from_logits(test_logits_ts)

    metrics = calculate_metrics(test_labels, test_logits)
    confidence = {
        "ts": {
            "T": float(temperature),
            "ece15": float(ece_top1(test_probs_ts, test_labels, n_bins=15)),
            "brier": float(brier_score(test_probs_ts, test_labels)),
            "brier_top1": float(brier_top1(test_probs_ts, test_labels)),
            "reliability_bins15": reliability_bins_top1(test_probs_ts, test_labels, n_bins=15),
            "classwise_ece15": classwise_ece_top1(test_probs_ts, test_labels, n_bins=15),
            "classwise_brier": classwise_brier_score(test_probs_ts, test_labels),
            "true_class_conditioned_brier_top1": true_class_conditioned_brier_top1(test_probs_ts, test_labels),
        },
        "margin": {
            "selective": selective_metrics(test_conf_margin, test_labels, test_pred, include_curve=True),
            "true_class_conditioned_selective": true_class_conditioned_selective_metrics(
                test_conf_margin, test_labels, test_pred
            ),
        },
    }
    arrays = {
        "val_logits": val_logits,
        "val_labels": val_labels,
        "val_row_indices": val_outputs["row_indices"],
        "test_logits": test_logits,
        "test_labels": test_labels,
        "test_row_indices": test_outputs["row_indices"],
        "test_probs_raw": test_probs_raw,
        "test_probs_ts": test_probs_ts,
        "test_conf_margin": test_conf_margin,
        "test_conf_ts_maxprob": test_probs_ts.max(axis=1),
        "test_fused_embeddings": test_outputs["fused_embeddings"],
        "test_graph_embeddings": test_outputs["graph_embeddings"],
    }
    return metrics, confidence, arrays


def main():
    args = get_args()
    setup_seed(args.seed)

    config = build_run_config(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, train_data = load_fold_data(config.fold, config.processed_dir)

    run_name = training_run_name(config.fold, config.seed, asdict(config))
    run_dir = os.path.join(config.out_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    graph_num_features = int(train_data[0].x.size(-1))
    edge_attr_dim = int(train_data[0].edge_attr.size(-1))
    model = TasteBaselineModel(
        embed_dim=EMBED_DIM,
        num_graph_features=graph_num_features,
        edge_attr_dim=edge_attr_dim,
        gat_dim=GAT_DIM,
        contrast_dim=CONTRAST_DIM,
        graph_aux_hidden_dim=128,
        num_classes=config.num_classes,
    ).to(device)

    pretrain_losses = []
    if not config.skip_pretrain:
        criterion = SupConHardLoss(temperature=TEMPERATURE)
        optimizer = optim.Adam(model.parameters(), lr=5e-6, weight_decay=1e-5)
        model, pretrain_losses = pretrain_contrastive(model, train_loader, optimizer, criterion, device, PRETRAIN_EPOCHS)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "pretrain_model.pth"))

    graph_warmup_losses = []
    graph_warmup_val_f1 = []
    if config.graph_warmup_epochs > 0:
        model, graph_warmup_losses, graph_warmup_val_f1 = graph_only_warmup(
            model,
            train_loader,
            val_loader,
            device,
            config.graph_warmup_epochs,
            config.graph_warmup_lr,
            train_data,
            config.num_classes,
        )
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "graph_warmup_model.pth"))

    optimizer = optim.Adam(model.parameters(), lr=3e-6, weight_decay=1e-4)
    model, finetune_losses, graph_aux_losses, finetune_val_f1 = finetune_classification(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        FINETUNE_EPOCHS,
        PATIENCE,
        train_data,
        config.num_classes,
        config.graph_aux_weight,
    )

    best_ckpt_path = os.path.join(ckpt_dir, "best_classify_model.pth")
    torch.save(model.state_dict(), best_ckpt_path)
    test_metrics, confidence_block, arrays = evaluate_confbest(model, val_loader, test_loader, device)
    manifest_path = os.path.join(config.processed_dir, f"fold_{config.fold}", "split_manifest.csv")
    processing_metadata_path = os.path.join(config.processed_dir, "processing_metadata.json")
    checkpoint_relpath = os.path.relpath(best_ckpt_path, run_dir).replace("\\", "/")
    payload = {
        "config": asdict(config),
        "timestamp": int(time.time()),
        "run_dir": os.path.abspath(run_dir),
        "run_name": run_name,
        "model_tag": build_model_tag(config),
        "checkpoints": {
            "best_classify_model": checkpoint_relpath,
            "sha256": sha256_file(best_ckpt_path),
        },
        "temperature_T": confidence_block["ts"]["T"],
        "metrics": test_metrics,
        "confidence": confidence_block,
        "training": {
            "pretrain_losses": pretrain_losses,
            "graph_warmup_losses": graph_warmup_losses,
            "graph_warmup_val_f1": graph_warmup_val_f1,
            "finetune_train_losses": finetune_losses,
            "finetune_graph_aux_losses": graph_aux_losses,
            "finetune_val_f1": finetune_val_f1,
            "model": {
                "graph_encoder": "gatv2",
                "graph_num_features": graph_num_features,
                "edge_attr_dim": edge_attr_dim,
                "use_graph_aux_head": True,
                "graph_aux_weight": config.graph_aux_weight,
                "graph_warmup_epochs": config.graph_warmup_epochs,
            },
        },
        "provenance": {
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "split_manifest_sha256": sha256_file(manifest_path) if os.path.exists(manifest_path) else None,
            "processing_metadata_sha256": (
                sha256_file(processing_metadata_path) if os.path.exists(processing_metadata_path) else None
            ),
            "batch_size": BATCH_SIZE,
            "pretrain_epochs_max": PRETRAIN_EPOCHS,
            "finetune_epochs_max": FINETUNE_EPOCHS,
            "patience": PATIENCE,
            "contrastive_temperature": TEMPERATURE,
            "pretrain_optimizer": {"name": "Adam", "learning_rate": 5e-6, "weight_decay": 1e-5},
            "finetune_optimizer": {"name": "Adam", "learning_rate": 3e-6, "weight_decay": 1e-4},
        },
    }
    save_run_artifacts(run_dir, payload, arrays)
    save_test_prediction_csv(run_dir, arrays, config.num_classes)

    print(f"[DONE] Run saved to: {run_dir}")
    print(
        f"Accuracy={float(test_metrics['overall_accuracy']):.4f} "
        f"WeightedF1={float(test_metrics['weighted_f1']):.4f} "
        f"MacroF1={float(test_metrics['macro_f1']):.4f} "
        f"MacroAUROC={float(test_metrics['macro_auroc']):.4f}"
    )


if __name__ == "__main__":
    main()
