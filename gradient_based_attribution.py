from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from model import MBNet_Contrastive
from tools import CONTRAST_DIM, EMBED_DIM, GAT_DIM, PROCESSED_DIR


CLASS_NAMES = ["bitter", "sweet", "umami", "salty", "sour", "tasteless"]
CLASS_ORDER = list(range(len(CLASS_NAMES)))


def get_args():
    parser = argparse.ArgumentParser(
        description="Export gradient-based atom attributions for representative molecules."
    )
    parser.add_argument("--run_root", type=str, required=True, help="Run root containing fold subdirectories.")
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=PROCESSED_DIR,
        help="Processed data root used by this run.",
    )
    parser.add_argument("--input_csv", type=str, required=True, help="Source CSV containing ID, Name, SMILES, Label.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save explanations.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=1,
        help="Number of representative molecules selected per class.",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="correct_top_conf",
        choices=["correct_top_conf", "top_conf"],
        help="Representative sample selection strategy.",
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        default="",
        help="Optional comma-separated sample IDs to explain. Overrides automatic selection.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="integrated_gradients",
        choices=["integrated_gradients", "grad_x_input"],
        help="Gradient-based attribution method.",
    )
    parser.add_argument(
        "--score_mode",
        type=str,
        default="abs",
        choices=["abs", "signed"],
        help="Whether atom scores should keep only absolute magnitude or preserve signed contributions.",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="pred",
        choices=["pred", "true"],
        help="Whether to attribute the predicted class logit or the true class logit.",
    )
    parser.add_argument("--ig_steps", type=int, default=32, help="Number of steps for integrated gradients.")
    parser.add_argument("--image_size", type=int, default=520)
    parser.add_argument("--encoding", type=str, default="utf-8")
    return parser.parse_args()


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return cleaned or "sample"


def find_fold_dirs(run_root: str) -> List[str]:
    fold_dirs = []
    for path in sorted(glob.glob(os.path.join(run_root, "fold*"))):
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "result.json")):
            fold_dirs.append(path)
    if not fold_dirs:
        raise FileNotFoundError(f"No fold result directories found under: {run_root}")
    return fold_dirs


def load_run_payload(fold_dir: str) -> Dict:
    with open(os.path.join(fold_dir, "result.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_source_table(path: str, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding)
    required = {"ID", "Name", "SMILES", "Label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns {sorted(required)}; got {df.columns.tolist()}")
    df = df.copy()
    df["ID_str"] = df["ID"].astype(str)
    df["class_name"] = df["Label"].astype(int).map(lambda x: CLASS_NAMES[x - 1])
    return df


def infer_graph_dims(data_list) -> tuple[int, int]:
    if not data_list:
        return 61, 18
    graph_num_features = int(data_list[0].x.size(-1)) if hasattr(data_list[0], "x") else 61
    sample_edge_attr = getattr(data_list[0], "edge_attr", None)
    graph_edge_attr_dim = int(sample_edge_attr.size(-1)) if sample_edge_attr is not None else 18
    return graph_num_features, graph_edge_attr_dim


def build_model_from_payload(payload: Dict, data_list, device: torch.device) -> MBNet_Contrastive:
    cfg = payload["config"]
    graph_num_features, graph_edge_attr_dim = infer_graph_dims(data_list)
    model = MBNet_Contrastive(
        embed_dim=EMBED_DIM,
        num_features_xd=graph_num_features,
        gat_dim=GAT_DIM,
        contrast_dim=CONTRAST_DIM,
        num_classes=int(cfg.get("num_classes", 6)),
        use_graph=bool(cfg.get("use_graph", True)),
        use_mixfp=bool(cfg.get("use_mixfp", True)),
        use_bert=bool(cfg.get("use_bert", True)),
        use_branch_gates=bool(cfg.get("use_branch_gates", True)),
        edge_attr_dim=graph_edge_attr_dim,
        gnn_type=str(cfg.get("gnn_type", "gatv2")),
        cls_head_type=str(cfg.get("cls_head_type", "linear")),
        cls_hidden_dim=int(cfg.get("cls_hidden_dim", 128)),
        graph_gate_init=float(cfg.get("graph_gate_init", 0.35)),
        mixfp_gate_init=float(cfg.get("mixfp_gate_init", 0.85)),
        bert_gate_init=float(cfg.get("bert_gate_init", 0.75)),
        use_graph_aux_head=bool(cfg.get("use_graph_aux_head", False)),
        graph_aux_hidden_dim=int(cfg.get("graph_aux_hidden_dim", 128)),
        fuse_graph_logits_residual=bool(cfg.get("fuse_graph_logits_residual", False)),
        graph_logit_residual_weight=float(cfg.get("graph_logit_residual_weight", 0.0)),
    ).to(device)
    ckpt_path = payload["checkpoints"]["best_classify_model"]
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def load_split_data(processed_dir: str, fold: int, split: str):
    path = os.path.join(processed_dir, f"fold_{fold}", f"{split}_pyg.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed split file not found: {path}")
    return torch.load(path, weights_only=False)


def build_sample_table(run_root: str, processed_dir: str, split: str) -> pd.DataFrame:
    rows = []
    for fold_dir in find_fold_dirs(run_root):
        payload = load_run_payload(fold_dir)
        fold = int(payload["config"]["fold"])
        art_dir = os.path.join(fold_dir, "artifacts")
        labels = np.load(os.path.join(art_dir, f"{split}_labels.npy")).astype(int)
        prob_path = os.path.join(art_dir, f"{split}_probs_ts.npy")
        if not os.path.exists(prob_path):
            prob_path = os.path.join(art_dir, f"{split}_probs.npy")
        probs = np.load(prob_path)
        pred = probs.argmax(axis=1).astype(int)
        conf_path = os.path.join(art_dir, f"{split}_conf_ts_maxprob.npy")
        conf = np.load(conf_path) if os.path.exists(conf_path) else probs.max(axis=1)
        margin_path = os.path.join(art_dir, f"{split}_conf_margin.npy")
        margin = np.load(margin_path) if os.path.exists(margin_path) else np.partition(probs, -2, axis=1)[:, -1] - np.partition(probs, -2, axis=1)[:, -2]

        data_list = load_split_data(processed_dir, fold, split)
        if len(data_list) != len(labels):
            raise ValueError(f"Mismatch between processed data and predictions in fold {fold}.")

        for idx, data in enumerate(data_list):
            row = {
                "fold": fold,
                "sample_index_within_fold": idx,
                "id": getattr(data, "id", None),
                "id_str": str(getattr(data, "id", None)),
                "name": getattr(data, "name", None),
                "true_label": int(labels[idx]),
                "true_label_name": CLASS_NAMES[int(labels[idx])],
                "pred_label": int(pred[idx]),
                "pred_label_name": CLASS_NAMES[int(pred[idx])],
                "correct": int(pred[idx] == labels[idx]),
                "conf_ts_maxprob": float(conf[idx]),
                "conf_margin": float(margin[idx]),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def select_samples(sample_df: pd.DataFrame, sample_ids: List[str], selection: str, samples_per_class: int) -> pd.DataFrame:
    if sample_ids:
        selected = sample_df[sample_df["id_str"].isin(sample_ids)].copy()
        if selected.empty:
            raise ValueError("No samples matched the provided --sample_ids.")
        return selected.sort_values(["fold", "sample_index_within_fold"]).reset_index(drop=True)

    selected_parts = []
    for class_name in CLASS_NAMES:
        subset = sample_df[sample_df["true_label_name"] == class_name].copy()
        if selection == "correct_top_conf":
            correct_subset = subset[subset["correct"] == 1].copy()
            if len(correct_subset) >= samples_per_class:
                subset = correct_subset
        subset = subset.sort_values(
            by=["conf_ts_maxprob", "conf_margin"],
            ascending=[False, False],
        )
        selected_parts.append(subset.head(samples_per_class))
    return pd.concat(selected_parts, axis=0, ignore_index=True)


def color_from_score(score: float) -> tuple[float, float, float]:
    s = float(np.clip(score, 0.0, 1.0))
    low = np.array([0.89, 0.94, 0.98])
    high = np.array([0.65, 0.00, 0.15])
    rgb = low * (1.0 - s) + high * s
    return tuple(rgb.tolist())


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.max() > scores.min():
        return (scores - scores.min()) / (scores.max() - scores.min())
    return np.zeros_like(scores)


def normalize_signed_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    max_abs = float(np.max(np.abs(scores))) if len(scores) else 0.0
    if max_abs > 0.0:
        return scores / max_abs
    return np.zeros_like(scores)


def prepare_single_graph(data, device: torch.device):
    sample = data.clone()
    sample.batch = torch.zeros(sample.x.size(0), dtype=torch.long)
    return sample.to(device)


def aggregate_atom_scores(feature_attr: torch.Tensor) -> Dict[str, np.ndarray]:
    contrib = feature_attr.detach().cpu().numpy()
    abs_raw = np.abs(contrib).sum(axis=1)
    signed_raw = contrib.sum(axis=1)
    return {
        "abs_raw": abs_raw,
        "abs_norm": normalize_scores(abs_raw),
        "signed_raw": signed_raw,
        "signed_norm": normalize_signed_scores(signed_raw),
    }


def compute_grad_x_input(model, sample, target_idx: int) -> Dict[str, np.ndarray]:
    model.zero_grad(set_to_none=True)
    sample.x = sample.x.detach().clone().requires_grad_(True)
    logits = model(sample, mode="classify")
    target_logit = logits[0, target_idx]
    grads = torch.autograd.grad(target_logit, sample.x, retain_graph=False, create_graph=False)[0]
    return aggregate_atom_scores(grads * sample.x)


def compute_integrated_gradients(model, sample, target_idx: int, steps: int) -> Dict[str, np.ndarray]:
    original_x = sample.x.detach().clone()
    baseline_x = torch.zeros_like(original_x)
    total_grads = torch.zeros_like(original_x)
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=original_x.device)

    for alpha in alphas:
        interp_x = (baseline_x + alpha * (original_x - baseline_x)).detach().clone().requires_grad_(True)
        sample_step = sample.clone()
        sample_step.x = interp_x
        model.zero_grad(set_to_none=True)
        logits = model(sample_step, mode="classify")
        target_logit = logits[0, target_idx]
        grads = torch.autograd.grad(target_logit, interp_x, retain_graph=False, create_graph=False)[0]
        total_grads += grads

    avg_grads = total_grads / float(steps)
    integrated = (original_x - baseline_x) * avg_grads
    return aggregate_atom_scores(integrated)


def color_from_signed_score(score: float) -> tuple[float, float, float]:
    s = float(np.clip(abs(score), 0.0, 1.0))
    if score >= 0:
        low = np.array([0.90, 0.96, 0.90])
        high = np.array([0.12, 0.55, 0.22])
    else:
        low = np.array([0.99, 0.91, 0.93])
        high = np.array([0.78, 0.10, 0.19])
    rgb = low * (1.0 - s) + high * s
    return tuple(rgb.tolist())


def render_molecule(scores: np.ndarray, smiles: str, save_stem: str, image_size: int, score_mode: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.Mol(mol)
    if mol.GetNumConformers() == 0:
        Chem.rdDepictor.Compute2DCoords(mol)

    if score_mode == "signed":
        highlight_atoms = [idx for idx, score in enumerate(scores) if abs(float(score)) >= 0.05]
        atom_colors = {idx: color_from_signed_score(float(scores[idx])) for idx in highlight_atoms}
        atom_radii = {idx: 0.24 + 0.24 * abs(float(scores[idx])) for idx in highlight_atoms}
    else:
        highlight_atoms = list(range(mol.GetNumAtoms()))
        atom_colors = {idx: color_from_score(float(scores[idx])) for idx in highlight_atoms}
        atom_radii = {idx: 0.34 + 0.22 * float(scores[idx]) for idx in highlight_atoms}

    svg_drawer = rdMolDraw2D.MolDraw2DSVG(image_size, image_size)
    svg_drawer.drawOptions().addAtomIndices = False
    svg_drawer.drawOptions().padding = 0.04
    rdMolDraw2D.PrepareAndDrawMolecule(
        svg_drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )
    svg_drawer.FinishDrawing()
    with open(f"{save_stem}.svg", "w", encoding="utf-8") as f:
        f.write(svg_drawer.GetDrawingText())

    cairo_drawer = rdMolDraw2D.MolDraw2DCairo(image_size, image_size)
    cairo_drawer.drawOptions().addAtomIndices = False
    cairo_drawer.drawOptions().padding = 0.04
    rdMolDraw2D.PrepareAndDrawMolecule(
        cairo_drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )
    cairo_drawer.FinishDrawing()
    with open(f"{save_stem}.png", "wb") as f:
        f.write(cairo_drawer.GetDrawingText())


def explain_selected_samples(
    selected_df: pd.DataFrame,
    run_root: str,
    processed_dir: str,
    split: str,
    source_df: pd.DataFrame,
    output_dir: str,
    image_size: int,
    method: str,
    score_mode: str,
    target_mode: str,
    ig_steps: int,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fold_payload_cache: Dict[int, Dict] = {}
    fold_model_cache: Dict[int, MBNet_Contrastive] = {}
    fold_data_cache: Dict[int, list] = {}

    selected_rows = []
    atom_rows = []

    for row in selected_df.itertuples(index=False):
        fold = int(row.fold)
        if fold not in fold_payload_cache:
            candidates = sorted(glob.glob(os.path.join(run_root, f"fold{fold}_*")))
            if not candidates:
                raise FileNotFoundError(f"No fold directory found for fold {fold} under {run_root}")
            fold_dir = candidates[0]
            payload = load_run_payload(fold_dir)
            data_list = load_split_data(processed_dir, fold, split)
            model = build_model_from_payload(payload, data_list, device)
            fold_payload_cache[fold] = payload
            fold_model_cache[fold] = model
            fold_data_cache[fold] = data_list

        data_obj = fold_data_cache[fold][int(row.sample_index_within_fold)]
        sample = prepare_single_graph(data_obj, device)
        model = fold_model_cache[fold]

        with torch.no_grad():
            probs = torch.softmax(model(sample, mode="classify"), dim=1).detach().cpu().numpy().reshape(-1)

        target_idx = int(row.pred_label if target_mode == "pred" else row.true_label)
        if method == "grad_x_input":
            score_payload = compute_grad_x_input(model, sample, target_idx=target_idx)
        else:
            score_payload = compute_integrated_gradients(model, sample, target_idx=target_idx, steps=ig_steps)
        atom_scores = score_payload["signed_norm"] if score_mode == "signed" else score_payload["abs_norm"]

        source_match = source_df[source_df["ID_str"] == str(row.id_str)]
        if source_match.empty:
            raise ValueError(f"Cannot find source SMILES for ID={row.id_str}")
        source_info = source_match.iloc[0]
        smiles = str(source_info["SMILES"])
        mol = Chem.MolFromSmiles(smiles)
        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()] if mol is not None else ["?"] * len(atom_scores)

        stem = sanitize_filename(f"{method}_fold{fold}_{row.id_str}_{row.name}_{row.true_label_name}")
        render_molecule(atom_scores, smiles, os.path.join(output_dir, stem), image_size, score_mode=score_mode)

        selected_rows.append(
            {
                "fold": fold,
                "id": row.id,
                "name": row.name,
                "smiles": smiles,
                "true_label": row.true_label,
                "true_label_name": row.true_label_name,
                "pred_label": row.pred_label,
                "pred_label_name": row.pred_label_name,
                "target_mode": target_mode,
                "target_label_name": CLASS_NAMES[target_idx],
                "correct": row.correct,
                "conf_ts_maxprob": row.conf_ts_maxprob,
                "conf_margin": row.conf_margin,
                "method": method,
                "score_mode": score_mode,
                "attention_png": os.path.join(output_dir, f"{stem}.png"),
                "attention_svg": os.path.join(output_dir, f"{stem}.svg"),
            }
        )

        for atom_idx, score in enumerate(atom_scores):
            atom_rows.append(
                {
                    "fold": fold,
                    "id": row.id,
                    "name": row.name,
                    "true_label_name": row.true_label_name,
                    "pred_label_name": row.pred_label_name,
                    "target_label_name": CLASS_NAMES[target_idx],
                    "atom_index": atom_idx,
                    "atom_symbol": atom_symbols[atom_idx] if atom_idx < len(atom_symbols) else "?",
                    "attribution_score": float(score),
                    "attribution_score_abs": float(score_payload["abs_norm"][atom_idx]),
                    "attribution_score_signed": float(score_payload["signed_norm"][atom_idx]),
                    "attribution_score_abs_raw": float(score_payload["abs_raw"][atom_idx]),
                    "attribution_score_signed_raw": float(score_payload["signed_raw"][atom_idx]),
                }
            )

    return pd.DataFrame(selected_rows), pd.DataFrame(atom_rows)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    source_df = load_source_table(args.input_csv, args.encoding)
    sample_df = build_sample_table(args.run_root, args.processed_dir, args.split)
    sample_ids = [item.strip() for item in args.sample_ids.split(",") if item.strip()]
    selected_df = select_samples(
        sample_df=sample_df,
        sample_ids=sample_ids,
        selection=args.selection,
        samples_per_class=args.samples_per_class,
    )

    selected_info_df, atom_score_df = explain_selected_samples(
        selected_df=selected_df,
        run_root=args.run_root,
        processed_dir=args.processed_dir,
        split=args.split,
        source_df=source_df,
        output_dir=args.output_dir,
        image_size=args.image_size,
        method=args.method,
        score_mode=args.score_mode,
        target_mode=args.target_mode,
        ig_steps=args.ig_steps,
    )

    selected_path = os.path.join(args.output_dir, f"{args.method}_selected_samples.csv")
    atom_score_path = os.path.join(args.output_dir, f"{args.method}_atom_scores.csv")
    summary_path = os.path.join(args.output_dir, f"{args.method}_summary.json")

    selected_info_df.to_csv(selected_path, index=False, encoding="utf-8-sig")
    atom_score_df.to_csv(atom_score_path, index=False, encoding="utf-8-sig")

    summary = {
        "run_root": args.run_root,
        "processed_dir": args.processed_dir,
        "input_csv": args.input_csv,
        "split": args.split,
        "selection": args.selection,
        "samples_per_class": args.samples_per_class,
        "method": args.method,
        "score_mode": args.score_mode,
        "target_mode": args.target_mode,
        "ig_steps": args.ig_steps if args.method == "integrated_gradients" else None,
        "n_selected": int(len(selected_info_df)),
        "class_order": CLASS_NAMES,
        "selected_samples_csv": selected_path,
        "atom_scores_csv": atom_score_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Selected samples saved to: {selected_path}")
    print(f"[DONE] Atom attribution scores saved to: {atom_score_path}")
    print(f"[DONE] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
