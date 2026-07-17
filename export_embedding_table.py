"""Aggregate fold test embeddings and metadata for plot_embedding_umap_tsne.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_NAMES = ["bitter", "sweet", "umami", "salty", "sour", "tasteless"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one matched test embedding table from five TasteMM runs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, default=Path("data/compound.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.source_csv).reset_index(names="row_index")
    fused_parts, graph_parts, metadata_parts = [], [], []
    for fold in range(5):
        candidates = [path for path in args.run_root.glob(f"fold{fold}_*") if (path / "result.json").exists()]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one completed run for fold {fold}, found {len(candidates)}.")
        artifacts = candidates[0] / "artifacts"
        row_indices = np.load(artifacts / "test_row_indices.npy").astype(int)
        labels = np.load(artifacts / "test_labels.npy").astype(int)
        probs = np.load(artifacts / "test_probs_ts.npy")
        fused = np.load(artifacts / "test_fused_embeddings.npy")
        graph = np.load(artifacts / "test_graph_embeddings.npy")
        n = len(row_indices)
        if not (len(labels) == len(probs) == len(fused) == len(graph) == n):
            raise ValueError(f"Fold {fold} artifact lengths are inconsistent.")
        expected_labels = source.iloc[row_indices]["Label"].astype(int).to_numpy() - 1
        if not np.array_equal(labels, expected_labels):
            raise ValueError(f"Fold {fold} labels do not align with source row_index.")
        frame = source.iloc[row_indices].copy().reset_index(drop=True)
        frame["fold"] = fold
        frame["true_label"] = labels
        frame["true_label_name"] = [CLASS_NAMES[value] for value in labels]
        frame["pred_label"] = probs.argmax(axis=1)
        frame["correct"] = (frame["pred_label"].to_numpy() == labels).astype(int)
        frame["conf_ts_maxprob"] = probs.max(axis=1)
        metadata_parts.append(frame)
        fused_parts.append(fused)
        graph_parts.append(graph)

    metadata = pd.concat(metadata_parts, ignore_index=True)
    if metadata["row_index"].duplicated().any():
        raise ValueError("A source row appears in more than one exported test fold.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output_dir / "test_embedding_metadata.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        args.output_dir / "test_embeddings.npz",
        fused_embeddings=np.concatenate(fused_parts),
        graph_embeddings=np.concatenate(graph_parts),
    )
    summary = {
        "n_samples": int(len(metadata)),
        "class_order": CLASS_NAMES,
        "folds": list(range(5)),
        "note": "Each row appears once because the five outer held-out sets are disjoint; this export uses each fold's test half.",
    }
    (args.output_dir / "test_embedding_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
