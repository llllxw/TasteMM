"""CPU-friendly dimensional and forward-pass test for the public TasteMM package."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("USE_TF", "0")

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process import EDGE_FEATURE_DIM, MIXFP_DIM, NODE_FEATURE_DIM, get_mix_fingerprint, smiles_to_graph
from loss import SupConHardLoss
from metric import calculate_metrics
from model import TasteBaselineModel


def build_mock_batch():
    examples = ["CCO", "CC(=O)O", "CCN", "c1ccccc1", "O=C=O", "CCS"]
    records = []
    for label, smiles in enumerate(examples):
        x, edge_index, edge_attr = smiles_to_graph(smiles)
        records.append(
            Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                mixfp=torch.tensor(get_mix_fingerprint(smiles), dtype=torch.float32).reshape(1, -1),
                bert=torch.zeros((1, 768), dtype=torch.float32),
                y=torch.tensor([label], dtype=torch.long),
            )
        )
    return next(iter(DataLoader(records, batch_size=len(records), shuffle=False)))


def main():
    torch.manual_seed(42)
    batch = build_mock_batch()
    assert batch.x.shape[1] == NODE_FEATURE_DIM == 61
    assert batch.edge_attr.shape[1] == EDGE_FEATURE_DIM == 18
    assert batch.mixfp.shape[1] == MIXFP_DIM == 3239
    assert batch.bert.shape[1] == 768

    model = TasteBaselineModel(
        num_graph_features=NODE_FEATURE_DIM,
        edge_attr_dim=EDGE_FEATURE_DIM,
        mixfp_dim=MIXFP_DIM,
        num_classes=6,
    )
    model.eval()
    with torch.no_grad():
        logits = model(batch, mode="classify")
        features = model(batch, mode="contrastive")
    assert tuple(logits.shape) == (6, 6)
    assert tuple(features.shape) == (6, 64)

    metrics = calculate_metrics(batch.y.numpy(), logits.numpy())
    paired_features = torch.cat([features, features], dim=0)
    paired_labels = torch.arange(6).repeat(2)
    contrastive_loss = SupConHardLoss()(paired_features, paired_labels)
    assert torch.isfinite(contrastive_loss)
    no_negative_loss = SupConHardLoss()(features, torch.zeros(6, dtype=torch.long))
    no_positive_loss = SupConHardLoss()(features, torch.arange(6))
    assert float(no_negative_loss) == 0.0 and float(no_positive_loss) == 0.0

    batch_without_edges = batch.clone()
    del batch_without_edges.edge_attr
    custom_edge_model = TasteBaselineModel(
        num_graph_features=NODE_FEATURE_DIM, edge_attr_dim=7, mixfp_dim=MIXFP_DIM, num_classes=6
    )
    custom_edge_model.eval()
    with torch.no_grad():
        assert custom_edge_model(batch_without_edges).shape == (6, 6)

    try:
        calculate_metrics(batch.y.numpy(), np.full((6, 6), np.nan))
    except ValueError:
        pass
    else:
        raise AssertionError("calculate_metrics must reject non-finite logits")

    result = {
        "status": "PASS",
        "node_feature_dim": NODE_FEATURE_DIM,
        "edge_feature_dim": EDGE_FEATURE_DIM,
        "mixed_fingerprint_dim": MIXFP_DIM,
        "bert_dim": 768,
        "logits_shape": list(logits.shape),
        "contrastive_shape": list(features.shape),
        "macro_auroc_is_finite": bool(np.isfinite(metrics["macro_auroc"])),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
