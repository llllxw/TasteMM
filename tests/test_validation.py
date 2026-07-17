from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "model_comparison"
for path in (ROOT, BENCHMARK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_benchmark import read_outputs, validate_paired_inputs
from benchmark_utils import save_fold_outputs
from data_process import _indices_from_frozen_manifest
from loss import SupConHardLoss
from metric import calculate_metrics


@pytest.mark.parametrize(
    "data_path,manifest_path,num_classes",
    [
        (ROOT / "data" / "compound.csv", BENCHMARK / "manifests" / "six_class_split_manifest.csv", 6),
        (
            BENCHMARK / "scope_matched" / "manifests" / "tastemolnet_scope3_compound.csv",
            BENCHMARK / "scope_matched" / "manifests" / "tastemolnet_scope3_split_manifest.csv",
            3,
        ),
        (
            BENCHMARK / "scope_matched" / "manifests" / "virtuous_scope4_compound.csv",
            BENCHMARK / "scope_matched" / "manifests" / "virtuous_scope4_split_manifest.csv",
            4,
        ),
        (
            BENCHMARK / "scope_matched" / "manifests" / "fart_scope5_compound.csv",
            BENCHMARK / "scope_matched" / "manifests" / "fart_scope5_split_manifest.csv",
            5,
        ),
    ],
)
def test_frozen_manifest_partitions_input_exactly(data_path, manifest_path, num_classes):
    data = pd.read_csv(data_path)
    manifest = pd.read_csv(manifest_path)
    for fold in range(5):
        partitions = _indices_from_frozen_manifest(manifest, fold, data, num_classes)
        combined = partitions["train"] + partitions["val"] + partitions["test"]
        assert sorted(combined) == list(range(len(data)))


def test_contrastive_loss_handles_missing_positive_or_negative_pairs():
    features = torch.randn(6, 64, requires_grad=True)
    criterion = SupConHardLoss()
    assert criterion(features, torch.arange(6)).item() == 0.0
    assert criterion(features, torch.zeros(6, dtype=torch.long)).item() == 0.0


def test_metrics_reject_nonfinite_logits():
    with pytest.raises(ValueError, match="NaN or infinite"):
        calculate_metrics(np.arange(6), np.full((6, 6), np.nan))


def test_benchmark_validation_rejects_sample_misalignment(tmp_path):
    manifest_path = BENCHMARK / "manifests" / "six_class_split_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    rng = np.random.default_rng(42)
    models = ["TasteMM", "TasteMolNet", "Virtuous MultiTaste", "FART"]
    for model_index, model in enumerate(models):
        output = tmp_path / f"model_{model_index}"
        for fold in range(5):
            test = manifest[(manifest["fold"] == fold) & (manifest["partition"] == "test")].reset_index(drop=True)
            probabilities = rng.random((len(test), 6)) + 0.1
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            save_fold_outputs(output, model, fold, test, probabilities)

    metrics, predictions = read_outputs(tmp_path)
    assert validate_paired_inputs(metrics, predictions, manifest_path)["validation"] == "PASS"
    predictions.loc[0, "sample_uid"] = "corrupted"
    with pytest.raises(ValueError, match="sample_uid"):
        validate_paired_inputs(metrics, predictions, manifest_path)
