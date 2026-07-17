from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    EarlyStoppingCallback, Trainer,
)

from benchmark_utils import save_classification_fold_outputs, validate_folds
from run_fart_sixclass import (
    DEFAULT_CHECKPOINT, DEFAULT_REVISION, SmilesDataset, aggregate_smiles_ensemble, model_frame, training_arguments,
)


HERE = Path(__file__).resolve().parent
MODEL_NAME = "FART"
CLASS_NAMES = ["bitter", "sweet", "sour", "umami", "undefined"]
TASK = "scope5_bitter_sweet_sour_umami_undefined_12706"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FART on its five-class scope-matched task.")
    parser.add_argument("--manifest", type=Path, default=HERE / "scope_matched" / "manifests" / "fart_scope5_split_manifest.csv")
    parser.add_argument("--output", type=Path, default=HERE / "scope_matched" / "outputs" / "fart_scope5")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--augment-factor", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--inference-augmentations", type=int, default=10)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.folds = validate_folds(args.folds)
    RDLogger.DisableLog("rdApp.*")
    if not torch.cuda.is_available():
        raise RuntimeError("FART scope5 fine-tuning requires a CUDA GPU.")

    manifest = pd.read_csv(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({
        "task": TASK, "model": MODEL_NAME, "classes": CLASS_NAMES,
        "method_source_doi": "10.1038/s41538-025-00474-z",
        "implementation_boundary": "Independent task adaptation; this is not the authors' hosted ensemble.",
    })
    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    for fold in args.folds:
        pred_path = args.output / f"fold{fold}_predictions.csv"
        metric_path = args.output / f"fold{fold}_metrics.csv"
        if args.resume and pred_path.exists() and metric_path.exists():
            print(f"[SKIP] fold {fold}")
            continue
        train = manifest[(manifest["fold"] == fold) & (manifest["partition"] == "train")]
        val = manifest[(manifest["fold"] == fold) & (manifest["partition"] == "val")]
        test = manifest[(manifest["fold"] == fold) & (manifest["partition"] == "test")].reset_index(drop=True)
        train_frame = model_frame(train, args.augment_factor, 42 + fold)
        val_frame = model_frame(val, args.augment_factor, 142 + fold)
        test_frame = model_frame(test, args.inference_augmentations, 242 + fold)
        fold_seed = 42 + fold
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        torch.cuda.manual_seed_all(fold_seed)

        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint, revision=args.revision, local_files_only=args.local_files_only
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.checkpoint, revision=args.revision, num_labels=5, ignore_mismatched_sizes=True,
            local_files_only=args.local_files_only,
        )
        trainer = Trainer(
            model=model,
            args=training_arguments(args.output, args, fold),
            train_dataset=SmilesDataset(train_frame, tokenizer, args.max_length),
            eval_dataset=SmilesDataset(val_frame, tokenizer, args.max_length),
            data_collator=DataCollatorWithPadding(tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer.train(resume_from_checkpoint=bool(args.resume and (args.output / f"fold{fold}_checkpoints").exists()))
        prediction = trainer.predict(SmilesDataset(test_frame, tokenizer, args.max_length))
        logits = np.asarray(prediction.predictions)
        logits -= logits.max(axis=1, keepdims=True)
        variant_probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        probabilities, agreement = aggregate_smiles_ensemble(
            variant_probabilities, test_frame["parent_index"].to_numpy(int), len(test)
        )
        save_classification_fold_outputs(args.output, MODEL_NAME, fold, test, probabilities, CLASS_NAMES, TASK)
        saved = pd.read_csv(pred_path)
        saved["fart_smiles_ensemble_size"] = args.inference_augmentations
        saved["fart_vote_agreement"] = agreement
        saved["fart_unanimous"] = agreement == 1.0
        saved.to_csv(pred_path, index=False, encoding="utf-8-sig")
        trainer.save_model(args.output / f"fold{fold}_best_model")
        tokenizer.save_pretrained(args.output / f"fold{fold}_best_model")
        print(f"[DONE] fold={fold}")


if __name__ == "__main__":
    main()
