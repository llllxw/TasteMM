from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from pathlib import Path

# Prevent an unrelated local Keras 3 installation from breaking transformers.Trainer.
os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger, rdBase
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from benchmark_utils import save_fold_outputs, validate_folds


HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = "seyonec/SMILES_tokenized_PubChem_shard00_160k"
DEFAULT_REVISION = "f0854db6cbaad4655ce3bb0c073b9ba0199f4a7d"
MODEL_NAME = "FART"


class SmilesDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.smiles = frame["model_smiles"].astype(str).tolist()
        self.labels = frame["model_label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> dict:
        encoded = self.tokenizer(
            self.smiles[index], truncation=True, max_length=self.max_length, padding=False
        )
        encoded["labels"] = self.labels[index]
        return encoded


def randomized_smiles(smiles: str, n: int, rng: random.Random) -> list[str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    variants: dict[str, None] = {}
    attempts = 0
    while len(variants) < n and attempts < n * 10:
        variants.setdefault(Chem.MolToSmiles(mol, canonical=False, doRandom=True), None)
        attempts += 1
    if not variants:
        variants.setdefault(Chem.MolToSmiles(mol, canonical=True), None)
    values = list(variants)
    rng.shuffle(values)
    while len(values) < n:
        values.append(values[len(values) % len(variants)])
    return values[:n]


def model_frame(partition: pd.DataFrame, augment_factor: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rdBase.SeedRandomNumberGenerator(seed)
    rows = []
    for parent_index, row in enumerate(partition.reset_index(drop=True).itertuples(index=False)):
        variants = randomized_smiles(row.smiles, augment_factor, rng) if augment_factor > 1 else [row.smiles]
        for value in variants:
            rows.append(
                {
                    "model_smiles": value,
                    "model_label": int(row.true_label) - 1,
                    "parent_index": parent_index,
                }
            )
    return pd.DataFrame(rows)


def aggregate_smiles_ensemble(
    probabilities: np.ndarray, parent_indices: np.ndarray, n_molecules: int
) -> tuple[np.ndarray, np.ndarray]:
    averaged = np.zeros((n_molecules, probabilities.shape[1]), dtype=float)
    agreement = np.zeros(n_molecules, dtype=float)
    votes = probabilities.argmax(axis=1)
    for parent_index in range(n_molecules):
        mask = parent_indices == parent_index
        if not np.any(mask):
            raise RuntimeError(f"Missing FART predictions for test molecule {parent_index}")
        averaged[parent_index] = probabilities[mask].mean(axis=0)
        counts = np.bincount(votes[mask], minlength=probabilities.shape[1])
        agreement[parent_index] = counts.max() / mask.sum()
    return averaged, agreement


def training_arguments(output: Path, args, fold: int) -> TrainingArguments:
    kwargs = dict(
        output_dir=str(output / f"fold{fold}_checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        seed=42 + fold,
        data_seed=42 + fold,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=True,
    )
    parameter_names = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in parameter_names else "evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def smoke_test(checkpoint: str, revision: str, local_files_only: bool, max_length: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, revision=revision, num_labels=6, ignore_mismatched_sizes=True, local_files_only=local_files_only
    )
    batch = tokenizer(["CCO", "CC(=O)O"], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(**batch).logits
    if logits.shape != (2, 6):
        raise RuntimeError(f"Unexpected smoke-test output shape: {tuple(logits.shape)}")
    print(f"Smoke test passed: logits shape={tuple(logits.shape)}, device={next(model.parameters()).device}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FART task adaptation on frozen six-class TasteMM splits.")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifests" / "six_class_split_manifest.csv")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--variant", choices=["unaugmented", "augmented"], default="augmented")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--augment-factor", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--inference-augmentations", type=int, default=10)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    args.folds = validate_folds(args.folds)
    if args.output is None:
        args.output = HERE / "outputs" / f"fart_{args.variant}"
    if args.epochs is None:
        args.epochs = 2.0 if args.variant == "augmented" else 20.0
    RDLogger.DisableLog("rdApp.*")

    if args.smoke_test:
        smoke_test(args.checkpoint, args.revision, args.local_files_only, args.max_length)
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Full five-fold FART fine-tuning is intentionally not started on CPU. "
            "Use --smoke-test locally or run this command on a GPU server."
        )

    manifest = pd.read_csv(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["model"] = MODEL_NAME
    config["method_source_doi"] = "10.1038/s41538-025-00474-z"
    config["manifest"] = str(args.manifest)
    config["output"] = str(args.output)
    config["task_adaptation"] = "FART is retrained with six output classes for the common benchmark."
    config["implementation_boundary"] = "Independent task adaptation; this is not the authors' hosted ensemble."
    config["reproduction_note"] = (
        f"Uses the published ChemBERTa checkpoint; variant={args.variant}, epochs={args.epochs}, "
        f"training augmentation factor={args.augment_factor if args.variant == 'augmented' else 1}, "
        f"batch size={args.batch_size}, weight decay=0.01, and mean probabilities over "
        f"{args.inference_augmentations} randomized test SMILES."
    )
    (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    for fold in args.folds:
        pred_path = args.output / f"fold{fold}_predictions.csv"
        metric_path = args.output / f"fold{fold}_metrics.csv"
        if args.resume and pred_path.exists() and metric_path.exists():
            print(f"[SKIP] fold {fold}")
            continue
        train = manifest[(manifest["fold"] == fold) & (manifest["partition"] == "train")]
        val = manifest[(manifest["fold"] == fold) & (manifest["partition"] == "val")]
        test = manifest[(manifest["fold"] == fold) & (manifest["partition"] == "test")].reset_index(drop=True)
        factor = args.augment_factor if args.variant == "augmented" else 1
        train_frame = model_frame(train, factor, 42 + fold)
        val_frame = model_frame(val, factor, 142 + fold)
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
            args.checkpoint, revision=args.revision, num_labels=6, ignore_mismatched_sizes=True,
            local_files_only=args.local_files_only
        )
        trainer = Trainer(
            model=model,
            args=training_arguments(args.output, args, fold),
            train_dataset=SmilesDataset(train_frame, tokenizer, args.max_length),
            eval_dataset=SmilesDataset(val_frame, tokenizer, args.max_length),
            data_collator=DataCollatorWithPadding(tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        print(f"[RUN] fold={fold} train={len(train_frame)} val={len(val_frame)} test={len(test_frame)}")
        trainer.train(resume_from_checkpoint=bool(args.resume and (args.output / f"fold{fold}_checkpoints").exists()))
        prediction = trainer.predict(SmilesDataset(test_frame, tokenizer, args.max_length))
        logits = np.asarray(prediction.predictions)
        logits = logits - logits.max(axis=1, keepdims=True)
        variant_prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        y_prob, vote_agreement = aggregate_smiles_ensemble(
            variant_prob,
            test_frame["parent_index"].to_numpy(dtype=int),
            len(test),
        )
        save_fold_outputs(args.output, MODEL_NAME, fold, test, y_prob)
        saved_predictions = pd.read_csv(pred_path)
        saved_predictions["fart_smiles_ensemble_size"] = args.inference_augmentations
        saved_predictions["fart_vote_agreement"] = vote_agreement
        saved_predictions["fart_unanimous"] = vote_agreement == 1.0
        saved_predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
        trainer.save_model(args.output / f"fold{fold}_best_model")
        tokenizer.save_pretrained(args.output / f"fold{fold}_best_model")
        print(f"[DONE] fold={fold}")


if __name__ == "__main__":
    main()
