# TasteMM

TasteMM is an interpretable multimodal framework for six-class molecular taste prediction. Given a SMILES string, it combines a GATv2 molecular-graph representation, a 3,239-dimensional mixed fingerprint (MACCS + RDKFingerprint + ECFP4), and a 768-dimensional mean-pooled BERT representation to predict bitter, sweet, umami, salty, sour, or tasteless.

The sequence encoder is pinned to `google-bert/bert-base-uncased` revision `86b5e0934494bd15c9632b12f734a8a67f723594`; cache metadata records the revision, input-order hash, pooling rule, maximum length, and array shape.

This repository contains the released benchmark CSV, preprocessing and split-generation code, model training and inference code, calibration and selective-prediction metrics, interpretation utilities, a same-split model-comparison package, and a CPU-friendly installation test.

## 1. Environment

The intended GPU environment is pinned in `environment.yml` and `requirements.txt`. Python 3.10 is used because several locked package releases are not compatible with Python 3.8.

### Conda installation (recommended)

```bash
conda env create -f environment.yml
conda activate tastemm
python scripts/smoke_test.py
```

### Pip installation with CUDA 12.1

```bash
conda create -n tastemm python=3.10 -y
conda activate tastemm
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter==2.1.2+pt22cu121 torch-sparse==0.6.18+pt22cu121 torch-cluster==1.6.3+pt22cu121 torch-spline-conv==1.2.2+pt22cu121 -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-geometric==2.6.1
pip install -r requirements.txt
python scripts/smoke_test.py
```

The smoke test does not download BERT weights or train a model. It checks molecular graph construction, the 61-dimensional node features, 18-dimensional edge features, 3,239-dimensional mixed fingerprints, 768-dimensional sequence inputs, batched forward passes, metrics, and contrastive loss. A successful run prints `"status": "PASS"`.

For the full validation suite, install the development requirements and run:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests additionally verify both frozen manifests, degenerate contrastive-loss batches, invalid-metric rejection, and strict model-comparison pairing with a deliberate sample-misalignment failure.

## 2. Dataset and preprocessing

The original experiment table is available directly as [`data/compound.csv`](data/compound.csv). It is a UTF-8 CSV with four columns (`ID`, `Name`, `SMILES`, and `Label`) and is readable without Python. The label map and a full data dictionary are provided in [`data/label_mapping.csv`](data/label_mapping.csv) and [`data/README.md`](data/README.md). Dataset provenance is summarized in [`DATA_SOURCES.md`](DATA_SOURCES.md).

| Taste | Label | Rows |
| --- | ---: | ---: |
| Bitter | 1 | 3,005 |
| Sweet | 2 | 2,635 |
| Umami | 3 | 328 |
| Salty | 4 | 28 |
| Sour | 5 | 1,700 |
| Tasteless | 6 | 5,010 |
| Total |  | 12,706 |

Audit the released table:

```bash
python scripts/audit_dataset.py --input data/compound.csv --output-dir data/audit
```

The audit verifies SMILES validity, counts classes and duplicates, canonicalizes structures, and exports cross-label structures. The released final table contains 115 canonical structures associated with more than one label (282 rows). Thus, the model is trained with one target per row, but the repository does not claim that every chemical structure has only one possible taste. The pre-curation master table and exclusion log are not available in this package, so the number of ambiguous source records removed during harmonization cannot be reconstructed from the final CSV.

Build the PyTorch Geometric data for five matched stratified 80/10/10 train/validation/test runs:

```bash
python data_process.py --input_csv data/compound.csv --encoding utf-8
```

For every run, the script creates stratified partitions containing approximately 80% training data, 10% validation data, and 10% test data. It saves `train_pyg.pt`, `val_pyg.pt`, `test_pyg.pt`, and a human-readable `split_manifest.csv` under `data/Multi/processed/fold_<n>/`.

Supervised contrastive pretraining, graph-branch warm-up, and end-to-end fine-tuning use only the training partition of each run. The validation partition is used for early stopping, model selection, and temperature scaling; the test partition is used only for final evaluation.

## 3. Checkpoints and outputs

Pretrained weights are not bundled in this repository. Running the training commands below creates one authoritative checkpoint under `runs/<run_name>/checkpoints/`. Each completed run also contains:

- `result.json`: configuration, checkpoint path, metrics, calibration results, and training histories;
- `artifacts/test_logits.npy`, `test_probs_raw.npy`, `test_labels.npy`, and `test_row_indices.npy`: raw test outputs and auditable source-row alignment;
- temperature-scaled probabilities and confidence arrays used for reliability analyses.
- fused and graph test embeddings used by the public embedding-export workflow.

Checkpoint paths stored in `result.json` are relative to the run directory, so a complete run directory can be moved without editing its metadata.

## 4. Training, evaluation, and prediction

Run one matched split:

```bash
python train.py --fold 0 --seed 42 --out_dir runs
```

Run all five matched splits and export mean and standard deviation:

```bash
python train_5fold.py --folds 0,1,2,3,4 --seed 42 --out_dir runs --summary_csv results/tastemm_5fold_summary.csv
```

To migrate an older completed run to the current auditable artifact schema without retraining, regenerate the matching processed data and re-evaluate its existing checkpoint:

```bash
python reevaluate_run.py --run-dir runs/fold0_seed42_baseline --processed-dir data/Multi/processed
```

Repeat for split indices 0–4. This adds source-row indices, raw probabilities, embeddings, a prediction CSV, a relative checkpoint path, and a checkpoint hash.

Predict new compounds from a CSV containing `ID`, `Name`, and `SMILES`:

```bash
python predict.py --run_dir runs/fold0_seed42_baseline --input_csv examples/prediction_input.csv --output_csv results/predictions/example.csv
```

The evaluation code reports accuracy, macro/weighted F1, class-wise and macro AUROC, and confidence metrics. ECE measures the gap between predicted confidence and observed accuracy; the multiclass Brier score measures squared probability error; AURC summarizes error as low-confidence predictions are rejected; and EAURC is the excess AURC above the theoretically optimal ranking. Lower ECE, Brier, AURC, and EAURC indicate better reliability.

## 5. Same-split model comparisons

[`benchmarks/model_comparison/`](benchmarks/model_comparison/) contains the common-split comparison framework for TasteMM, TasteMolNet, Virtuous MultiTaste, and FART. The three published comparators are independent, paper-informed task adaptations implemented for the frozen TasteMM partitions; they are not claimed to be byte-for-byte executions of the original authors' repositories. The framework stores per-compound probabilities, validates sample-level pairing before analysis, summarizes the five run scores as mean ± SD, computes one-vs-rest taste metrics, and performs split-blocked ANOVA and Tukey HSD comparisons.

Six-class adaptations and original-label-range scope-matched experiments are kept separate. Model names use their original published forms, while each run configuration records the change in output space. Code and frozen manifests are included for TasteMolNet's three-class, Virtuous MultiTaste's four-class, and FART's five-class scopes. No final comparison tables are currently bundled; verified GPU outputs and their strict import audits must be completed before final manuscript reporting.

## 6. Interpretability and visualization

```bash
python export_embedding_table.py --run-root runs --source-csv data/compound.csv --output-dir results/embeddings
python plot_embedding_umap_tsne.py --input_dir results/embeddings
python plot_embedding_umap_tsne.py --help
python gradient_based_attribution.py --help
python plot_gradient_based_attribution.py --help
```

`gradient_based_attribution.py` supports Integrated Gradients and gradient-times-input atom attribution. `plot_gradient_based_attribution.py` renders the exported atom table without relying on an unpublished helper module.

## 7. Repository layout

```text
TasteMM/
├── data/                         # released CSV, label map, and audit outputs
├── scripts/                      # dataset audit and installation smoke test
├── tests/                        # manifest, metric, loss, and pairing validation
├── benchmarks/model_comparison/  # same-split comparisons and statistical tests
├── data_process.py               # features, validation, splits, and manifests
├── model.py                      # multimodal GATv2/fingerprint/BERT model
├── train.py                      # one-split training and evaluation
├── train_5fold.py                # five matched-run driver and mean/SD summary
├── reevaluate_run.py             # migrate existing checkpoints without retraining
├── predict.py                    # inference on new SMILES
├── export_embedding_table.py     # auditable embedding-table export
├── calibration_metrics.py        # ECE, Brier, AURC, and EAURC
└── gradient_based_attribution.py # atom-level explanations
```

## 8. Citation

If you use the dataset, model, or benchmark code, please cite the TasteMM manuscript and the original databases listed in `DATA_SOURCES.md`. The final journal citation and DOI should be added here after publication.
