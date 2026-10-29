# TasteMM

TasteMM is an interpretable multimodal framework for six-class molecular taste prediction. Given a SMILES string, it combines a GATv2 molecular-graph representation, a 3,239-dimensional mixed fingerprint (MACCS + RDKFingerprint + ECFP4), and a 768-dimensional mean-pooled BERT representation to predict bitter, sweet, umami, salty, sour, or tasteless.

The sequence encoder is pinned to `google-bert/bert-base-uncased` revision `86b5e0934494bd15c9632b12f734a8a67f723594`; cache metadata records the revision, input-order hash, pooling rule, maximum length, and array shape.

This repository contains the released benchmark CSV, preprocessing and split-generation code, model training and inference code, calibration and selective-prediction metrics, interpretation utilities, matched molecular pair (MMP) analysis, a same-split model-comparison package, and a CPU-friendly installation test.

## Desktop application

[TasteMM Desktop v0.1.0: downloads, matching source archives, model configuration and checksums](https://github.com/llllxw/TasteMM/releases/tag/desktop-v0.1.0).

| Platform | Package | Start |
| --- | --- | --- |
| Linux x86_64 | [TasteMM-Linux-x86_64.tar.gz](https://github.com/llllxw/TasteMM/releases/download/desktop-v0.1.0/TasteMM-Linux-x86_64.tar.gz) | `./TasteMM/TasteMM` |
| Windows x64 | [TasteMM-Windows-amd64.zip](https://github.com/llllxw/TasteMM/releases/download/desktop-v0.1.0/TasteMM-Windows-amd64.zip) | `TasteMM/TasteMM.exe` |
| macOS Apple Silicon | [TasteMM-macOS-arm64.zip](https://github.com/llllxw/TasteMM/releases/download/desktop-v0.1.0/TasteMM-macOS-arm64.zip) | `TasteMM.app` |

Extract the complete archive and keep its contents together. The packages include the CPU runtime and model resources for offline prediction. The interface supports single-SMILES prediction, batch CSV input, molecular structure display, six-class probabilities and CSV export. Use the platform-specific source and configuration attached to the versioned release to reproduce its predictions.

## 1. Environment

The intended GPU environment is pinned in `environment.yml` and `requirements.txt`.

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

The tests verify all four task manifests, degenerate contrastive-loss batches, invalid-metric rejection, and sample alignment in model comparisons.

## 2. Dataset and preprocessing

The current curated table is available directly as [`data/compound.csv`](data/compound.csv). It is a UTF-8 CSV with four columns (`ID`, `Name`, `SMILES`, and `Label`) and is readable without Python. `ID` is an optional source/curation identifier, `Name` is the compound name, `SMILES` is the molecular representation, and `Label` is the integer taste class defined in [`data/label_mapping.csv`](data/label_mapping.csv). Use zero-based row indices as record identifiers; neither `ID` nor `SMILES` is a unique key.

The table contains 12,706 records. Its SHA-256 is `4b3fc2b850254cd1769a807941629b55e512da4bb7e09691d7ebd8a7dd68871b`. 

The multilabel dataset is available as [`data/multilabel_dataset.csv`](data/multilabel_dataset.csv) and contains 147 records with compound identifiers, names, SMILES, taste annotations, and source metadata. `Labels` lists semicolon-separated class IDs using the same taste-label definitions; the six `label_*` columns encode label membership as 0 or 1. These records are provided separately from the 12,706-record single-label benchmark.

Data sources include the taste databases and literature cited in the TasteMM manuscript, ChemTastesDB, PubChem/HSDB, TastePepMap, sensory publications and patent sensory tables. Labels represent the collected source annotations. Consult the original sources for attribution and terms of use.

Morgan fingerprints in `data_process.py` use `includeChirality=True` for both the default radius-2 generator and the custom-parameter branch. The default mixed fingerprint is 3,239 bits (167 MACCS + 1,024 RDK + 2,048 Morgan).

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

The audit verifies SMILES validity, counts classes and duplicates, canonicalizes structures, and exports cross-label structures. 

Build the PyTorch Geometric data for five matched stratified 80/10/10 train/validation/test runs:

```bash
python benchmarks/model_comparison/prepare_benchmark.py
python benchmarks/model_comparison/prepare_scope_matched.py
python data_process.py --input_csv data/compound.csv --encoding utf-8 --split_manifest benchmarks/model_comparison/manifests/six_class_split_manifest.csv
```

For every run, the script creates stratified partitions containing approximately 80% training data, 10% validation data, and 10% test data. It saves `train_pyg.pt`, `val_pyg.pt`, `test_pyg.pt`, and a human-readable `split_manifest.csv` under `data/Multi/processed/fold_<n>/`.

Splitting uses rows as the sampling unit with seed 42; identical structures are not grouped. The manifest audit records train/test canonical-structure overlap. Keep each checkpoint with its matching source CSV, feature configuration and split manifest.

Supervised contrastive pretraining, graph-branch warm-up, and end-to-end fine-tuning use only the training partition of each run. The validation partition is used for early stopping, model selection, and temperature scaling; the test partition is used only for final evaluation.

## 3. Checkpoints and outputs

Training writes a checkpoint under `runs/<run_name>/checkpoints/`. Each completed run also contains:

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

Re-evaluate a saved checkpoint using its matching processed data:

```bash
python reevaluate_run.py --run-dir runs/fold0_seed42_baseline --processed-dir data/Multi/processed
```

The command exports source-row indices, probabilities, embeddings, predictions and the checkpoint hash. It evaluates the saved weights without updating them.

Predict new compounds from a CSV containing `ID`, `Name`, and `SMILES`:

```bash
python predict.py --run_dir runs/fold0_seed42_baseline --input_csv examples/prediction_input.csv --output_csv results/predictions/example.csv
```

The evaluation code reports accuracy, macro/weighted F1, class-wise and macro AUROC, and confidence metrics. ECE measures the gap between predicted confidence and observed accuracy; the multiclass Brier score measures squared probability error; AURC summarizes error as low-confidence predictions are rejected; and EAURC is the excess AURC above the theoretically optimal ranking. Lower ECE, Brier, AURC, and EAURC indicate better reliability.

## 5. Same-split model comparisons

[`benchmarks/model_comparison/`](benchmarks/model_comparison/) contains the common-split comparison framework for TasteMM, TasteMolNet, Virtuous MultiTaste, and FART. The three published comparators are independent, paper-informed task adaptations implemented for the frozen TasteMM partitions. The framework stores per-compound probabilities, validates sample-level pairing before analysis, summarizes the five run scores as mean ± SD, computes one-vs-rest taste metrics, and performs split-blocked ANOVA and Tukey HSD comparisons.

Six-class adaptations and original-label-range scope-matched experiments are kept separate. Model names use their original published forms, while each run configuration records the change in output space. Code and frozen manifests are included for TasteMolNet's three-class, Virtuous MultiTaste's four-class, and FART's five-class scopes. 

```bash
pip install -r requirements-benchmark.txt
python benchmarks/model_comparison/run_tastemolnet_sixclass.py
python benchmarks/model_comparison/run_virtuous_multitaste_sixclass.py
python benchmarks/model_comparison/run_fart_sixclass.py
python benchmarks/model_comparison/import_tastemm_results.py --run-root runs
python benchmarks/model_comparison/analyze_benchmark.py
```

For scope-matched runs, use the corresponding scope CSV, manifest, class count and separate output directories:

| Task | Manifest/CSV prefix | Classes | Comparator script |
| --- | --- | ---: | --- |
| `scope3` | `tastemolnet_scope3` | 3 | `run_tastemolnet_scope3.py` |
| `scope4` | `virtuous_scope4` | 4 | `run_virtuous_multitaste_scope4.py` |
| `scope5` | `fart_scope5` | 5 | `run_fart_scope5.py` |

For example, run the three-class task with:

```bash
python data_process.py --input_csv benchmarks/model_comparison/scope_matched/manifests/tastemolnet_scope3_compound.csv --processed_dir data/Multi/processed_scope3 --num_classes 3 --split_manifest benchmarks/model_comparison/scope_matched/manifests/tastemolnet_scope3_split_manifest.csv
python train_5fold.py --processed_dir data/Multi/processed_scope3 --num_classes 3 --out_dir runs_scope3 --summary_csv results/tastemm_scope3_summary.csv
python benchmarks/model_comparison/run_tastemolnet_scope3.py
python benchmarks/model_comparison/import_scope_tastemm_results.py --task scope3 --run-root runs_scope3
python benchmarks/model_comparison/analyze_scope_matched.py --task scope3
```

## 6. Interpretability and visualization

```bash
python export_embedding_table.py --run-root runs --source-csv data/compound.csv --output-dir results/embeddings
python plot_embedding_umap_tsne.py --input_dir results/embeddings
python plot_embedding_umap_tsne.py --help
python gradient_based_attribution.py --help
python plot_gradient_based_attribution.py --help
```

`gradient_based_attribution.py` supports Integrated Gradients and gradient-times-input atom attribution. `plot_gradient_based_attribution.py` renders the exported atom table.

## 7. Matched molecular pair analysis

[`mmp_analysis.py`](mmp_analysis.py) reproduces the matched molecular pair analysis used to connect local structural edits with changes in TasteMM predictions and atom-level attributions. It identifies taste-switching MMPs among compounds held out from model training, matches each pair to a same-taste non-switching control, and computes contrastive Integrated Gradients with the corresponding trained split model. MMP construction uses Morgan fingerprints (radius 2, 2,048 bits), a minimum Tanimoto similarity of 0.60, a maximum heavy-atom-count difference of five, and a maximum of five changed heavy atoms per molecule after maximum-common-substructure matching.

Stage A requires the processed PyTorch Geometric data and the five completed TasteMM run directories. Each run directory must contain `result.json` and its referenced classification checkpoint. A GPU is recommended for the attribution calculations, although `--device cpu` is supported.

```bash
python mmp_analysis.py --stage A \
  --input_csv data/compound.csv \
  --split_manifest benchmarks/model_comparison/manifests/six_class_split_manifest.csv \
  --run_root runs \
  --processed_dir data/Multi/processed \
  --output_dir results/mmp
```

Stage B uses the Stage A tables to calculate paired bootstrap confidence intervals, paired permutation tests, paired effect sizes, and Spearman attribution–prediction consistency statistics, and then creates the analysis panels.

```bash
python mmp_analysis.py --stage B --output_dir results/mmp
```

Both stages can be run consecutively with `--stage AB` and the same path arguments used for Stage A. The principal auditable outputs are:

- `stage_A_mmp_pairs.csv`: taste-switching MMPs and their matched non-switching controls;
- `stage_A_contrastive_ig_atoms.csv` and `stage_A_ig_task_summary.csv`: atom-level and molecule-level contrastive Integrated Gradients;
- `stage_A_panel_representatives.csv`: identifiers for the prespecified representative examples;
- `stage_B_validation_statistics.csv`: paired group comparison, bootstrap interval, permutation test, and effect size;
- `stage_B_attribution_prediction_consistency.csv` and `stage_B_consistency_statistics.csv`: attribution–prediction consistency data and statistics;
- `PanelA`, `PanelB`, `PanelC`, and `Fig_structural_optimization_ABC` in PNG and SVG formats.

The random seed, MMP thresholds, 32 Integrated Gradients steps, 5,000 bootstrap resamples, and 20,000 permutation repetitions are defined at the top of the script. Stage A caches the pre-attribution MMP table as `stage_A_mmp_pairs_pre_ig.csv`; pass `--rebuild_mmp` after changing the input data or MMP criteria.

## 8. Repository layout

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
├── reevaluate_run.py             # evaluate saved checkpoints and export artifacts
├── predict.py                    # inference on new SMILES
├── export_embedding_table.py     # auditable embedding-table export
├── calibration_metrics.py        # ECE, Brier, AURC, and EAURC
├── gradient_based_attribution.py # atom-level explanations
└── mmp_analysis.py               # matched pairs, contrastive IG, statistics, and figures
```

## 9. Citation

If you use the dataset, model, or benchmark code, please cite the TasteMM manuscript and the original data sources cited there.
