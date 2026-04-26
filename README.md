# TasteMM

## Introduction

TasteMM is an interpretable multimodal deep learning framework for six-class molecular taste prediction. The model takes a SMILES string as input and integrates three complementary molecular representations: a molecular graph, a mixed structural fingerprint, and a SMILES-BERT sequence embedding.

The graph branch encodes atom-bond topology with a GATv2 encoder, the fingerprint branch uses concatenated MACCS, RDKFingerprint, and ECFP4 fingerprints, and the sequence branch uses mean-pooled final hidden states from a pretrained BERT model. The three modality-specific embeddings are fused for prediction of bitter, sweet, umami, salty, sour, and tasteless classes. The training pipeline includes supervised contrastive pretraining, graph-branch warmup, end-to-end fine-tuning, validation-based temperature scaling, and confidence-based selective prediction analysis.

## Environment Requirement

The code has been tested running under Python 3.8.20. The required packages are as follows (adjust versions to match your environment):

```text
pytorch
torch-geometric
numpy
pandas
scikit-learn
transformers
rdkit
tqdm
matplotlib
Pillow
```

Optional packages for visualization:

```text
umap-learn
```

## Source codes

`data_process.py`: data preprocessing script for building stratified five-fold PyTorch Geometric datasets. It converts SMILES strings into molecular graphs, generates mixed fingerprints using MACCS, RDKFingerprint, and ECFP4, extracts mean-pooled SMILES-BERT embeddings, validates invalid SMILES, and saves train/validation/test splits.

`model.py`: TasteMM main model, including the GATv2 graph encoder, fingerprint encoder, SMILES-BERT embedding encoder, multimodal fusion layer, contrastive projection head, main classification head, and auxiliary graph classification head.

`loss.py`: supervised contrastive loss used during contrastive pretraining.

`metric.py`: classification metric utilities, including accuracy, weighted F1, macro F1, class-wise F1, confusion matrix, class-wise AUROC, macro AUROC, and weighted AUROC.

`calibration_metrics.py`: calibration and selective prediction utilities, including ECE, multiclass Brier score, reliability bins, risk-coverage curves, AURC, EAURC, and class-wise selective prediction metrics.

`confidence_methods.py`: confidence utility functions for softmax probabilities, margin-based confidence, and temperature scaling.

`train.py`: single-fold training script. It performs optional contrastive pretraining, graph-only warmup, end-to-end fine-tuning, weighted-F1-based early stopping, validation-set temperature scaling, test-set evaluation, calibration analysis, and selective prediction analysis.

`train_5fold.py`: five-fold cross-validation driver. It runs `train.py` across folds, collects per-fold results, and exports mean and standard deviation summaries.

`predict.py`: inference script for new compounds. It preprocesses SMILES inputs using the same graph, fingerprint, and SMILES-BERT pipelines as training, loads a trained fold model, applies temperature scaling, and exports predicted labels, probabilities, margins, and high-confidence flags.

`plot_embedding_umap_tsne.py`: visualization script for learned molecular representations. It projects fused or graph embeddings into two dimensions using UMAP or t-SNE and exports plots colored by class, correctness, or confidence.

`gradient_based_attribution.py`: gradient-based interpretability script for representative molecules. It supports Integrated Gradients and gradient-times-input attribution, aggregates feature attributions into atom-level scores, and exports attribution tables and molecule renderings.

`plot_gradient_based_attribution.py`: plotting script for assembling atom-level attribution visualizations generated from gradient-based explanations.

`tools.py`: shared configuration file for project paths, model hyperparameters, training constants, directory creation, and random seed initialization.

`CLEANUP_NOTES.md`: notes describing the simplified final baseline package and removed experimental branches.

## Typical commands

Build processed data:

```bash
python data_process.py --input_csv /path/to/compound_info.csv --encoding utf-8
```

Run five-fold training:

```bash
python train_5fold.py --out_dir runs --summary_csv results/baseline_5fold_summary.csv
```

Run prediction:

```bash
python predict.py --run_dir runs/fold0_seed42_baseline_gatv2_rich_pre1_gaux0p2_gwarm30 --input_csv /path/to/new_data.csv
```
