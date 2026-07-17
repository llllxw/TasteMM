# Same-split model-comparison package

This directory implements the reviewer-facing comparison protocol on the released 12,706-row dataset. Every model receives the same split-specific training, validation, and test rows. The split ID is retained as the pairing/blocking unit for statistical analysis.

## Included experiments

- `run_tastemolnet_sixclass.py`: independent TasteMolNet MACCS-XGBoost task adaptation;
- `run_virtuous_multitaste_sixclass.py`: independent Virtuous MultiTaste 15-descriptor random-forest task adaptation;
- `run_fart_sixclass.py`: independent FART/ChemBERTa task adaptation with SMILES augmentation and ensemble inference (CUDA required for full training);
- `prepare_scope_matched.py`: frozen manifests for the three-, four-, and five-class scope-matched tasks;
- `run_tastemolnet_scope3.py`, `run_virtuous_multitaste_scope4.py`, and `run_fart_scope5.py`: comparator-side scope-matched runs;
- `import_scope_tastemm_results.py` and `analyze_scope_matched.py`: strict TasteMM import and two-model paired inference for each scope;
- `analyze_benchmark.py`: mean/SD aggregation, one-vs-rest taste metrics, split-blocked ANOVA, paired effects, and Tukey HSD comparisons.

Model names in outputs use their original published names. Each configuration file separately records that the method was retrained with six outputs for the common benchmark. Six-class results and original-label-range scope-matched results must be reported in separate tables.

These scripts are paper-informed reimplementations for a new common task, not claims that the original authors' complete training repositories were executed unchanged. Important deviations are explicit: the output heads are changed to six classes; Virtuous MultiTaste uses the reported 15-descriptor family with split-local preprocessing and random oversampling; and FART aggregates randomized-SMILES predictions by mean probability. Cite the original method papers and repositories in the manuscript, and describe these implementations as task adaptations.

FART uses the pinned Hugging Face checkpoint `seyonec/SMILES_tokenized_PubChem_shard00_160k` at revision `f0854db6cbaad4655ce3bb0c073b9ba0199f4a7d`. Override `--revision` only when intentionally running a separately documented sensitivity experiment.

## Method sources and implementation boundary

- TasteMolNet: [Shi et al., Journal of Food Composition and Analysis (2026), DOI 10.1016/j.jfca.2026.108888](https://doi.org/10.1016/j.jfca.2026.108888). The public adaptation uses the reported MACCS-XGBoost model family with fixed, recorded hyperparameters.
- Virtuous MultiTaste: [Pallante et al., npj Science of Food (2024), DOI 10.1038/s41538-024-00287-6](https://doi.org/10.1038/s41538-024-00287-6) and the [authors' repository](https://github.com/lorenzopallante/VirtuousMultiTaste). The adaptation uses the published 15 Mordred descriptors, 95-tree random forest, training-only kNN imputation/scaling, and explicit random oversampling; it does not rerun the proprietary InSyBio evolutionary search.
- FART: [Zimmermann et al., npj Science of Food (2025), DOI 10.1038/s41538-025-00474-z](https://doi.org/10.1038/s41538-025-00474-z) and the [FART model collection](https://huggingface.co/collections/FartLabs/fart-67558c6e24f6715cc81c07c6). The adaptation independently fine-tunes the pinned ChemBERTa checkpoint and uses mean-probability randomized-SMILES inference; it is not the authors' hosted ensemble.

Accordingly, manuscript wording should say “independent implementation of the published model specification adapted to the frozen task,” not “the original web service was rerun.”

## Reproduce the common split

From the repository root:

```bash
pip install -r requirements-benchmark.txt
python benchmarks/model_comparison/prepare_benchmark.py
python benchmarks/model_comparison/prepare_scope_matched.py
```

Run the comparison-model adaptations:

```bash
python benchmarks/model_comparison/run_tastemolnet_sixclass.py
python benchmarks/model_comparison/run_virtuous_multitaste_sixclass.py
python benchmarks/model_comparison/run_fart_sixclass.py --resume
```

Import the already trained TasteMM runs and analyze the complete four-model set. The analyzer deliberately stops if any model, split, sample, label, or probability row is missing or misaligned:

```bash
python benchmarks/model_comparison/import_tastemm_results.py --run-root /path/to/TasteMM/runs
python benchmarks/model_comparison/analyze_benchmark.py
```

If the TasteMM runs predate `test_row_indices.npy`, first run `reevaluate_run.py` once per split using the matching processed directory. This reuses the checkpoint and does not retrain the model.

Reproduce the three scope-matched comparisons without overwriting the six-class processed data. Each TasteMM task uses its own processed and run directory:

```bash
python data_process.py --input_csv benchmarks/model_comparison/scope_matched/manifests/tastemolnet_scope3_compound.csv --processed_dir data/Multi/processed_scope3 --num_classes 3 --split_manifest benchmarks/model_comparison/scope_matched/manifests/tastemolnet_scope3_split_manifest.csv
python train_5fold.py --processed_dir data/Multi/processed_scope3 --num_classes 3 --out_dir runs_scope3 --summary_csv results/tastemm_scope3_summary.csv
python benchmarks/model_comparison/run_tastemolnet_scope3.py
python benchmarks/model_comparison/import_scope3_tastemm_results.py --run-root runs_scope3
python benchmarks/model_comparison/analyze_scope3.py

python data_process.py --input_csv benchmarks/model_comparison/scope_matched/manifests/virtuous_scope4_compound.csv --processed_dir data/Multi/processed_scope4 --num_classes 4 --split_manifest benchmarks/model_comparison/scope_matched/manifests/virtuous_scope4_split_manifest.csv
python train_5fold.py --processed_dir data/Multi/processed_scope4 --num_classes 4 --out_dir runs_scope4 --summary_csv results/tastemm_scope4_summary.csv
python benchmarks/model_comparison/run_virtuous_multitaste_scope4.py
python benchmarks/model_comparison/import_scope_tastemm_results.py --task scope4 --run-root runs_scope4
python benchmarks/model_comparison/analyze_scope_matched.py --task scope4

python data_process.py --input_csv benchmarks/model_comparison/scope_matched/manifests/fart_scope5_compound.csv --processed_dir data/Multi/processed_scope5 --num_classes 5 --split_manifest benchmarks/model_comparison/scope_matched/manifests/fart_scope5_split_manifest.csv
python train_5fold.py --processed_dir data/Multi/processed_scope5 --num_classes 5 --out_dir runs_scope5 --summary_csv results/tastemm_scope5_summary.csv
python benchmarks/model_comparison/run_fart_scope5.py --resume
python benchmarks/model_comparison/import_scope_tastemm_results.py --task scope5 --run-root runs_scope5
python benchmarks/model_comparison/analyze_scope_matched.py --task scope5
```

## Statistical interpretation

The primary inferential analysis treats the five matched split scores as blocked observations. It reports the overall model effect, Tukey-adjusted pairwise comparisons, paired mean differences with confidence intervals, and one-vs-rest analyses. With only five splits, statistical power is limited; effect sizes and confidence intervals should accompany adjusted p-values. Salty-taste inference is exploratory because the full dataset contains only 28 salty rows.

No final comparison tables are bundled at present. The previously generated local tables were removed because they combined available CPU baselines with existing TasteMM outputs but did not contain a complete, verified set of GPU-server results. Add result tables only after every model in a comparison has predictions for the same five frozen test splits.

The code covers all planned unified and scope-matched experiment definitions. Final tables still require verified GPU outputs for FART and the corresponding TasteMM scope runs. Outputs produced from older datasets or independent train/test splits must not be mixed with this matched-run analysis.
