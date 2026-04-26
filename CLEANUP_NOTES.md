# Cleanup Notes

The GitHub package was reduced to the final baseline only.

## Removed components

1. `GINE`
- Why: not part of the final baseline and underperformed the retained GATv2 baseline.

2. External classifier heads (`RF`, `SVM`, `KNN`, `XGBoost`)
- Why: ablation-only branches, not part of the deployed model, and they introduced extra dependencies and branching logic.

3. Residual logits branch
- Why: experimental path only, not retained in the final model.

4. Static branch gates
- Why: final baseline is `no_gate`.

5. Legacy checkpoint save/load fallbacks
- Why: they increased ambiguity during inference and were only kept for backward compatibility with older experiments.

## Files simplified

- `model.py`
  - reduced to one model: `TasteBaselineModel`
  - removed GINE, gate logic, external heads, residual logits

- `train.py`
  - reduced to baseline-only training path
  - removed branch toggles and external-classifier flow
  - removed legacy artifact save logic

- `train_5fold.py`
  - reduced to baseline-only CV driver

- `predict.py`
  - now requires `run_dir`
  - removed run-name guessing logic and external-classifier loading

- `tools.py`
  - changed to local relative paths for GitHub portability
