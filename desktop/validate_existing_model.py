"""Compare desktop checkpoint inference with archived reference logits."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from engine import Predictor
parser = argparse.ArgumentParser()
parser.add_argument('--bundle', type=Path, required=True)
parser.add_argument('--run-dir', type=Path, required=True)
parser.add_argument('--test-pyg', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
predictor = Predictor(args.bundle)
samples = torch.load(args.test_pyg, map_location='cpu', weights_only=False)[:8]
for sample in samples:
    sample.mixfp = sample.mixfp.reshape(1, -1)
    sample.bert = sample.bert.reshape(1, -1)
with torch.no_grad():
    actual = predictor.model(next(iter(DataLoader(samples, batch_size=8))), mode='classify').numpy()
expected = np.load(args.run_dir / 'artifacts/test_logits.npy')[:8]
error = float(np.max(np.abs(actual - expected)))
if not np.allclose(actual, expected, atol=2e-05, rtol=2e-05):
    raise AssertionError(f'Archived logits do not match: max error {error}')
frame = pd.DataFrame({'SMILES': ['CCO', 'not a smiles', '', '[Na+].[Cl-]']})
result = predictor.predict(frame)
assert result.status.tolist() == ['ok', 'invalid_smiles', 'invalid_smiles', 'ok']
probs = result.loc[result.status == 'ok', ['probability_' + t for t in ['bitter', 'sweet', 'umami', 'salty', 'sour', 'tasteless']]].values.astype(float)
assert np.allclose(probs.sum(axis=1), 1)
summary = {'archived_test_samples': 8, 'archived_logits_max_absolute_error': error, 'archived_logits_match': True, 'invalid_input_rows_preserved': True, 'six_probabilities_sum_to_one': True, 'device': 'CPU', 'platform_validation': 'Checkpoint inference and reference-input regression'}
args.output.write_text(json.dumps(summary, indent=2))
result.to_csv(args.output.with_suffix('.csv'), index=False)
print(json.dumps(summary, indent=2))
