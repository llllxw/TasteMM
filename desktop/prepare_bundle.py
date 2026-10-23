"""Copy an existing checkpoint and local BERT snapshot into a portable bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', type=Path, required=True)
parser.add_argument('--bert-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit('Output already exists; choose a new directory.')
payload = json.loads((args.run_dir / 'result.json').read_text())
config = payload['config']
if (config.get('use_branch_gates', True) or config.get('gnn_type') != 'gatv2'
        or config.get('cls_head_type') != 'linear' or config.get('fuse_graph_logits_residual', True)
        or not all(config.get(k) for k in ['use_graph', 'use_mixfp', 'use_bert'])):
    raise SystemExit('Unsupported model architecture; bundle was not created.')
weights = Path(payload['checkpoints']['best_classify_model'])
if not weights.exists():
    weights = args.run_dir / 'checkpoints' / weights.name
if not weights.exists():
    raise SystemExit('Checkpoint not found.')
if not (args.bert_dir / 'config.json').exists():
    raise SystemExit('Local BERT config.json not found.')
args.output.mkdir(parents=True)
shutil.copy2(weights, args.output / 'model.pth')
shutil.copytree(args.bert_dir, args.output / 'bert', symlinks=False)
temperature = payload.get('temperature_T', 1.0)
manifest = {'class_names': ['bitter', 'sweet', 'umami', 'salty', 'sour', 'tasteless'],
            'use_branch_gates': False,
            'temperature': temperature, 'fold': payload['config']['fold'],
            'checkpoint_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
            'input_policy': 'Original SMILES, no salt removal, tautomer conversion or stereochemistry enumeration',
            'training': 'Existing checkpoint; no retraining performed'}
(args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(args.output)
