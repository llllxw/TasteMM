"""Offline prediction using the bundled checkpoint and input SMILES."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import BertModel, BertTokenizer

sys.path.insert(0, str(Path(__file__).parent / 'tastemm_engine'))
from features import get_mix_fingerprint, smiles_to_graph
from model import TasteBaselineModel

TASTES = ['bitter', 'sweet', 'umami', 'salty', 'sour', 'tasteless']


class Predictor:
    def __init__(self, bundle):
        self.bundle = Path(bundle)
        config = json.loads((self.bundle / 'manifest.json').read_text())
        self.temperature = float(config.get('temperature', 1.0))
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Invalid model temperature')
        if config['class_names'] != TASTES:
            raise ValueError('Model class order does not match this software')
        self.model = TasteBaselineModel().cpu().eval()
        state = torch.load(self.bundle / 'model.pth', map_location='cpu', weights_only=True)
        # Older training checkpoints stored unused branch gates even when disabled.
        if config.get('use_branch_gates') is not False:
            raise ValueError('This inference build requires a verified ungated model bundle')
        for key in ['graph_gate_logit', 'mixfp_gate_logit', 'bert_gate_logit']:
            state.pop(key, None)
        self.model.load_state_dict(state, strict=True)
        resources = str(self.bundle / 'bert')
        self.tokenizer = BertTokenizer.from_pretrained(resources, local_files_only=True)
        self.bert = BertModel.from_pretrained(resources, local_files_only=True).cpu().eval()

    def predict(self, frame):
        if 'SMILES' not in frame:
            raise ValueError('Input must contain a SMILES column')
        result = frame.reset_index(drop=True).copy()
        result['status'] = 'invalid_smiles'
        result['predicted_taste'] = ''
        for taste in TASTES:
            result['probability_' + taste] = np.nan
        valid = []
        data = []
        for idx, row in result.iterrows():
            smiles = row.SMILES
            if not isinstance(smiles, str) or not smiles.strip():
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                continue
            tokens = self.tokenizer(smiles, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            with torch.no_grad():
                bert = self.bert(**tokens).last_hidden_state[:, 0, :].cpu()
            x, edges, attrs = smiles_to_graph(smiles)
            data.append(Data(x=x, edge_index=edges, edge_attr=attrs,
                             mixfp=torch.tensor(get_mix_fingerprint(smiles)).reshape(1, -1), bert=bert))
            valid.append(idx)
        if data:
            with torch.no_grad():
                logits = torch.cat([self.model(batch, mode='classify') for batch in DataLoader(data, batch_size=32)])
                probs = torch.softmax(logits / self.temperature, dim=1).cpu().numpy()
            for idx, values in zip(valid, probs):
                result.loc[idx, 'status'] = 'ok'
                result.loc[idx, 'predicted_taste'] = TASTES[int(values.argmax())]
                result.loc[idx, ['probability_' + t for t in TASTES]] = values
        return result
