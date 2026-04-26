import argparse
import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from transformers import BertModel, BertTokenizer

from tools import PROCESSED_DIR, ensure_dirs, setup_seed


setup_seed(42)

BERT_MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MACCS_BITS = 167
RDK_BITS = 1024
ECFP_BITS = 2048
MIXFP_DIM = MACCS_BITS + RDK_BITS + ECFP_BITS

ATOM_SYMBOLS = [
    "C", "H", "O", "N", "S", "P", "Cl", "Br", "F", "I",
    "B", "Si", "Se", "Na", "K", "Li", "Ca", "Fe", "Al",
]
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]
BOND_DIRS = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
    Chem.rdchem.BondDir.EITHERDOUBLE,
    Chem.rdchem.BondDir.UNKNOWN,
]

NODE_FEATURE_DIM = (
    len(ATOM_SYMBOLS) + 1
    + 6
    + 6
    + len(HYBRIDIZATION_TYPES) + 1
    + 1
    + 1
    + len(CHIRAL_TAGS)
    + 5
    + 7
    + 3
    + 1
    + 1
)
EDGE_FEATURE_DIM = (
    len(BOND_TYPES) + 1
    + 1
    + 1
    + len(BOND_STEREOS)
    + len(BOND_DIRS)
)


def one_hot_with_unknown(value, candidates):
    feat = [0] * (len(candidates) + 1)
    try:
        feat[candidates.index(value)] = 1
    except ValueError:
        feat[-1] = 1
    return feat


def one_hot_bucket(value, upper_bound):
    feat = [0] * (upper_bound + 1)
    feat[min(max(int(value), 0), upper_bound)] = 1
    return feat


def bucket_formal_charge(charge):
    buckets = [-2, -1, 0, 1, 2]
    feat = [0] * (len(buckets) + 1)
    try:
        feat[buckets.index(int(charge))] = 1
    except ValueError:
        feat[-1] = 1
    return feat


def mol_from_smiles(smiles):
    smiles_str = str(smiles).strip()
    if not smiles_str or smiles_str.lower() == "nan":
        raise ValueError("Empty SMILES string.")
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles_str!r}")
    return mol


def validate_smiles_list(smiles_list, ids=None):
    invalid = []
    for idx, smiles in enumerate(smiles_list):
        try:
            mol_from_smiles(smiles)
        except ValueError as exc:
            sample_id = ids[idx] if ids is not None else idx
            invalid.append(f"row={idx}, id={sample_id}, {exc}")
    if invalid:
        preview = "\n".join(invalid[:20])
        suffix = "" if len(invalid) <= 20 else f"\n... {len(invalid) - 20} more invalid rows"
        raise ValueError(f"Invalid SMILES found. Please fix the input data before processing:\n{preview}{suffix}")


def build_atom_features(atom):
    return (
        one_hot_with_unknown(atom.GetSymbol(), ATOM_SYMBOLS)
        + one_hot_bucket(atom.GetDegree(), 5)
        + bucket_formal_charge(atom.GetFormalCharge())
        + one_hot_with_unknown(atom.GetHybridization(), HYBRIDIZATION_TYPES)
        + [float(atom.GetIsAromatic())]
        + [float(atom.IsInRing())]
        + [1 if atom.GetChiralTag() == tag else 0 for tag in CHIRAL_TAGS]
        + one_hot_bucket(atom.GetTotalNumHs(), 4)
        + one_hot_bucket(atom.GetTotalValence(), 6)
        + one_hot_bucket(atom.GetNumRadicalElectrons(), 2)
        + [atom.GetMass() / 200.0]
        + [atom.GetIsotope() / 200.0]
    )


def build_bond_features(bond):
    return (
        one_hot_with_unknown(bond.GetBondType(), BOND_TYPES)
        + [float(bond.GetIsConjugated())]
        + [float(bond.IsInRing())]
        + [1 if bond.GetStereo() == stereo else 0 for stereo in BOND_STEREOS]
        + [1 if bond.GetBondDir() == bond_dir else 0 for bond_dir in BOND_DIRS]
    )


def smiles_to_graph(smiles):
    mol = mol_from_smiles(smiles)

    atom_features = [build_atom_features(atom) for atom in mol.GetAtoms()]
    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        edges.extend([[src, dst], [dst, src]])
        bond_feat = build_bond_features(bond)
        edge_features.extend([bond_feat, bond_feat])

    x = (
        torch.tensor(atom_features, dtype=torch.float32)
        if atom_features
        else torch.tensor([[0] * NODE_FEATURE_DIM], dtype=torch.float32)
    )
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.tensor([[0], [0]], dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(edge_features, dtype=torch.float32)
        if edge_features
        else torch.tensor([[0] * EDGE_FEATURE_DIM], dtype=torch.float32)
    )
    return x, edge_index, edge_attr


def get_mix_fingerprint(smiles, radius=2, ecfp_bits=ECFP_BITS, rdk_bits=RDK_BITS):
    mol = mol_from_smiles(smiles)

    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    rdk = np.array(Chem.RDKFingerprint(mol, fpSize=rdk_bits), dtype=np.float32)
    ecfp4 = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=ecfp_bits), dtype=np.float32)
    mixfp = np.concatenate([maccs, rdk, ecfp4])
    if mixfp.shape[0] != MIXFP_DIM:
        raise RuntimeError(f"Unexpected mixed fingerprint length: {mixfp.shape[0]} != {MIXFP_DIM}")
    return mixfp


def mean_pool_last_hidden_state(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def get_bert_embeds(smiles_list):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
    model.eval()

    outputs = []
    with torch.no_grad():
        for smiles in smiles_list:
            inputs = tokenizer(
                smiles,
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(DEVICE)
            output = model(**inputs)
            embeds = mean_pool_last_hidden_state(output.last_hidden_state, inputs["attention_mask"])
            embeds = embeds.squeeze(0).cpu().numpy()
            outputs.append(embeds)
    embeds = np.array(outputs)
    np.save(os.path.join(PROCESSED_DIR, "bert_mean_embeds.npy"), embeds)
    return embeds


def create_pyg_data(indices, smiles_list, labels, ids, names, bert_embeds, mix_fps):
    data_list = []
    for idx in indices:
        x, edge_index, edge_attr = smiles_to_graph(smiles_list[idx])
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mixfp=torch.tensor(mix_fps[idx], dtype=torch.float32),
            bert=torch.tensor(bert_embeds[idx], dtype=torch.float32),
            y=torch.tensor([labels[idx] - 1], dtype=torch.long),
            id=ids[idx],
            name=names[idx],
        )
        data_list.append(data)
    return data_list


def split_and_save_folds(df):
    ensure_dirs()
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    smiles_list = [str(smiles).strip() for smiles in df["SMILES"].tolist()]
    labels = df["Label"].tolist()
    ids = df["ID"].tolist()
    names = df["Name"].tolist()
    validate_smiles_list(smiles_list, ids)

    bert_path = os.path.join(PROCESSED_DIR, "bert_mean_embeds.npy")
    bert_embeds = np.load(bert_path) if os.path.exists(bert_path) else get_bert_embeds(smiles_list)
    mix_fps = [get_mix_fingerprint(smiles) for smiles in smiles_list]

    outer_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_test_idx) in enumerate(outer_split.split(smiles_list, labels)):
        fold_dir = os.path.join(PROCESSED_DIR, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        val_test_smiles = [smiles_list[i] for i in val_test_idx]
        val_test_labels = [labels[i] for i in val_test_idx]

        max_attempts = 100
        val_idx = None
        test_idx = None
        for attempt in range(max_attempts):
            inner_split = StratifiedKFold(n_splits=2, shuffle=True, random_state=42 + attempt)
            val_idx, test_idx = next(inner_split.split(val_test_smiles, val_test_labels))
            val_unique = {val_test_labels[i] for i in val_idx}
            test_unique = {val_test_labels[i] for i in test_idx}
            if len(val_unique) == 6 and len(test_unique) == 6:
                break

        train_orig_idx = list(train_idx)
        val_orig_idx = [val_test_idx[i] for i in val_idx]
        test_orig_idx = [val_test_idx[i] for i in test_idx]

        train_data = create_pyg_data(train_orig_idx, smiles_list, labels, ids, names, bert_embeds, mix_fps)
        val_data = create_pyg_data(val_orig_idx, smiles_list, labels, ids, names, bert_embeds, mix_fps)
        test_data = create_pyg_data(test_orig_idx, smiles_list, labels, ids, names, bert_embeds, mix_fps)

        torch.save(train_data, os.path.join(fold_dir, "train_pyg.pt"))
        torch.save(val_data, os.path.join(fold_dir, "val_pyg.pt"))
        torch.save(test_data, os.path.join(fold_dir, "test_pyg.pt"))

        print(
            f"Fold {fold} | train={len(train_data)} | val={len(val_data)} | test={len(test_data)}"
        )


def get_args():
    parser = argparse.ArgumentParser(description="Build 5-fold processed data for the final baseline.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--encoding", type=str, default="utf-8")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    df = pd.read_csv(args.input_csv, encoding=args.encoding)
    split_and_save_folds(df)
    print(f"Saved processed folds to: {PROCESSED_DIR}")
