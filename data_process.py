import argparse
import hashlib
import json
import os

os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from transformers import BertModel, BertTokenizer

from tools import PROCESSED_DIR, setup_seed


BERT_MODEL_NAME = "google-bert/bert-base-uncased"
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
MAX_LEN = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MACCS_BITS = 167
RDK_BITS = 1024
ECFP_BITS = 2048
MIXFP_DIM = MACCS_BITS + RDK_BITS + ECFP_BITS
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=ECFP_BITS)

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
    if radius == 2 and ecfp_bits == ECFP_BITS:
        ecfp4 = np.array(MORGAN_GENERATOR.GetFingerprint(mol), dtype=np.float32)
    else:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=ecfp_bits)
        ecfp4 = np.array(generator.GetFingerprint(mol), dtype=np.float32)
    mixfp = np.concatenate([maccs, rdk, ecfp4])
    if mixfp.shape[0] != MIXFP_DIM:
        raise RuntimeError(f"Unexpected mixed fingerprint length: {mixfp.shape[0]} != {MIXFP_DIM}")
    return mixfp


def mean_pool_last_hidden_state(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def ordered_smiles_sha256(smiles_list) -> str:
    digest = hashlib.sha256()
    for smiles in smiles_list:
        digest.update(str(smiles).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_smiles_bert(smiles_list, batch_size=32, device=DEVICE):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, revision=BERT_REVISION)
    model = BertModel.from_pretrained(BERT_MODEL_NAME, revision=BERT_REVISION).to(device)
    model.eval()

    outputs = []
    with torch.no_grad():
        for start in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[start : start + batch_size]
            inputs = tokenizer(
                batch_smiles,
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)
            output = model(**inputs)
            embeds = mean_pool_last_hidden_state(output.last_hidden_state, inputs["attention_mask"])
            outputs.append(embeds.cpu().numpy())
    return np.concatenate(outputs, axis=0), getattr(model.config, "_commit_hash", None)


def get_bert_embeds(smiles_list, processed_dir, batch_size=32):
    embeds, resolved_commit = encode_smiles_bert(smiles_list, batch_size=batch_size, device=DEVICE)
    cache_path = os.path.join(processed_dir, "bert_mean_embeds.npy")
    np.save(cache_path, embeds)
    metadata = {
        "bert_model": BERT_MODEL_NAME,
        "bert_revision": BERT_REVISION,
        "resolved_model_commit": resolved_commit,
        "max_length": MAX_LEN,
        "pooling": "attention-mask mean of last_hidden_state",
        "ordered_smiles_sha256": ordered_smiles_sha256(smiles_list),
        "shape": list(embeds.shape),
    }
    with open(os.path.join(processed_dir, "bert_mean_embeds.metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return embeds


def load_validated_bert_cache(smiles_list, processed_dir):
    cache_path = os.path.join(processed_dir, "bert_mean_embeds.npy")
    metadata_path = os.path.join(processed_dir, "bert_mean_embeds.metadata.json")
    if not os.path.exists(cache_path) or not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "bert_model": BERT_MODEL_NAME,
        "bert_revision": BERT_REVISION,
        "max_length": MAX_LEN,
        "pooling": "attention-mask mean of last_hidden_state",
        "ordered_smiles_sha256": ordered_smiles_sha256(smiles_list),
        "shape": [len(smiles_list), 768],
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        print(f"[CACHE MISS] BERT metadata mismatch; embeddings will be recomputed: {mismatches}")
        return None
    embeddings = np.load(cache_path)
    if embeddings.shape != (len(smiles_list), 768) or not np.isfinite(embeddings).all():
        print(f"[CACHE MISS] Invalid cached BERT array shape or values: {embeddings.shape}")
        return None
    return embeddings


def create_pyg_data(indices, smiles_list, labels, ids, names, bert_embeds, mix_fps):
    data_list = []
    for idx in indices:
        x, edge_index, edge_attr = smiles_to_graph(smiles_list[idx])
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mixfp=torch.tensor(mix_fps[idx], dtype=torch.float32).reshape(1, -1),
            bert=torch.tensor(bert_embeds[idx], dtype=torch.float32).reshape(1, -1),
            y=torch.tensor([labels[idx] - 1], dtype=torch.long),
            row_index=torch.tensor([int(idx)], dtype=torch.long),
            id=ids[idx],
            name=names[idx],
        )
        data_list.append(data)
    return data_list


def _indices_from_frozen_manifest(manifest, fold, df, num_classes):
    partition_column = "partition" if "partition" in manifest.columns else "split"
    index_column = "scope_row_index" if "scope_row_index" in manifest.columns else "row_index"
    label_column = "true_label" if "true_label" in manifest.columns else "Label"
    required = {"fold", partition_column, index_column, label_column}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Frozen split manifest is missing columns: {sorted(missing)}")
    fold_rows = manifest[manifest["fold"].astype(int) == fold].copy()
    if fold_rows.empty:
        raise ValueError(f"Frozen split manifest has no rows for fold {fold}.")
    mapping = {"train": "train", "val": "val", "validation": "val", "test": "test"}
    fold_rows["_partition"] = fold_rows[partition_column].astype(str).str.lower().map(mapping)
    if fold_rows["_partition"].isna().any():
        raise ValueError(f"Fold {fold} contains unsupported partition names.")
    indices = {}
    for split_name in ("train", "val", "test"):
        values = fold_rows.loc[fold_rows["_partition"] == split_name, index_column].astype(int).to_numpy()
        if len(values) == 0:
            raise ValueError(f"Fold {fold} has an empty {split_name} partition.")
        indices[split_name] = values.tolist()
    all_indices = indices["train"] + indices["val"] + indices["test"]
    if len(all_indices) != len(df) or sorted(all_indices) != list(range(len(df))):
        raise ValueError(f"Fold {fold} does not partition dataframe rows 0..{len(df) - 1} exactly once.")
    ordered_manifest = fold_rows.set_index(index_column).sort_index()
    expected_labels = ordered_manifest[label_column].astype(int).to_numpy()
    observed_labels = df["Label"].astype(int).to_numpy()
    if not np.array_equal(expected_labels, observed_labels):
        raise ValueError(f"Fold {fold} manifest labels do not match the input CSV row order.")
    manifest_smiles_column = "smiles" if "smiles" in ordered_manifest.columns else "SMILES"
    if manifest_smiles_column in ordered_manifest.columns:
        expected_smiles = ordered_manifest[manifest_smiles_column].astype(str).str.strip().to_numpy()
        observed_smiles = df["SMILES"].astype(str).str.strip().to_numpy()
        if not np.array_equal(expected_smiles, observed_smiles):
            raise ValueError(f"Fold {fold} manifest SMILES do not match the input CSV row order.")
    for split_name, values in indices.items():
        if set(observed_labels[values]) != set(range(1, num_classes + 1)):
            raise ValueError(f"Fold {fold} {split_name} partition does not contain all {num_classes} classes.")
    return indices


def split_and_save_folds(
    df,
    bert_batch_size=32,
    processed_dir=PROCESSED_DIR,
    num_classes=6,
    split_manifest=None,
):
    setup_seed(42)
    processed_dir = os.path.abspath(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

    required_columns = ["ID", "Name", "SMILES", "Label"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Input CSV is missing required columns: {missing_columns}")
    model_required = ["SMILES", "Label"]
    if df[model_required].isnull().any().any():
        bad_columns = df[model_required].columns[df[model_required].isnull().any()].tolist()
        raise ValueError(f"Input CSV contains missing values in modeling columns: {bad_columns}")
    labels_numeric = pd.to_numeric(df["Label"], errors="raise").astype(int)
    invalid_labels = sorted(set(labels_numeric.tolist()) - set(range(1, num_classes + 1)))
    if invalid_labels:
        raise ValueError(f"TasteMM expects labels 1..{num_classes}; found invalid labels: {invalid_labels}")
    if set(labels_numeric.tolist()) != set(range(1, num_classes + 1)):
        raise ValueError(f"Input CSV must contain every label 1..{num_classes}.")
    df = df.copy().reset_index(drop=True)
    df["Label"] = labels_numeric

    smiles_list = [str(smiles).strip() for smiles in df["SMILES"].tolist()]
    labels = df["Label"].tolist()
    ids = df["ID"].fillna("").astype(str).tolist()
    names = df["Name"].fillna("").astype(str).tolist()
    validate_smiles_list(smiles_list, ids)

    bert_embeds = load_validated_bert_cache(smiles_list, processed_dir)
    if bert_embeds is None:
        bert_embeds = get_bert_embeds(smiles_list, processed_dir, batch_size=bert_batch_size)
    mix_fps = [get_mix_fingerprint(smiles) for smiles in smiles_list]

    frozen_manifest = pd.read_csv(split_manifest) if split_manifest else None
    generated_splits = []
    if frozen_manifest is None:
        outer_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        generated_splits = list(outer_split.split(smiles_list, labels))

    for fold in range(5):
        fold_dir = os.path.join(processed_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        if frozen_manifest is not None:
            indices = _indices_from_frozen_manifest(frozen_manifest, fold, df, num_classes)
            train_orig_idx, val_orig_idx, test_orig_idx = indices["train"], indices["val"], indices["test"]
        else:
            train_idx, val_test_idx = generated_splits[fold]
            val_test_smiles = [smiles_list[i] for i in val_test_idx]
            val_test_labels = [labels[i] for i in val_test_idx]
            val_idx = test_idx = None
            for attempt in range(100):
                inner_split = StratifiedKFold(n_splits=2, shuffle=True, random_state=42 + attempt)
                val_idx, test_idx = next(inner_split.split(val_test_smiles, val_test_labels))
                if (
                    len({val_test_labels[i] for i in val_idx}) == num_classes
                    and len({val_test_labels[i] for i in test_idx}) == num_classes
                ):
                    break
            else:
                raise RuntimeError(f"Could not create {num_classes}-class validation/test partitions for fold {fold}.")
            train_orig_idx = list(train_idx)
            val_orig_idx = [int(val_test_idx[i]) for i in val_idx]
            test_orig_idx = [int(val_test_idx[i]) for i in test_idx]

        train_data = create_pyg_data(train_orig_idx, smiles_list, labels, ids, names, bert_embeds, mix_fps)
        val_data = create_pyg_data(val_orig_idx, smiles_list, labels, ids, names, bert_embeds, mix_fps)
        test_data = create_pyg_data(test_orig_idx, smiles_list, labels, ids, names, bert_embeds, mix_fps)

        torch.save(train_data, os.path.join(fold_dir, "train_pyg.pt"))
        torch.save(val_data, os.path.join(fold_dir, "val_pyg.pt"))
        torch.save(test_data, os.path.join(fold_dir, "test_pyg.pt"))

        split_rows = []
        for split_name, original_indices in (
            ("train", train_orig_idx),
            ("validation", val_orig_idx),
            ("test", test_orig_idx),
        ):
            for original_index in original_indices:
                split_rows.append(
                    {
                        "fold": fold,
                        "split": split_name,
                        "row_index": int(original_index),
                        "ID": ids[original_index],
                        "Name": names[original_index],
                        "SMILES": smiles_list[original_index],
                        "Label": int(labels[original_index]),
                    }
                )
        pd.DataFrame(split_rows).to_csv(
            os.path.join(fold_dir, "split_manifest.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"Fold {fold} | train={len(train_data)} | val={len(val_data)} | test={len(test_data)}"
        )


def get_args():
    parser = argparse.ArgumentParser(description="Build five matched 80/10/10 TasteMM data partitions.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--bert_batch_size", type=int, default=32)
    parser.add_argument("--processed_dir", type=str, default=PROCESSED_DIR)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--split_manifest", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    df = pd.read_csv(args.input_csv, encoding=args.encoding)
    if args.bert_batch_size < 1:
        raise ValueError("--bert_batch_size must be positive")
    if args.num_classes < 2:
        raise ValueError("--num_classes must be at least 2")
    split_and_save_folds(
        df,
        bert_batch_size=args.bert_batch_size,
        processed_dir=args.processed_dir,
        num_classes=args.num_classes,
        split_manifest=args.split_manifest or None,
    )
    processing_metadata = {
        "input_csv": os.path.basename(args.input_csv),
        "input_csv_sha256": file_sha256(args.input_csv),
        "split_manifest": os.path.basename(args.split_manifest) if args.split_manifest else None,
        "split_manifest_sha256": file_sha256(args.split_manifest) if args.split_manifest else None,
        "num_classes": args.num_classes,
        "bert_model": BERT_MODEL_NAME,
        "bert_revision": BERT_REVISION,
        "bert_max_length": MAX_LEN,
        "node_feature_dim": NODE_FEATURE_DIM,
        "edge_feature_dim": EDGE_FEATURE_DIM,
        "mixed_fingerprint_dim": MIXFP_DIM,
    }
    with open(os.path.join(os.path.abspath(args.processed_dir), "processing_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(processing_metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Saved processed folds to: {os.path.abspath(args.processed_dir)}")
