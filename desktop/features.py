"""Feature extraction for the versioned TasteMM desktop checkpoint."""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

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
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles("C")

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

def get_mix_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles("C")

    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), dtype=np.float32)
    topo = np.array(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits), dtype=np.float32)
    return np.concatenate([maccs, morgan, topo])[:3239]
