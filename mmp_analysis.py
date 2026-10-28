"""Matched molecular pair analysis for TasteMM.

This single-file workflow reproduces the MMP construction, out-of-fold (OOF)
contrastive Integrated Gradients, and the two population-level validation plots
using saved checkpoints and the TasteMM model implementation in this directory.

Examples
--------
Run MMP construction and OOF attribution::

    python mmp_analysis.py --stage A \
        --input_csv data/compound.csv \
        --split_manifest benchmarks/model_comparison/manifests/six_class_split_manifest.csv \
        --run_root runs \
        --processed_dir data/Multi/processed \
        --output_dir results/mmp

Run statistics and plotting from Stage A outputs::

    python mmp_analysis.py --stage B --output_dir results/mmp

Run both stages::

    python mmp_analysis.py --stage AB [the same path arguments as Stage A]

Input requirements
------------------
``input_csv`` must contain ID, Name, SMILES, and one-based Label columns.
``split_manifest`` must contain row_index, fold, and partition columns. Each
compound must occur in exactly one validation/test partition across the five
folds. The processed PyG files are expected at
``processed_dir/fold_{fold}/{partition}_pyg.pt``. Each fold run directory must
contain ``result.json`` and its referenced classification checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFMCS, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

from model import TasteBaselineModel


SEED = 42
CLASS_NAMES = ["bitter", "sweet", "umami", "salty", "sour", "tasteless"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MIN_TANIMOTO = 0.60
MAX_HEAVY_ATOM_DELTA = 5
MAX_CHANGED_HEAVY_ATOMS_PER_SIDE = 5
IG_STEPS = 32
BOOTSTRAP_REPEATS = 5000
PERMUTATION_REPEATS = 20000
RED = "#EC6D78"
BLUE = "#4094D1"
REPRESENTATIVE_SPECS = [
    ("add a functional group", "switch_00266", 12338, "Ethoxy analogue (record 1852)"),
    ("remove a functional group", "switch_00070", 10759, "Methyl p-coumarate (record 777)"),
    (
        "replace a functional group",
        "switch_00267",
        1821,
        "2-(4-chlorobenzoyl)benzoic acid",
    ),
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(*parts: object, seed: int = SEED) -> int:
    payload = "|".join(map(str, (seed, *parts))).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def json_list(value: object) -> List[int]:
    if isinstance(value, str):
        return [int(item) for item in json.loads(value)]
    if isinstance(value, Iterable):
        return [int(item) for item in value]
    return []


def load_inputs(input_csv: Path, manifest_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    compounds = pd.read_csv(input_csv).reset_index(names="row_index")
    required = {"ID", "Name", "SMILES", "Label"}
    missing = required - set(compounds.columns)
    if missing:
        raise ValueError(f"Missing compound columns: {sorted(missing)}")
    compounds["label_index"] = compounds["Label"].astype(int) - 1
    if not compounds["label_index"].between(0, len(CLASS_NAMES) - 1).all():
        raise ValueError("Label must use one-based class IDs 1-6.")
    compounds["taste"] = compounds["label_index"].map(dict(enumerate(CLASS_NAMES)))

    manifest = pd.read_csv(manifest_csv)
    required_manifest = {"row_index", "fold", "partition"}
    missing = required_manifest - set(manifest.columns)
    if missing:
        raise ValueError(f"Missing split-manifest columns: {sorted(missing)}")
    manifest = manifest.copy()
    manifest["row_index"] = manifest["row_index"].astype(int)
    manifest["fold"] = manifest["fold"].astype(int)
    return compounds, manifest


@dataclass(frozen=True)
class LocalChange:
    mcs_smarts: str
    mcs_atoms: int
    parent_match: tuple[int, ...]
    analogue_match: tuple[int, ...]
    parent_unmatched: tuple[int, ...]
    analogue_unmatched: tuple[int, ...]
    parent_modified: tuple[int, ...]
    analogue_modified: tuple[int, ...]
    parent_attachment_atoms: tuple[int, ...]
    analogue_attachment_atoms: tuple[int, ...]
    edit_type: str


def _heavy_atoms(molecule: Chem.Mol) -> set[int]:
    return {atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1}


def _component_count(molecule: Chem.Mol, atom_indices: set[int]) -> int:
    remaining = set(atom_indices)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            for neighbor in molecule.GetAtomWithIdx(current).GetNeighbors():
                index = neighbor.GetIdx()
                if index in remaining:
                    remaining.remove(index)
                    stack.append(index)
    return components


def _attachment_atoms(
    molecule: Chem.Mol, unmatched: set[int], matched: set[int]
) -> set[int]:
    result: set[int] = set()
    for atom_index in unmatched:
        for neighbor in molecule.GetAtomWithIdx(atom_index).GetNeighbors():
            if neighbor.GetIdx() in matched:
                result.add(neighbor.GetIdx())
    return result


def _edit_type(parent_unmatched: int, analogue_unmatched: int) -> str:
    if parent_unmatched == 0 and analogue_unmatched > 0:
        return "addition"
    if parent_unmatched > 0 and analogue_unmatched == 0:
        return "deletion"
    if parent_unmatched == analogue_unmatched == 1:
        return "single-atom substitution"
    return "local replacement"


def identify_local_change(
    parent: Chem.Mol, analogue: Chem.Mol, timeout: int = 3
) -> LocalChange | None:
    """Identify one local edit and its MCS-defined modified atom regions."""
    result = rdFMCS.FindMCS(
        [parent, analogue],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        matchValences=True,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=timeout,
    )
    if result.canceled or result.numAtoms == 0:
        return None
    query = Chem.MolFromSmarts(result.smartsString)
    if query is None:
        return None
    parent_match = parent.GetSubstructMatch(query)
    analogue_match = analogue.GetSubstructMatch(query)
    if not parent_match or not analogue_match or len(parent_match) != len(analogue_match):
        return None

    parent_heavy = _heavy_atoms(parent)
    analogue_heavy = _heavy_atoms(analogue)
    parent_matched = set(parent_match) & parent_heavy
    analogue_matched = set(analogue_match) & analogue_heavy
    parent_unmatched = parent_heavy - parent_matched
    analogue_unmatched = analogue_heavy - analogue_matched
    if not parent_unmatched and not analogue_unmatched:
        return None
    if len(parent_unmatched) > MAX_CHANGED_HEAVY_ATOMS_PER_SIDE:
        return None
    if len(analogue_unmatched) > MAX_CHANGED_HEAVY_ATOMS_PER_SIDE:
        return None
    if _component_count(parent, parent_unmatched) > 1:
        return None
    if _component_count(analogue, analogue_unmatched) > 1:
        return None

    parent_attachment = _attachment_atoms(parent, parent_unmatched, parent_matched)
    analogue_attachment = _attachment_atoms(analogue, analogue_unmatched, analogue_matched)
    if len(parent_attachment) > 2 or len(analogue_attachment) > 2:
        return None
    analogue_to_parent = {
        analogue_match[index]: parent_match[index] for index in range(len(parent_match))
    }
    parent_boundary = {
        analogue_to_parent[index]
        for index in analogue_attachment
        if index in analogue_to_parent
    }
    parent_modified = parent_unmatched | parent_attachment | parent_boundary
    if not parent_modified or (not parent_attachment and not analogue_attachment):
        return None
    parent_to_analogue = {value: key for key, value in analogue_to_parent.items()}
    analogue_boundary = {
        parent_to_analogue[index]
        for index in parent_attachment
        if index in parent_to_analogue
    }
    analogue_modified = analogue_unmatched | analogue_attachment | analogue_boundary
    return LocalChange(
        mcs_smarts=result.smartsString,
        mcs_atoms=int(result.numAtoms),
        parent_match=tuple(map(int, parent_match)),
        analogue_match=tuple(map(int, analogue_match)),
        parent_unmatched=tuple(sorted(parent_unmatched)),
        analogue_unmatched=tuple(sorted(analogue_unmatched)),
        parent_modified=tuple(sorted(parent_modified)),
        analogue_modified=tuple(sorted(analogue_modified)),
        parent_attachment_atoms=tuple(sorted(parent_attachment | parent_boundary)),
        analogue_attachment_atoms=tuple(sorted(analogue_attachment | analogue_boundary)),
        edit_type=_edit_type(len(parent_unmatched), len(analogue_unmatched)),
    )


def operation_category(edit_type: str) -> str:
    if edit_type == "addition":
        return "add a functional group"
    if edit_type == "deletion":
        return "remove a functional group"
    return "replace a functional group"


def reverse_local_change(change: LocalChange) -> LocalChange:
    return LocalChange(
        mcs_smarts=change.mcs_smarts,
        mcs_atoms=change.mcs_atoms,
        parent_match=change.analogue_match,
        analogue_match=change.parent_match,
        parent_unmatched=change.analogue_unmatched,
        analogue_unmatched=change.parent_unmatched,
        parent_modified=change.analogue_modified,
        analogue_modified=change.parent_modified,
        parent_attachment_atoms=change.analogue_attachment_atoms,
        analogue_attachment_atoms=change.parent_attachment_atoms,
        edit_type=_edit_type(
            len(change.analogue_unmatched), len(change.parent_unmatched)
        ),
    )


class MMPBuilder:
    def __init__(self, compounds: pd.DataFrame, manifest: pd.DataFrame):
        self.frame = compounds
        self.manifest = manifest
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
        )
        self.molecules = [Chem.MolFromSmiles(str(value)) for value in compounds["SMILES"]]
        if any(molecule is None for molecule in self.molecules):
            bad = [index for index, molecule in enumerate(self.molecules) if molecule is None]
            raise ValueError(f"Invalid SMILES rows: {bad[:20]}")
        self.fingerprints = [
            self.generator.GetFingerprint(molecule) for molecule in self.molecules
        ]
        self.heavy_atoms = np.asarray(
            [molecule.GetNumHeavyAtoms() for molecule in self.molecules], dtype=int
        )
        self.canonical = [
            Chem.MolToSmiles(molecule, canonical=True) for molecule in self.molecules
        ]
        self.training_canonical = {
            fold: {
                self.canonical[int(row_index)]
                for row_index in manifest.loc[
                    (manifest["fold"] == fold) & (manifest["partition"] == "train"),
                    "row_index",
                ]
            }
            for fold in sorted(manifest["fold"].unique())
        }

    def partition(self, row_index: int, fold: int) -> str:
        rows = self.manifest[
            (self.manifest["row_index"] == int(row_index))
            & (self.manifest["fold"] == int(fold))
        ]
        if len(rows) != 1:
            raise KeyError((row_index, fold))
        return str(rows.iloc[0]["partition"])

    def pair_row(
        self,
        source_index: int,
        analogue_index: int,
        fold: int,
        change: LocalChange,
    ) -> Dict:
        source_taste = str(self.frame.iloc[source_index]["taste"])
        analogue_taste = str(self.frame.iloc[analogue_index]["taste"])
        similarity = DataStructs.TanimotoSimilarity(
            self.fingerprints[source_index], self.fingerprints[analogue_index]
        )
        return {
            "source_row_index": int(source_index),
            "analogue_row_index": int(analogue_index),
            "fold": int(fold),
            "source_partition": self.partition(source_index, fold),
            "analogue_partition": self.partition(analogue_index, fold),
            "source_name": str(self.frame.iloc[source_index]["Name"]),
            "analogue_name": str(self.frame.iloc[analogue_index]["Name"]),
            "source_smiles": str(self.frame.iloc[source_index]["SMILES"]),
            "analogue_smiles": str(self.frame.iloc[analogue_index]["SMILES"]),
            "source_taste": source_taste,
            "analogue_taste": analogue_taste,
            "tanimoto": float(similarity),
            "source_heavy_atoms": int(self.heavy_atoms[source_index]),
            "analogue_heavy_atoms": int(self.heavy_atoms[analogue_index]),
            "heavy_atom_delta": int(
                self.heavy_atoms[analogue_index] - self.heavy_atoms[source_index]
            ),
            "mcs_smarts": change.mcs_smarts,
            "mcs_atoms": int(change.mcs_atoms),
            "mcs_source_fraction": float(
                change.mcs_atoms / max(self.heavy_atoms[source_index], 1)
            ),
            "source_unmatched_atoms": json.dumps(change.parent_unmatched),
            "analogue_unmatched_atoms": json.dumps(change.analogue_unmatched),
            "source_modified_atoms": json.dumps(change.parent_modified),
            "analogue_modified_atoms": json.dumps(change.analogue_modified),
            "source_attachment_atoms": json.dumps(change.parent_attachment_atoms),
            "analogue_attachment_atoms": json.dumps(change.analogue_attachment_atoms),
            "modified_source_size": int(len(change.parent_modified)),
            "modified_analogue_size": int(len(change.analogue_modified)),
            "edit_type": change.edit_type,
            "operation_category": operation_category(change.edit_type),
            "both_members_held_out": True,
            "source_canonical_in_training": bool(
                self.canonical[source_index] in self.training_canonical[fold]
            ),
            "analogue_canonical_in_training": bool(
                self.canonical[analogue_index] in self.training_canonical[fold]
            ),
            "structure_level_held_out": bool(
                self.canonical[source_index] not in self.training_canonical[fold]
                and self.canonical[analogue_index] not in self.training_canonical[fold]
            ),
        }

    def enumerate_pairs(self, switching: bool) -> pd.DataFrame:
        labels = self.frame["label_index"].to_numpy(dtype=int)
        rows: List[Dict] = []
        folds = sorted(self.manifest["fold"].unique())
        for fold in folds:
            heldout = self.manifest.loc[
                (self.manifest["fold"] == fold)
                & self.manifest["partition"].isin(["val", "test"]),
                "row_index",
            ].astype(int).to_numpy()
            heldout.sort()
            for position, first_index in enumerate(heldout[:-1]):
                other_indices = heldout[position + 1 :]
                similarities = np.asarray(
                    DataStructs.BulkTanimotoSimilarity(
                        self.fingerprints[int(first_index)],
                        [self.fingerprints[int(index)] for index in other_indices],
                    ),
                    dtype=float,
                )
                taste_mask = (
                    labels[other_indices] != labels[int(first_index)]
                    if switching
                    else labels[other_indices] == labels[int(first_index)]
                )
                eligible = (
                    taste_mask
                    & (similarities >= MIN_TANIMOTO)
                    & (
                        np.abs(
                            self.heavy_atoms[other_indices]
                            - self.heavy_atoms[int(first_index)]
                        )
                        <= MAX_HEAVY_ATOM_DELTA
                    )
                )
                for second_index in other_indices[eligible]:
                    first_index = int(first_index)
                    second_index = int(second_index)
                    if self.canonical[first_index] == self.canonical[second_index]:
                        continue
                    if switching and labels[first_index] > labels[second_index]:
                        source_index, analogue_index = second_index, first_index
                    else:
                        source_index, analogue_index = first_index, second_index
                    change = identify_local_change(
                        self.molecules[source_index], self.molecules[analogue_index]
                    )
                    if change is None:
                        continue
                    row = self.pair_row(source_index, analogue_index, fold, change)
                    if switching:
                        rows.append(row)
                    else:
                        control_pair_id = (
                            f"same_{min(source_index, analogue_index):05d}_"
                            f"{max(source_index, analogue_index):05d}"
                        )
                        row["control_pair_id"] = control_pair_id
                        rows.append(row)
                        reverse = self.pair_row(
                            analogue_index,
                            source_index,
                            fold,
                            reverse_local_change(change),
                        )
                        reverse["control_pair_id"] = control_pair_id
                        rows.append(reverse)
        return pd.DataFrame(rows)

    def enumerate_control_candidates(self) -> pd.DataFrame:
        """Pre-rank same-taste candidates before the relatively costly MCS step."""
        labels = self.frame["label_index"].to_numpy(dtype=int)
        specifications: List[Dict] = []
        for fold in sorted(self.manifest["fold"].unique()):
            heldout = self.manifest.loc[
                (self.manifest["fold"] == fold)
                & self.manifest["partition"].isin(["val", "test"]),
                "row_index",
            ].astype(int).to_numpy()
            heldout.sort()
            for position, first_index in enumerate(heldout[:-1]):
                other_indices = heldout[position + 1 :]
                similarities = np.asarray(
                    DataStructs.BulkTanimotoSimilarity(
                        self.fingerprints[int(first_index)],
                        [self.fingerprints[int(index)] for index in other_indices],
                    ),
                    dtype=float,
                )
                eligible = (
                    (labels[other_indices] == labels[int(first_index)])
                    & (similarities >= MIN_TANIMOTO)
                    & (
                        np.abs(
                            self.heavy_atoms[other_indices]
                            - self.heavy_atoms[int(first_index)]
                        )
                        <= MAX_HEAVY_ATOM_DELTA
                    )
                )
                for second_index, similarity in zip(
                    other_indices[eligible], similarities[eligible]
                ):
                    first_index = int(first_index)
                    second_index = int(second_index)
                    if self.canonical[first_index] == self.canonical[second_index]:
                        continue
                    specifications.append(
                        {
                            "first_index": first_index,
                            "second_index": second_index,
                            "fold": int(fold),
                            "taste": str(self.frame.iloc[first_index]["taste"]),
                            "tanimoto": float(similarity),
                        }
                    )
        ranked = pd.DataFrame(specifications).sort_values(
            ["taste", "tanimoto", "first_index", "second_index"],
            ascending=[True, False, True, True],
        )
        limits = {
            "bitter": 850,
            "sweet": 1250,
            "umami": 250,
            "salty": 100,
            "sour": 450,
            "tasteless": 1050,
        }
        ranked = pd.concat(
            [
                part.head(limits.get(str(taste), 500))
                for taste, part in ranked.groupby("taste")
            ],
            ignore_index=True,
        )
        rows: List[Dict] = []
        for specification in ranked.itertuples(index=False):
            first_index = int(specification.first_index)
            second_index = int(specification.second_index)
            change = identify_local_change(
                self.molecules[first_index], self.molecules[second_index]
            )
            if change is None:
                continue
            control_pair_id = f"same_{first_index:05d}_{second_index:05d}"
            forward = self.pair_row(
                first_index, second_index, int(specification.fold), change
            )
            forward["control_pair_id"] = control_pair_id
            rows.append(forward)
            reverse = self.pair_row(
                second_index,
                first_index,
                int(specification.fold),
                reverse_local_change(change),
            )
            reverse["control_pair_id"] = control_pair_id
            rows.append(reverse)
        return pd.DataFrame(rows)


def match_controls(switching: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    features = [
        "tanimoto",
        "heavy_atom_delta",
        "modified_source_size",
        "mcs_source_fraction",
    ]
    all_values = pd.concat([switching[features], candidates[features]], ignore_index=True)
    scale = all_values.std(ddof=0).replace(0, 1.0)
    center = all_values.mean()
    candidate_z = (candidates[features] - center) / scale
    used_pairs: set[str] = set()
    rows: List[Dict] = []
    for switch in switching.itertuples(index=False):
        endpoints = {str(switch.source_taste), str(switch.analogue_taste)}
        available = (
            ~candidates["control_pair_id"].isin(used_pairs)
            & (candidates["operation_category"] == switch.operation_category)
            & candidates["source_taste"].isin(endpoints)
        )
        endpoint_matched = True
        if not available.any():
            available = (
                ~candidates["control_pair_id"].isin(used_pairs)
                & (candidates["operation_category"] == switch.operation_category)
            )
            endpoint_matched = False
        if not available.any():
            raise RuntimeError(f"No unused control for {switch.pair_id}.")
        switch_values = pd.Series(
            {feature: getattr(switch, feature) for feature in features}, dtype=float
        )
        switch_z = (switch_values - center) / scale
        indices = candidates.index[available]
        distances = np.sqrt(((candidate_z.loc[indices] - switch_z) ** 2).sum(axis=1))
        selected = int(distances.sort_values(kind="stable").index[0])
        row = candidates.loc[selected].to_dict()
        used_pairs.add(str(row["control_pair_id"]))
        control_taste = str(row["source_taste"])
        target_taste = (
            str(switch.analogue_taste)
            if control_taste != str(switch.analogue_taste)
            else str(switch.source_taste)
        )
        row.update(
            {
                "pair_group": "non-switching",
                "pair_id": f"control_{len(rows):05d}",
                "matched_switch_pair_id": str(switch.pair_id),
                "matching_distance": float(distances.loc[selected]),
                "control_taste_matched_to_switch_endpoint": endpoint_matched,
                "contrast_source_taste": control_taste,
                "contrast_target_taste": target_taste,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_all_unique_mmp_table(
    compounds: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    builder = MMPBuilder(compounds, manifest)
    switching = builder.enumerate_pairs(switching=True).sort_values(
        ["source_taste", "analogue_taste", "source_row_index", "analogue_row_index"]
    ).reset_index(drop=True)
    switching["pair_group"] = "taste-switching"
    switching["pair_id"] = [f"switch_{index:05d}" for index in range(len(switching))]
    switching["matched_switch_pair_id"] = ""
    switching["matching_distance"] = np.nan
    switching["control_taste_matched_to_switch_endpoint"] = True
    switching["contrast_source_taste"] = switching["source_taste"]
    switching["contrast_target_taste"] = switching["analogue_taste"]

    controls = builder.enumerate_control_candidates()
    if controls.empty:
        raise RuntimeError("No non-switching MMP controls passed the fixed criteria.")
    controls = controls.sort_values(
        ["source_taste", "tanimoto", "source_row_index", "analogue_row_index"],
        ascending=[True, False, True, True],
    )
    matched = match_controls(switching, controls)
    return pd.concat([switching, matched], ignore_index=True, sort=False)


def _torch_load(path: Path, map_location=None):
    try:
        return torch.load(path, weights_only=False, map_location=map_location)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class FoldBundle:
    model: torch.nn.Module
    temperature: float


class OOFRuntime:
    """Load public TasteMM checkpoints and compute graph-input contrastive IG."""

    def __init__(
        self,
        compounds: pd.DataFrame,
        manifest: pd.DataFrame,
        run_root: Path,
        processed_dir: Path,
        device: str | None,
        ig_steps: int,
    ):
        self.compounds = compounds
        self.manifest = manifest
        self.run_root = run_root
        self.processed_dir = processed_dir
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.ig_steps = int(ig_steps)
        self.fold_dirs = self._find_fold_dirs()
        self.fold_cache: Dict[int, FoldBundle] = {}
        self.data_cache: Dict[tuple[int, str], list] = {}
        self.position_cache: Dict[tuple[int, str], Dict[int, int]] = {}

    def _find_fold_dirs(self) -> Dict[int, Path]:
        result: Dict[int, Path] = {}
        for value in sorted(glob.glob(str(self.run_root / "fold*"))):
            path = Path(value)
            result_json = path / "result.json"
            if not result_json.exists():
                continue
            payload = json.loads(result_json.read_text(encoding="utf-8"))
            result[int(payload["config"]["fold"])] = path
        expected = set(map(int, self.manifest["fold"].unique()))
        if set(result) != expected:
            raise ValueError(f"Expected folds {sorted(expected)}; found {sorted(result)}")
        return result

    def load_data(self, fold: int, partition: str) -> list:
        key = (int(fold), str(partition))
        if key not in self.data_cache:
            path = self.processed_dir / f"fold_{fold}" / f"{partition}_pyg.pt"
            data = _torch_load(path)
            expected = self.manifest[
                (self.manifest["fold"] == fold)
                & (self.manifest["partition"] == partition)
            ]
            if len(data) != len(expected):
                raise ValueError(f"Processed/manifest mismatch for fold={fold}, {partition}")
            self.data_cache[key] = data
            self.position_cache[key] = {
                int(row_index): position
                for position, row_index in enumerate(expected["row_index"].astype(int))
            }
        return self.data_cache[key]

    def load_fold(self, fold: int) -> FoldBundle:
        fold = int(fold)
        if fold not in self.fold_cache:
            fold_dir = self.fold_dirs[fold]
            payload = json.loads((fold_dir / "result.json").read_text(encoding="utf-8"))
            reference = self.load_data(fold, "test")[0]
            edge_attr = getattr(reference, "edge_attr", None)
            model = TasteBaselineModel(
                num_graph_features=int(reference.x.size(-1)),
                edge_attr_dim=int(edge_attr.size(-1)) if edge_attr is not None else 18,
                num_classes=int(payload["config"].get("num_classes", 6)),
            ).to(self.device)
            recorded = Path(payload["checkpoints"]["best_classify_model"])
            candidates = [
                recorded,
                fold_dir / "checkpoints" / recorded.name,
                fold_dir / recorded.name,
            ]
            checkpoint = next((path for path in candidates if path.exists()), None)
            if checkpoint is None:
                raise FileNotFoundError(f"Checkpoint not found; tried {candidates}")
            model.load_state_dict(_torch_load(checkpoint, map_location=self.device))
            model.eval()
            self.fold_cache[fold] = FoldBundle(
                model=model,
                temperature=float(payload.get("temperature_T", 1.0)),
            )
        return self.fold_cache[fold]

    def sample(self, row_index: int, fold: int, partition: str):
        key = (int(fold), str(partition))
        data = self.load_data(*key)
        sample = data[self.position_cache[key][int(row_index)]].clone()
        sample.batch = torch.zeros(sample.x.size(0), dtype=torch.long)
        return sample.to(self.device)

    @staticmethod
    def _aggregate(feature_attribution: torch.Tensor) -> Dict[str, np.ndarray]:
        contribution = feature_attribution.detach().cpu().numpy()
        absolute = np.abs(contribution).sum(axis=1)
        signed = contribution.sum(axis=1)
        return {"abs_raw": absolute, "signed_raw": signed}

    def contrastive_ig_pairs(
        self,
        sample,
        fold: int,
        contrasts: Sequence[tuple[int, int]],
    ) -> Dict[tuple[int, int], Dict[str, np.ndarray]]:
        model = self.load_fold(fold).model
        contrasts = list(dict.fromkeys((int(a), int(b)) for a, b in contrasts))
        original = sample.x.detach().clone()
        baseline = torch.zeros_like(original)
        gradients = {contrast: torch.zeros_like(original) for contrast in contrasts}
        alphas = torch.linspace(
            1.0 / self.ig_steps, 1.0, self.ig_steps, device=original.device
        )
        for alpha in alphas:
            interpolated = (baseline + alpha * (original - baseline)).detach()
            interpolated.requires_grad_(True)
            step_sample = sample.clone()
            step_sample.x = interpolated
            model.zero_grad(set_to_none=True)
            logits = model(step_sample, mode="classify").reshape(-1)
            for index, (source, target) in enumerate(contrasts):
                gradient = torch.autograd.grad(
                    logits[source] - logits[target],
                    interpolated,
                    retain_graph=index < len(contrasts) - 1,
                )[0]
                gradients[(source, target)] += gradient
        return {
            contrast: self._aggregate(
                (original - baseline) * gradients[contrast] / float(self.ig_steps)
            )
            for contrast in contrasts
        }

    def logits_and_probabilities(self, sample, fold: int) -> tuple[np.ndarray, np.ndarray]:
        bundle = self.load_fold(fold)
        with torch.no_grad():
            logits = bundle.model(sample, mode="classify").reshape(-1)
            probabilities = torch.softmax(logits / bundle.temperature, dim=0)
        return logits.cpu().numpy(), probabilities.cpu().numpy()


def ig_key(fold: int, row_index: int, source: str, target: str) -> str:
    return f"fold{fold}_row{row_index}_{source}_minus_{target}"


def run_stage_a(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    RDLogger.DisableLog("rdApp.warning")
    compounds, manifest = load_inputs(args.input_csv, args.split_manifest)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_ig_path = output_dir / "stage_A_mmp_pairs_pre_ig.csv"
    if pre_ig_path.exists() and not args.rebuild_mmp:
        pairs = pd.read_csv(pre_ig_path)
        print(f"Reusing {pre_ig_path}")
    else:
        pairs = build_all_unique_mmp_table(compounds, manifest)
        write_csv(pairs, pre_ig_path)
    print(
        f"MMPs: {sum(pairs.pair_group == 'taste-switching')} switching and "
        f"{sum(pairs.pair_group == 'non-switching')} matched controls"
    )

    runtime = OOFRuntime(
        compounds,
        manifest,
        args.run_root,
        args.processed_dir,
        args.device,
        args.ig_steps,
    )
    pairs["source_ig_key"] = [
        ig_key(
            int(row.fold),
            int(row.source_row_index),
            str(row.contrast_source_taste),
            str(row.contrast_target_taste),
        )
        for row in pairs.itertuples(index=False)
    ]
    switching_mask = pairs["pair_group"] == "taste-switching"
    pairs["analogue_ig_key"] = ""
    pairs.loc[switching_mask, "analogue_ig_key"] = [
        ig_key(
            int(row.fold),
            int(row.analogue_row_index),
            str(row.contrast_source_taste),
            str(row.contrast_target_taste),
        )
        for row in pairs.loc[switching_mask].itertuples(index=False)
    ]
    task_rows: List[Dict] = []
    for pair in pairs.itertuples(index=False):
        task_rows.append(
            {
                "ig_key": pair.source_ig_key,
                "fold": int(pair.fold),
                "partition": str(pair.source_partition),
                "row_index": int(pair.source_row_index),
                "smiles": str(pair.source_smiles),
                "source_taste": str(pair.contrast_source_taste),
                "target_taste": str(pair.contrast_target_taste),
            }
        )
        if pair.pair_group == "taste-switching":
            task_rows.append(
                {
                    "ig_key": pair.analogue_ig_key,
                    "fold": int(pair.fold),
                    "partition": str(pair.analogue_partition),
                    "row_index": int(pair.analogue_row_index),
                    "smiles": str(pair.analogue_smiles),
                    "source_taste": str(pair.contrast_source_taste),
                    "target_taste": str(pair.contrast_target_taste),
                }
            )
    tasks = pd.DataFrame(task_rows).drop_duplicates("ig_key")
    atom_rows: List[Dict] = []
    summaries: List[Dict] = []
    grouped = tasks.groupby(["fold", "partition", "row_index", "smiles"], sort=True)
    completed = 0
    for _, task_group in grouped:
        first = task_group.iloc[0]
        sample = runtime.sample(int(first.row_index), int(first.fold), str(first.partition))
        molecule = Chem.MolFromSmiles(str(first.smiles))
        if molecule is None or molecule.GetNumAtoms() != sample.x.size(0):
            raise ValueError(f"Atom count/order mismatch for row {first.row_index}")
        contrasts = [
            (CLASS_TO_INDEX[row.source_taste], CLASS_TO_INDEX[row.target_taste])
            for row in task_group.itertuples(index=False)
        ]
        attributions = runtime.contrastive_ig_pairs(sample, int(first.fold), contrasts)
        logits, probabilities = runtime.logits_and_probabilities(sample, int(first.fold))
        for task, contrast in zip(task_group.itertuples(index=False), contrasts):
            payload = attributions[contrast]
            absolute = np.asarray(payload["abs_raw"], dtype=float)
            signed = np.asarray(payload["signed_raw"], dtype=float)
            total = max(float(absolute.sum()), 1e-12)
            summaries.append(
                {
                    "ig_key": task.ig_key,
                    "model_contrast": float(logits[contrast[0]] - logits[contrast[1]]),
                    "p_contrast_source": float(probabilities[contrast[0]]),
                    "p_contrast_target": float(probabilities[contrast[1]]),
                    "total_abs_ig": total,
                    "total_signed_ig": float(signed.sum()),
                }
            )
            for atom_index, atom in enumerate(molecule.GetAtoms()):
                atom_rows.append(
                    {
                        "ig_key": task.ig_key,
                        "fold": int(task.fold),
                        "row_index": int(task.row_index),
                        "contrast_source_taste": task.source_taste,
                        "contrast_target_taste": task.target_taste,
                        "atom_index": atom_index,
                        "atom_symbol": atom.GetSymbol(),
                        "attribution_abs_raw": float(absolute[atom_index]),
                        "attribution_abs_mass": float(absolute[atom_index] / total),
                        "attribution_signed_raw": float(signed[atom_index]),
                    }
                )
        completed += len(task_group)
        if completed % 25 < len(task_group) or completed == len(tasks):
            print(f"OOF contrastive IG: {completed}/{len(tasks)}")

    atoms = pd.DataFrame(atom_rows)
    summary = pd.DataFrame(summaries).set_index("ig_key")
    atom_lookup = {
        key: part.set_index("atom_index") for key, part in atoms.groupby("ig_key")
    }

    def region_summary(key: str, region: object) -> tuple[float, float]:
        part = atom_lookup[str(key)]
        indices = json_list(region)
        mass = float(part.loc[indices, "attribution_abs_raw"].sum()) / max(
            float(part["attribution_abs_raw"].sum()), 1e-12
        )
        signed_sum = float(part.loc[indices, "attribution_signed_raw"].sum())
        return mass, signed_sum

    source_values = [
        region_summary(row.source_ig_key, row.source_modified_atoms)
        for row in pairs.itertuples(index=False)
    ]
    pairs["modified_region_ig_mass"] = [value[0] for value in source_values]
    pairs["source_modified_ig_signed"] = [value[1] for value in source_values]
    pairs["source_model_contrast"] = pairs["source_ig_key"].map(
        summary["model_contrast"]
    )
    pairs["analogue_modified_ig_signed"] = np.nan
    pairs["analogue_model_contrast"] = np.nan
    for index in pairs.index[switching_mask]:
        row = pairs.loc[index]
        pairs.loc[index, "analogue_modified_ig_signed"] = region_summary(
            str(row.analogue_ig_key), row.analogue_modified_atoms
        )[1]
        pairs.loc[index, "analogue_model_contrast"] = summary.loc[
            str(row.analogue_ig_key), "model_contrast"
        ]
    pairs["delta_modified_attribution_raw"] = (
        pairs["source_modified_ig_signed"] - pairs["analogue_modified_ig_signed"]
    )
    pairs["delta_model_contrast"] = (
        pairs["source_model_contrast"] - pairs["analogue_model_contrast"]
    )
    write_csv(pairs, output_dir / "stage_A_mmp_pairs.csv")
    write_csv(atoms, output_dir / "stage_A_contrastive_ig_atoms.csv")
    write_csv(summary.reset_index(), output_dir / "stage_A_ig_task_summary.csv")
    representatives = plot_representative_mmps(
        pairs, atoms, compounds, manifest, output_dir / "PanelC"
    )
    write_csv(representatives, output_dir / "stage_A_panel_representatives.csv")


def bootstrap_ci(values: Sequence[float], seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_REPEATS)
    for index in range(BOOTSTRAP_REPEATS):
        estimates[index] = rng.choice(array, len(array), replace=True).mean()
    return tuple(map(float, np.quantile(estimates, [0.025, 0.975])))


def paired_permutation(differences: Sequence[float], seed: int) -> float:
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(PERMUTATION_REPEATS):
        signs = rng.choice([-1.0, 1.0], size=len(differences))
        exceed += abs(float(np.mean(signs * differences))) >= observed
    return float((exceed + 1) / (PERMUTATION_REPEATS + 1))


def spearman_statistics(values: pd.DataFrame, scope: str) -> Dict:
    from scipy.stats import rankdata, spearmanr

    x = values["delta_modified_attribution"].to_numpy(float)
    y = values["delta_model_contrast"].to_numpy(float)
    observed = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(stable_seed("stageB", "consistency", scope))
    bootstrap = np.empty(BOOTSTRAP_REPEATS)
    for index in range(BOOTSTRAP_REPEATS):
        selected = rng.integers(0, len(x), len(x))
        bootstrap[index] = spearmanr(x[selected], y[selected]).statistic
    ranked_x = rankdata(x)
    ranked_y = rankdata(y)
    ranked_x = (ranked_x - ranked_x.mean()) / ranked_x.std(ddof=0)
    ranked_y = (ranked_y - ranked_y.mean()) / ranked_y.std(ddof=0)
    exceed = 0
    for _ in range(PERMUTATION_REPEATS):
        permuted = float(np.mean(ranked_x * rng.permutation(ranked_y)))
        exceed += abs(permuted) >= abs(observed)
    return {
        "scope": scope,
        "n_pairs": len(values),
        "spearman_rho": observed,
        "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "permutation_p": float((exceed + 1) / (PERMUTATION_REPEATS + 1)),
    }


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 22,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 18,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _display_name(value: object, width: int = 27) -> str:
    lines = textwrap.wrap(
        str(value).replace("_", " ").strip(),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(lines) > 2:
        lines = [lines[0], lines[1][: max(width - 3, 4)].rstrip() + "..."]
    return "\n".join(lines)


def _black_molecule_image(
    smiles: str,
    modified_atoms: Sequence[int],
    reference_smiles: str | None = None,
    reference_atom_map: Sequence[tuple[int, int]] = (),
    size: tuple[int, int] = (640, 620),
) -> Image.Image:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    molecule = Chem.Mol(molecule)
    if reference_smiles is None:
        Chem.rdDepictor.Compute2DCoords(molecule)
    else:
        reference = Chem.MolFromSmiles(str(reference_smiles))
        if reference is None:
            raise ValueError(f"Invalid reference SMILES: {reference_smiles}")
        reference = Chem.Mol(reference)
        Chem.rdDepictor.Compute2DCoords(reference)
        Chem.rdDepictor.GenerateDepictionMatching2DStructure(
            molecule, reference, list(reference_atom_map)
        )
    conformer = molecule.GetConformer()
    for atom_index in range(molecule.GetNumAtoms()):
        point = conformer.GetAtomPosition(atom_index)
        conformer.SetAtomPosition(atom_index, (-point.y, point.x, point.z))

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    options = drawer.drawOptions()
    options.padding = 0.06
    options.clearBackground = True
    options.useBWAtomPalette()
    drawer.DrawMolecule(molecule)
    coordinates = [
        (
            float(drawer.GetDrawCoords(int(atom_index)).x),
            float(drawer.GetDrawCoords(int(atom_index)).y),
        )
        for atom_index in modified_atoms
        if 0 <= int(atom_index) < molecule.GetNumAtoms()
    ]
    drawer.FinishDrawing()
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    if coordinates:
        xs, ys = zip(*coordinates)
        x_padding = max(23.0, (max(xs) - min(xs)) * 0.18 + 16.0)
        y_padding = max(23.0, (max(ys) - min(ys)) * 0.18 + 16.0)
        overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
        painter = ImageDraw.Draw(overlay)
        painter.ellipse(
            (
                min(xs) - x_padding,
                min(ys) - y_padding,
                max(xs) + x_padding,
                max(ys) + y_padding,
            ),
            fill=(103, 183, 209, 55),
            outline=(68, 162, 193, 220),
            width=4,
        )
        image = Image.alpha_composite(image, overlay)
    return image.convert("RGB")


def plot_representative_mmps(
    pairs: pd.DataFrame,
    atoms: pd.DataFrame,
    compounds: pd.DataFrame,
    manifest: pd.DataFrame,
    stem: Path,
) -> pd.DataFrame:
    """Reproduce the three prespecified same-parent switching/control examples."""
    setup_plot_style()
    switching = pairs[pairs["pair_group"] == "taste-switching"]
    compounds = compounds.set_index("row_index")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    atom_lookup = {
        key: part.sort_values("atom_index") for key, part in atoms.groupby("ig_key")
    }
    examples: List[Dict] = []
    for operation, pair_id, control_index, control_display_name in REPRESENTATIVE_SPECS:
        selected = switching[switching["pair_id"] == pair_id]
        if len(selected) != 1 or control_index not in compounds.index:
            raise ValueError(
                f"Prespecified representative is unavailable: {pair_id}, row {control_index}"
            )
        switch = selected.iloc[0]
        control = compounds.loc[control_index]
        source_molecule = Chem.MolFromSmiles(str(switch.source_smiles))
        switch_molecule = Chem.MolFromSmiles(str(switch.analogue_smiles))
        control_molecule = Chem.MolFromSmiles(str(control.SMILES))
        switch_change = identify_local_change(source_molecule, switch_molecule)
        control_change = identify_local_change(source_molecule, control_molecule)
        if switch_change is None or control_change is None:
            raise ValueError(f"No local change for representative {pair_id}")
        if operation_category(control_change.edit_type) != operation:
            raise ValueError(f"Operation mismatch for representative {pair_id}")
        similarity = DataStructs.TanimotoSimilarity(
            generator.GetFingerprint(source_molecule),
            generator.GetFingerprint(control_molecule),
        )
        source_atoms = atom_lookup[str(switch.source_ig_key)]
        absolute = source_atoms.attribution_signed_raw.abs().to_numpy(float)
        control_region = list(control_change.parent_modified)
        control_mass = float(absolute[control_region].sum() / absolute.sum())
        examples.append(
            {
                "operation": operation,
                "switch": switch,
                "control": control,
                "control_display_name": control_display_name,
                "control_tanimoto": similarity,
                "control_mass": control_mass,
                "control_source_atoms": control_region,
                "control_analogue_atoms": list(control_change.analogue_modified),
                "switch_map": list(zip(switch_change.parent_match, switch_change.analogue_match)),
                "control_map": list(zip(control_change.parent_match, control_change.analogue_match)),
            }
        )

    frame_width_pt = 247.77
    frame_height_pt = 240.0
    figure = plt.figure(figsize=(18.2, 17.3))
    grid = figure.add_gridspec(
        3,
        7,
        width_ratios=[1.0, 0.30, 1.0, 0.08, 1.0, 0.30, 1.0],
        hspace=0.52,
        wspace=0.08,
        top=0.88,
        bottom=0.09,
        left=0.035,
        right=0.99,
    )
    axes = [[figure.add_subplot(grid[row, col]) for col in range(7)] for row in range(3)]
    audit_rows: List[Dict] = []
    for row_index, example in enumerate(examples):
        switch = example["switch"]
        images = [
            _black_molecule_image(switch.source_smiles, json_list(switch.source_modified_atoms)),
            _black_molecule_image(
                switch.analogue_smiles,
                json_list(switch.analogue_modified_atoms),
                switch.source_smiles,
                example["switch_map"],
            ),
            _black_molecule_image(switch.source_smiles, example["control_source_atoms"]),
            _black_molecule_image(
                example["control"].SMILES,
                example["control_analogue_atoms"],
                switch.source_smiles,
                example["control_map"],
            ),
        ]
        titles = [
            f"{_display_name(switch.source_name)}\n({switch.source_taste} parent)",
            f"{_display_name(switch.analogue_name)}\n({switch.analogue_taste} analogue)",
            f"{_display_name(switch.source_name)}\n({switch.source_taste} parent)",
            f"{_display_name(example['control_display_name'])}\n"
            f"({example['control'].taste} analogue)",
        ]
        molecule_axes = [axes[row_index][column] for column in (0, 2, 4, 6)]
        for axis, image, title, color in zip(
            molecule_axes, images, titles, [RED, BLUE, BLUE, BLUE]
        ):
            allocated = axis.get_position()
            width = frame_width_pt / (figure.get_figwidth() * 72.0)
            height = frame_height_pt / (figure.get_figheight() * 72.0)
            axis.set_position(
                [
                    allocated.x0 + (allocated.width - width) / 2,
                    allocated.y0 + (allocated.height - height) / 2,
                    width,
                    height,
                ]
            )
            axis.imshow(image, aspect="auto")
            axis.set_title(title, fontsize=17, pad=5)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set(color=color, linewidth=1.6)
        for arrow_column in (1, 5):
            axis = axes[row_index][arrow_column]
            axis.text(
                0.5,
                0.60,
                "\N{RIGHTWARDS ARROW}",
                ha="center",
                va="center",
                fontsize=56,
            )
            axis.text(
                0.5,
                0.35,
                "Structure\nmodification",
                ha="center",
                va="center",
                fontsize=16,
            )
            axis.axis("off")
        axes[row_index][3].axis("off")
        for axis, tanimoto, mass in [
            (axes[row_index][2], switch.tanimoto, switch.modified_region_ig_mass),
            (axes[row_index][6], example["control_tanimoto"], example["control_mass"]),
        ]:
            axis.text(
                0.98,
                0.03,
                f"Tanimoto={tanimoto:.2f}\nIG mass={mass:.2f}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=16,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )
        audit_rows.append(
            {
                "operation_category": example["operation"],
                "switch_pair_id": switch.pair_id,
                "shared_parent_row_index": int(switch.source_row_index),
                "control_analogue_row_index": int(example["control"].name),
                "switch_tanimoto": float(switch.tanimoto),
                "control_tanimoto": float(example["control_tanimoto"]),
            }
        )
    for row_index, example in enumerate(examples):
        row_box = axes[row_index][0].get_position()
        figure.text(
            0.5,
            row_box.y0 - 0.025,
            f"{chr(97 + row_index)}. {example['operation'].capitalize()}: "
            "same parent, taste-switching versus non-switching outcome.",
            ha="center",
            va="top",
            fontsize=21,
            fontweight="bold",
        )
    figure.text(0.255, 0.945, "Taste-switching MMP", ha="center", fontsize=24, fontweight="bold")
    figure.text(0.755, 0.945, "Non-switching MMP", ha="center", fontsize=24, fontweight="bold")
    figure.text(0.015, 0.992, "C", ha="left", va="top", fontsize=30, fontweight="bold")
    figure.text(
        0.5,
        0.012,
        "Each row uses one shared parent for both outcomes. Molecules are shown in black; "
        "blue circles mark the locally modified region.",
        ha="center",
        fontsize=18,
    )
    save_figure(figure, stem)
    return pd.DataFrame(audit_rows)


def combine_panels(output_dir: Path) -> None:
    import matplotlib.image as mpimg

    paths = [output_dir / f"Panel{letter}.png" for letter in "ABC"]
    if not all(path.exists() for path in paths):
        return
    setup_plot_style()
    images = [mpimg.imread(path) for path in paths]
    figure = plt.figure(figsize=(18.0, 22.0))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.28],
        height_ratios=[7.4, 14.6],
        hspace=0.015,
        wspace=0.04,
    )
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, :]),
    ]
    for axis, image in zip(axes, images):
        axis.imshow(image)
        axis.axis("off")
    save_figure(figure, output_dir / "Fig_structural_optimization_ABC")


def plot_raincloud(values: pd.DataFrame, statistics: pd.DataFrame, stem: Path) -> None:
    from scipy.stats import gaussian_kde

    setup_plot_style()
    figure, axis = plt.subplots(figsize=(8.8, 7.6))
    rng = np.random.default_rng(SEED)
    positions = [0.0, 0.62]
    for position, group, color in zip(
        positions, ["Taste-switching", "Non-switching"], [RED, BLUE]
    ):
        data = values.loc[values["group"] == group, "ig_mass"].to_numpy(float)
        span = max(float(np.ptp(data)), 1e-6)
        grid = np.linspace(
            max(0.0, float(data.min()) - 0.08 * span),
            min(1.0, float(data.max()) + 0.08 * span),
            256,
        )
        density = gaussian_kde(data)(grid)
        density = density / max(float(density.max()), 1e-12) * 0.22
        axis.fill_betweenx(
            grid, position + 0.015, position + 0.015 + density, color=color, alpha=0.28
        )
        jitter = np.clip(rng.normal(-0.11, 0.026, len(data)), -0.18, -0.045)
        axis.scatter(
            position + jitter,
            data,
            s=25,
            color=color,
            alpha=0.58,
            edgecolor="white",
            linewidth=0.45,
        )
        box = axis.boxplot(
            [data],
            positions=[position],
            widths=0.14,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 2.4},
            whiskerprops={"color": color, "linewidth": 2.0},
            capprops={"color": color, "linewidth": 2.0},
        )
        box["boxes"][0].set(facecolor=color, edgecolor=color, alpha=0.70)
    row = statistics.iloc[0]
    axis.set_xticks(
        positions,
        [
            f"Taste-switching\n(n={int(row.n_switching)})",
            f"Non-switching\n(n={int(row.n_non_switching)})",
        ],
    )
    axis.set_xlim(-0.30, 0.92)
    axis.set_ylim(-0.1, max(0.65, float(values.ig_mass.max()) + 0.04))
    axis.set_ylabel("Contrastive IG mass in modified region")
    axis.text(
        0.98,
        0.97,
        f"Paired permutation P = {float(row.permutation_p):.6f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=18,
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#c7c7c7", linestyle="--", linewidth=0.5, alpha=0.65)
    figure.tight_layout()
    save_figure(figure, stem)


def plot_consistency(values: pd.DataFrame, statistics: pd.DataFrame, stem: Path) -> None:
    setup_plot_style()
    figure, axis = plt.subplots(figsize=(10.8, 7.6))
    x = values["delta_modified_attribution"].to_numpy(float)
    y = values["delta_model_contrast"].to_numpy(float)
    axis.scatter(x, y, s=55, color=BLUE, alpha=0.66, edgecolor="white", linewidth=0.55)
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.linspace(np.quantile(x, 0.01), np.quantile(x, 0.99), 200)
    axis.plot(line_x, intercept + slope * line_x, color="#333333", linewidth=1.4)
    axis.axhline(0, color="#888888", linewidth=0.7, linestyle="--")
    axis.axvline(0, color="#888888", linewidth=0.7, linestyle="--")
    row = statistics.iloc[0]
    axis.text(
        0.02,
        0.97,
        f"Spearman rho={row.spearman_rho:.3f} "
        f"(95% CI {row.bootstrap_ci95_low:.3f} to {row.bootstrap_ci95_high:.3f})\n"
        f"Permutation P={row.permutation_p:.6f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=18,
    )
    axis.set_xlabel("Delta normalized signed attribution in modified region")
    axis.set_ylabel("Delta source-to-target model contrast (logit)")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save_figure(figure, stem)


def run_stage_b(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    pairs = pd.read_csv(output_dir / "stage_A_mmp_pairs.csv")
    tasks = pd.read_csv(output_dir / "stage_A_ig_task_summary.csv").set_index("ig_key")
    switching = pairs[pairs["pair_group"] == "taste-switching"].copy()
    controls = pairs[pairs["pair_group"] == "non-switching"].copy()
    aligned = controls.set_index("matched_switch_pair_id").loc[switching["pair_id"]]
    values = pd.concat(
        [
            pd.DataFrame(
                {
                    "pair_id": switching.pair_id.to_numpy(),
                    "group": "Taste-switching",
                    "ig_mass": switching.modified_region_ig_mass.to_numpy(float),
                }
            ),
            pd.DataFrame(
                {
                    "pair_id": aligned.pair_id.to_numpy(),
                    "group": "Non-switching",
                    "ig_mass": aligned.modified_region_ig_mass.to_numpy(float),
                }
            ),
        ],
        ignore_index=True,
    )
    differences = (
        switching.modified_region_ig_mass.to_numpy(float)
        - aligned.modified_region_ig_mass.to_numpy(float)
    )
    ci_low, ci_high = bootstrap_ci(
        differences, stable_seed("stageB", "raincloud", "ci")
    )
    validation = pd.DataFrame(
        [
            {
                "comparison": "taste-switching vs matched non-switching",
                "n_switching": len(switching),
                "n_non_switching": len(aligned),
                "switching_mean": float(switching.modified_region_ig_mass.mean()),
                "non_switching_mean": float(aligned.modified_region_ig_mass.mean()),
                "paired_mean_difference": float(differences.mean()),
                "bootstrap_ci95_low": ci_low,
                "bootstrap_ci95_high": ci_high,
                "permutation_p": paired_permutation(
                    differences, stable_seed("stageB", "raincloud", "p")
                ),
                "paired_effect_dz": float(
                    differences.mean() / differences.std(ddof=1)
                ),
            }
        ]
    )

    consistency = switching[
        [
            "pair_id",
            "source_taste",
            "analogue_taste",
            "operation_category",
            "source_ig_key",
            "analogue_ig_key",
            "source_modified_ig_signed",
            "analogue_modified_ig_signed",
            "delta_model_contrast",
            "structure_level_held_out",
        ]
    ].copy()
    consistency["source_total_abs_ig"] = consistency.source_ig_key.map(
        tasks.total_abs_ig
    )
    consistency["analogue_total_abs_ig"] = consistency.analogue_ig_key.map(
        tasks.total_abs_ig
    )
    consistency["delta_modified_attribution"] = (
        consistency.source_modified_ig_signed / consistency.source_total_abs_ig
        - consistency.analogue_modified_ig_signed / consistency.analogue_total_abs_ig
    )
    all_consistency = consistency.copy()
    main_consistency = consistency[
        consistency.structure_level_held_out.astype(bool)
    ].copy()
    statistics = pd.DataFrame(
        [
            spearman_statistics(
                main_consistency,
                "structure-level held-out taste-switching MMPs (main analysis)",
            ),
            spearman_statistics(
                all_consistency,
                f"all {len(all_consistency):,} unique taste-switching MMPs "
                "(sensitivity analysis)",
            ),
        ]
    )
    write_csv(values, output_dir / "stage_B_modified_region_masses.csv")
    write_csv(validation, output_dir / "stage_B_validation_statistics.csv")
    write_csv(
        main_consistency, output_dir / "stage_B_attribution_prediction_consistency.csv"
    )
    write_csv(
        all_consistency,
        output_dir / "stage_B_attribution_prediction_consistency_all_sensitivity.csv",
    )
    write_csv(statistics, output_dir / "stage_B_consistency_statistics.csv")
    plot_raincloud(values, validation, output_dir / "PanelA")
    plot_consistency(main_consistency, statistics.iloc[[0]], output_dir / "PanelB")
    combine_panels(output_dir)
    print(statistics.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["A", "B", "AB"], required=True)
    parser.add_argument("--input_csv", type=Path)
    parser.add_argument("--split_manifest", type=Path)
    parser.add_argument("--run_root", type=Path)
    parser.add_argument("--processed_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ig_steps", type=int, default=IG_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rebuild_mmp", action="store_true")
    args = parser.parse_args()
    if "A" in args.stage:
        required = ["input_csv", "split_manifest", "run_root", "processed_dir"]
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"Stage A requires: {', '.join('--' + name for name in missing)}")
    return args


def main() -> None:
    args = parse_args()
    if "A" in args.stage:
        run_stage_a(args)
    if "B" in args.stage:
        run_stage_b(args)


if __name__ == "__main__":
    main()
