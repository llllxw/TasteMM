#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps



def choose_score_column(atom_df: pd.DataFrame, requested: str = "") -> str:
    if requested:
        if requested not in atom_df.columns:
            raise ValueError(f"Requested score column {requested!r} is not present in the atom table.")
        return requested
    for candidate in ("attribution_score_signed", "attribution_score", "attribution_score_abs"):
        if candidate in atom_df.columns:
            return candidate
    raise ValueError("No supported attribution score column was found.")


def load_data(csv_path: str, structure_map: str, smiles_col: str, mol_id_col: str, mol_name_col: str):
    atom_df = pd.read_csv(csv_path)
    structure_df = pd.read_csv(structure_map)
    required_atom = {mol_id_col, "atom_index"}
    required_structure = {mol_id_col, smiles_col}
    if not required_atom.issubset(atom_df.columns):
        raise ValueError(f"Atom table must contain {sorted(required_atom)}.")
    if not required_structure.issubset(structure_df.columns):
        raise ValueError(f"Selected-sample table must contain {sorted(required_structure)}.")
    if mol_name_col not in structure_df.columns:
        structure_df[mol_name_col] = structure_df[mol_id_col].astype(str)
    return atom_df, structure_df, {"structure_source": structure_map}


def prepare_render_rows(atom_df: pd.DataFrame, structure_df: pd.DataFrame):
    rows = []
    skipped = []
    for _, structure in structure_df.iterrows():
        molecule_id = structure["id"]
        if "row_index" in structure_df.columns and "row_index" in atom_df.columns:
            atom_rows = atom_df[atom_df["row_index"].astype(int) == int(structure["row_index"])].copy()
        else:
            if structure_df["id"].astype(str).duplicated().any():
                raise ValueError("Duplicate IDs require row_index columns in both attribution inputs.")
            atom_rows = atom_df[atom_df["id"].astype(str) == str(molecule_id)].copy()
        smiles = str(structure["smiles"])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or atom_rows.empty or len(atom_rows) != mol.GetNumAtoms():
            skipped.append(str(molecule_id))
            continue
        rows.append(
            {
                "id": molecule_id,
                "name": structure.get("name", molecule_id),
                "mol": mol,
                "atom_rows": atom_rows,
                "target_label_name": structure.get("target_label_name", ""),
                "true_label_name": structure.get("true_label_name", ""),
            }
        )
    if not rows:
        raise ValueError("No valid molecule-attribution rows were available for rendering.")
    return rows, skipped


@dataclass
class SimilarityStyleConfig:
    base_size: int = 760
    dpi: int = 320
    figure_facecolor: str = "#FFFFFF"
    panel_cols: int = 3
    padding: float = 0.10
    bond_line_width: float = 2.4
    max_font_size: int = 120
    low_q: float = 0.05
    high_q: float = 0.95
    panel_label_size: int = 17
    panel_id_size: int = 14
    colorbar_label: str = "Standardized atom attribution"
    colorbar_tick_size: int = 12
    colorbar_label_size: int = 13


def calculate_aspect_ratio(molecule: Chem.Mol, base_size: int) -> tuple[int, int]:
    conf = molecule.GetConformer()
    atom_positions = [conf.GetAtomPosition(i) for i in range(molecule.GetNumAtoms())]
    x_coords = [pos.x for pos in atom_positions]
    y_coords = [pos.y for pos in atom_positions]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    aspect_ratio = width / height if height > 0 else 1.0
    canvas_width = max(base_size, int(base_size * aspect_ratio)) if aspect_ratio > 1 else base_size
    canvas_height = max(base_size, int(base_size / aspect_ratio)) if aspect_ratio < 1 else base_size
    return canvas_width, canvas_height


def robust_scale(values: np.ndarray, low_q: float, high_q: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros_like(arr)
    if np.allclose(arr.max(), arr.min()):
        return np.ones_like(arr) if arr.max() > 0 else np.zeros_like(arr)
    low = float(np.quantile(arr, low_q))
    high = float(np.quantile(arr, high_q))
    if high - low < 1e-8:
        low = float(arr.min())
        high = float(arr.max())
    if high - low < 1e-8:
        return np.ones_like(arr) if high > 0 else np.zeros_like(arr)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def build_template_weights(scores: np.ndarray, config: SimilarityStyleConfig) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    standardized, _ = SimilarityMaps.GetStandardizedWeights(scores.tolist())
    standardized = np.asarray(standardized, dtype=float)
    return standardized


def render_template_style_molecule(mol: Chem.Mol, atom_scores: np.ndarray, config: SimilarityStyleConfig) -> Image.Image:
    molecule = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(molecule)
    width, height = calculate_aspect_ratio(molecule, config.base_size)
    drawer = Draw.MolDraw2DCairo(width, height)
    dopts = drawer.drawOptions()
    dopts.padding = config.padding
    dopts.maxFontSize = config.max_font_size
    dopts.bondLineWidth = config.bond_line_width
    dopts.useBWAtomPalette()
    weights = build_template_weights(atom_scores, config=config)
    SimilarityMaps.GetSimilarityMapFromWeights(molecule, list(map(float, weights)), draw2d=drawer)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")


def render_panel(panel_rows: list[dict[str, Any]], output_dir: Path, score_col: str, config: SimilarityStyleConfig) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_panels = len(panel_rows)
    ncols = max(1, min(config.panel_cols, n_panels))
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.9 * ncols, 5.1 * nrows), facecolor=config.figure_facecolor)
    axes = np.atleast_1d(axes).ravel()

    manifest: list[dict[str, Any]] = []
    for panel_index, (ax, row) in enumerate(zip(axes, panel_rows), start=1):
        atom_scores = row["atom_rows"].sort_values("atom_index")[score_col].to_numpy(dtype=float)
        rendered = render_template_style_molecule(row["mol"], atom_scores, config=config)
        ax.axis("off")
        ax.imshow(rendered)
        ax.text(
            0.5,
            -0.020,
            str(row["target_label_name"] or row["true_label_name"] or "unknown"),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=config.panel_label_size,
            fontweight="bold",
            color="#222222",
        )
        ax.text(
            0.5,
            -0.090,
            str(row["id"]),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=config.panel_id_size,
            color="#222222",
        )
        manifest.append(
            {
                "panel_index": panel_index,
                "id": row["id"],
                "name": row["name"],
                "label": row["target_label_name"] or row["true_label_name"],
            }
        )

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.99, top=0.985, bottom=0.16, wspace=0.03, hspace=0.16)
    sm = cm.ScalarMappable(norm=Normalize(vmin=-1.0, vmax=1.0), cmap=cm.PiYG)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes[:n_panels].tolist(),
        orientation="horizontal",
        fraction=0.055,
        pad=0.12,
        aspect=28,
    )
    cbar.set_label(config.colorbar_label, fontsize=config.colorbar_label_size)
    cbar.ax.tick_params(labelsize=config.colorbar_tick_size)
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    png_path = output_dir / "similarity_style_panel.png"
    pdf_path = output_dir / "similarity_style_panel.pdf"
    svg_path = output_dir / "similarity_style_panel.svg"
    fig.savefig(png_path, dpi=config.dpi, bbox_inches="tight", facecolor=config.figure_facecolor)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=config.figure_facecolor)
    fig.savefig(svg_path, bbox_inches="tight", facecolor=config.figure_facecolor)
    plt.close(fig)

    (output_dir / "panel_manifest.csv").write_text(pd.DataFrame(manifest).to_csv(index=False), encoding="utf-8")
    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "svg": str(svg_path),
        "manifest": str(output_dir / "panel_manifest.csv"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render gradient-attribution panels with RDKit SimilarityMaps."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--selected-csv",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
    )
    parser.add_argument("--score-col", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimilarityStyleConfig()

    atom_df, structure_df, metadata = load_data(
        csv_path=args.csv,
        structure_map=args.selected_csv,
        smiles_col="smiles",
        mol_id_col="id",
        mol_name_col="name",
    )
    score_col = choose_score_column(atom_df, requested=args.score_col)
    panel_rows, skipped = prepare_render_rows(atom_df=atom_df, structure_df=structure_df)
    for row in panel_rows:
        atom_rows = row["atom_rows"]
        if "pred_label" in atom_rows.columns:
            row["_model_label"] = int(atom_rows["pred_label"].iloc[0]) + 1
        elif "true_label" in atom_rows.columns:
            row["_model_label"] = int(atom_rows["true_label"].iloc[0]) + 1
        else:
            row["_model_label"] = 999
    panel_rows.sort(key=lambda row: (row["_model_label"], str(row["id"])))
    output_dir = Path(args.outdir)
    outputs = render_panel(panel_rows, output_dir=output_dir, score_col=score_col, config=config)

    meta = {
        "rendering_note": (
            "This rendering uses the public RDKit workflow: "
            "MolFromSmiles -> Compute2DCoords -> MolDraw2DCairo -> useBWAtomPalette -> "
            "SimilarityMaps.GetSimilarityMapFromWeights(...). "
            "The current version keeps all atom contributions visible and sorts panels by model label 1..6."
        ),
        "source_atom_csv": args.csv,
        "source_structure_csv": args.selected_csv,
        "score_col": score_col,
        "skipped": skipped,
        "outputs": outputs,
        "config": asdict(config),
        "structure_source": metadata.get("structure_source", ""),
    }
    (output_dir / "render_log.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] notebook-template panel saved to {outputs['png']}")


if __name__ == "__main__":
    main()
