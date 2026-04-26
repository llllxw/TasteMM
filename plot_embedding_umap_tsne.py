from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_NAMES = ["bitter", "sweet", "umami", "salty", "sour", "tasteless"]
CLASS_COLORS = {
    "bitter": "#84c3b7",
    "sweet": "#f57c6e",
    "umami": "#fae69e",
    "salty": "#1F4E79",
    "sour": "#f2b56f",
    "tasteless": "#b8aeeb",
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Plot Figure 5A/B chemical-space projections from exported embeddings."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="Directory produced by export_embedding_table.py",
    )
    parser.add_argument(
        "--coordinates_csv",
        type=str,
        default="",
        help="Optional precomputed coordinates CSV. If provided, skips projection and replots directly.",
    )
    parser.add_argument(
        "--embedding_key",
        type=str,
        default="fused_embeddings",
        choices=["fused_embeddings", "graph_embeddings"],
        help="Which embedding matrix to project.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "umap", "tsne"],
        help="Projection method. 'auto' prefers UMAP and falls back to t-SNE.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. Defaults to <input_dir>/plots",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=35.0, help="Used only for t-SNE.")
    parser.add_argument(
        "--font_scale",
        type=float,
        default=1.0,
        help="Global font scaling factor applied to all plot text.",
    )
    parser.add_argument(
        "--panel",
        type=str,
        default="all",
        choices=["all", "true_class", "correctness", "confidence"],
        help="Which panel to export. 'all' keeps the original 3-panel layout.",
    )
    return parser.parse_args()


def setup_matplotlib(font_scale: float = 1.0):
    def scaled(size: float) -> float:
        return size * font_scale

    plt.style.use("default")
    plt.rcParams["font.family"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["font.size"] = scaled(12)
    plt.rcParams["axes.titlesize"] = scaled(13)
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = scaled(12)
    plt.rcParams["xtick.labelsize"] = scaled(11)
    plt.rcParams["ytick.labelsize"] = scaled(11)
    plt.rcParams["legend.fontsize"] = scaled(10)
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def save_figure_all_formats(fig, output_dir: str, stem: str, dpi: int):
    fig.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{stem}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{stem}.svg"), bbox_inches="tight")


def load_inputs(input_dir: str, embedding_key: str):
    summary_path = os.path.join(input_dir, "test_embedding_summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    metadata = pd.read_csv(os.path.join(input_dir, "test_embedding_metadata.csv"))
    arrays = np.load(os.path.join(input_dir, "test_embeddings.npz"))
    if embedding_key not in arrays:
        raise KeyError(f"{embedding_key} not found in NPZ. Available keys: {list(arrays.keys())}")
    emb = arrays[embedding_key]
    if len(metadata) != emb.shape[0]:
        raise ValueError(f"Metadata rows {len(metadata)} != embedding rows {emb.shape[0]}")
    return metadata, emb, summary


def project_embeddings(emb: np.ndarray, method: str, random_state: int, perplexity: float):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    x = StandardScaler().fit_transform(emb)
    used_method = method

    if method in {"auto", "umap"}:
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=25,
                min_dist=0.15,
                metric="euclidean",
                random_state=random_state,
            )
            coords = reducer.fit_transform(x)
            return coords, "umap"
        except Exception:
            if method == "umap":
                raise
            used_method = "tsne"

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    coords = tsne.fit_transform(x)
    return coords, used_method


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def plot_true_class(ax, plot_df: pd.DataFrame):
    for class_name in CLASS_NAMES:
        sub = plot_df[plot_df["true_label_name"] == class_name]
        if sub.empty:
            continue
        ax.scatter(
            sub["proj_x"],
            sub["proj_y"],
            s=14,
            alpha=0.72,
            c=CLASS_COLORS[class_name],
            edgecolors="none",
            label=class_name,
        )
    ax.set_title("Chemical space by true taste class")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.8)


def plot_correctness(ax, plot_df: pd.DataFrame):
    correct = plot_df[plot_df["correct"] == 1]
    incorrect = plot_df[plot_df["correct"] == 0]
    ax.scatter(
        correct["proj_x"],
        correct["proj_y"],
        s=12,
        alpha=0.35,
        c="#9BBCE0",
        edgecolors="none",
        label=f"Correct (n={len(correct)})",
    )
    ax.scatter(
        incorrect["proj_x"],
        incorrect["proj_y"],
        s=18,
        alpha=0.88,
        c="#C44E52",
        edgecolors="white",
        linewidths=0.25,
        label=f"Incorrect (n={len(incorrect)})",
    )
    ax.set_title("Same projection by prediction correctness")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", frameon=False)


def plot_confidence(ax, plot_df: pd.DataFrame):
    sc = ax.scatter(
        plot_df["proj_x"],
        plot_df["proj_y"],
        c=plot_df["conf_ts_maxprob"],
        cmap="Blues",
        s=14,
        alpha=0.78,
        edgecolors="none",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title("Same projection by calibrated confidence")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    return sc


def infer_used_method(args, coordinates_csv: str) -> str:
    if args.method != "auto":
        return args.method
    filename = os.path.basename(coordinates_csv).lower()
    if "umap" in filename:
        return "umap"
    if "tsne" in filename:
        return "tsne"
    return "precomputed"


def build_panel_stem(panel: str, embedding_key: str, used_method: str, font_scale: float) -> str:
    if panel == "all":
        base = f"Figure5AB_{embedding_key}_{used_method}"
    elif panel == "true_class":
        base = f"Figure5a_true_class_{embedding_key}_{used_method}"
    elif panel == "correctness":
        base = f"Figure5b_correctness_{embedding_key}_{used_method}"
    else:
        base = f"Figure5c_confidence_{embedding_key}_{used_method}"
    if font_scale != 1.0:
        base += f"_font{font_scale:g}x"
    return base


def main():
    args = get_args()
    setup_matplotlib(args.font_scale)

    if not args.input_dir and not args.coordinates_csv:
        raise ValueError("Provide either --input_dir to compute/load embeddings or --coordinates_csv to replot.")

    default_base_dir = args.input_dir or os.path.dirname(args.coordinates_csv)
    output_dir = args.output_dir or os.path.join(default_base_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    if args.coordinates_csv:
        plot_df = pd.read_csv(args.coordinates_csv)
        summary = {}
        used_method = infer_used_method(args, args.coordinates_csv)
    else:
        metadata, emb, summary = load_inputs(args.input_dir, args.embedding_key)
        coords, used_method = project_embeddings(
            emb=emb,
            method=args.method,
            random_state=args.random_state,
            perplexity=args.perplexity,
        )

        plot_df = metadata.copy()
        plot_df["proj_x"] = coords[:, 0]
        plot_df["proj_y"] = coords[:, 1]
        plot_df.to_csv(
            os.path.join(output_dir, f"{args.embedding_key}_{used_method}_coordinates.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    if args.panel == "all":
        fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.4))
        for ax in axes:
            style_axis(ax)

        plot_true_class(axes[0], plot_df)
        plot_correctness(axes[1], plot_df)
        sc = plot_confidence(axes[2], plot_df)
        cbar = fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.03)
        cbar.set_label("Calibrated confidence")

        emb_label = "Fused embedding" if args.embedding_key == "fused_embeddings" else "Graph embedding"
        method_label = "UMAP" if used_method == "umap" else ("t-SNE" if used_method == "tsne" else used_method)
        fig.suptitle(f"{method_label} projection of {emb_label.lower()}s", y=0.98, fontsize=15 * args.font_scale, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        stem = build_panel_stem(args.panel, args.embedding_key, used_method, args.font_scale)
        save_figure_all_formats(fig, output_dir, stem, args.dpi)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        style_axis(ax)
        if args.panel == "true_class":
            plot_true_class(ax, plot_df)
        elif args.panel == "correctness":
            plot_correctness(ax, plot_df)
        else:
            sc = plot_confidence(ax, plot_df)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label("Calibrated confidence")

        fig.tight_layout()
        stem = build_panel_stem(args.panel, args.embedding_key, used_method, args.font_scale)
        save_figure_all_formats(fig, output_dir, stem, args.dpi)
        plt.close(fig)

    log = {
        "input_dir": args.input_dir,
        "coordinates_csv": args.coordinates_csv,
        "embedding_key": args.embedding_key,
        "requested_method": args.method,
        "used_method": used_method,
        "panel": args.panel,
        "n_samples": int(len(plot_df)),
        "class_order": summary.get("class_order", CLASS_NAMES),
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, f"{stem}_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved plots to: {output_dir}")
    print(f"[INFO] Used method: {used_method}")
    print(f"[INFO] Embedding key: {args.embedding_key}")


if __name__ == "__main__":
    main()
