from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#8BCF8B",
    "red": "#B64342",
    "teal": "#42949E",
    "neutral": "#CFCECE",
    "ink": "#202124",
    "muted": "#626A73",
    "grid": "#E8EAED",
}


def apply_publication_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 21,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 2.2,
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(values) + 1, dtype=float) / float(len(values))
    return values, y


def threshold_label(threshold: float) -> str:
    return f"{int(round(threshold * 100))}% mass"


def load_threshold_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"mass_threshold", "required_ratio"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["mass_threshold", "required_ratio"]).copy()
    df["required_ratio_pct"] = 100.0 * df["required_ratio"].astype(float)
    return df


def plot_required_ratio_cdf(
    csv_path: Path,
    out_dir: Path,
    *,
    basename: str = "required_ratio_cdf",
) -> List[Path]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_publication_style()
    df = load_threshold_data(csv_path)

    colors: Dict[float, str] = {
        0.80: PALETTE["teal"],
        0.90: PALETTE["blue_main"],
        0.95: PALETTE["red"],
    }
    thresholds = sorted(float(v) for v in df["mass_threshold"].unique())

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    handles: List[Line2D] = []
    p90_stats: Dict[float, float] = {}

    for threshold in thresholds:
        sub = df.loc[np.isclose(df["mass_threshold"].astype(float), threshold), "required_ratio_pct"].to_numpy()
        if sub.size == 0:
            continue

        x, y = ecdf(sub)
        color = colors.get(round(threshold, 2), PALETTE["blue_secondary"])
        p90 = float(np.quantile(sub, 0.90))
        p90_stats[round(threshold, 2)] = p90

        label = threshold_label(threshold)
        ax.plot(x, 100.0 * y, color=color, linewidth=4.0, solid_capstyle="round", label=label)
        handles.append(Line2D([0], [0], color=color, linewidth=4.0, label=label))

    ax.axhline(90.0, color=PALETTE["ink"], linestyle=(0, (5, 5)), linewidth=1.7, alpha=0.35)

    annotation_pos = {
        0.80: (5.0, -14.0),
        0.90: (6.0, -18.0),
        0.95: (7.0, -10.5),
    }
    for threshold, (dx, dy) in annotation_pos.items():
        p90 = p90_stats.get(threshold)
        if p90 is None:
            continue
        color = colors[threshold]
        ax.vlines(p90, 0.0, 90.0, color=color, linewidth=2.0, linestyle=(0, (2, 3)), alpha=0.55)
        ax.scatter([p90], [90.0], s=115, color=color, edgecolor="white", linewidth=2.0, zorder=6)
        ax.annotate(
            f"{p90:.1f}%",
            xy=(p90, 90.0),
            xytext=(p90 + dx, 90.0 + dy),
            color=color,
            fontsize=22,
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 1.9,
                "color": color,
                "shrinkA": 3,
                "shrinkB": 5,
                "mutation_scale": 16,
            },
        )

    ax.set_xlabel("Token ratio (%)", fontsize=23, labelpad=10)
    ax.set_ylabel("Layer-head ratio (%)", fontsize=23, labelpad=10)

    max_visible = max(60.0, float(df["required_ratio_pct"].quantile(0.99)) + 5.0)
    ax.set_xlim(0.0, min(85.0, math.ceil(max_visible / 5.0) * 5.0))
    ax.set_ylim(0.0, 100.5)
    ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1e-9, 10.0))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis="both", labelsize=21)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=1.3)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.9, alpha=0.45)

    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.16),
        fontsize=20,
        handlelength=2.5,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_color(PALETTE["ink"])

    fig.tight_layout(pad=1.4)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        out_dir / f"{basename}.png",
        out_dir / f"{basename}.pdf",
    ]
    fig.savefig(outputs[0], dpi=350)
    fig.savefig(outputs[1])
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a publication-style CDF for oracle top-k required token ratio.")
    parser.add_argument(
        "--threshold-csv",
        type=Path,
        default=Path("/root/AdaptiveIMI/outputs/oracle_topk_sufficiency/oracle_topk_thresholds.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/root/AdaptiveIMI/outputs/oracle_topk_sufficiency/plots"),
    )
    parser.add_argument("--basename", type=str, default="required_ratio_cdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = plot_required_ratio_cdf(args.threshold_csv, args.out_dir, basename=args.basename)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
