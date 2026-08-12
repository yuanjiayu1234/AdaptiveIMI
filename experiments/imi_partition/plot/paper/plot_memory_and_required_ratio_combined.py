from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.imi_partition.plot.paper.plot_kv_cache_memory import (
    apply_publication_style,
    plot_kv_cache_memory_on_ax,
)
from experiments.imi_partition.plot.paper.plot_required_ratio_cdf_paper import PALETTE as CDF_PALETTE
from experiments.imi_partition.plot.paper.plot_required_ratio_cdf_paper import (
    ecdf,
    load_threshold_data,
    threshold_label,
)


FONT_FAMILY = "DejaVu Sans"
COMPACT_FONT_SIZE = 22


def plot_required_ratio_cdf_on_ax(ax, threshold_csv: Path, *, compact: bool = True) -> None:
    df = load_threshold_data(threshold_csv)
    colors: Dict[float, str] = {
        0.80: CDF_PALETTE["teal"],
        0.90: CDF_PALETTE["blue_main"],
        0.95: CDF_PALETTE["red"],
    }
    thresholds = sorted(float(v) for v in df["mass_threshold"].unique())
    handles: List[Line2D] = []
    p90_stats: Dict[float, float] = {}

    for threshold in thresholds:
        sub = df.loc[np.isclose(df["mass_threshold"].astype(float), threshold), "required_ratio_pct"].to_numpy()
        if sub.size == 0:
            continue
        x, y = ecdf(sub)
        color = colors.get(round(threshold, 2), CDF_PALETTE["blue_secondary"])
        p90 = float(np.quantile(sub, 0.90))
        p90_stats[round(threshold, 2)] = p90
        ax.plot(x, 100.0 * y, color=color, linewidth=3.2, solid_capstyle="round", label=threshold_label(threshold))
        handles.append(Line2D([0], [0], color=color, linewidth=3.2, label=threshold_label(threshold)))

    ax.axhline(90.0, color=CDF_PALETTE["ink"], linestyle=(0, (5, 5)), linewidth=1.4, alpha=0.35)
    annotation_pos = {
        0.80: (4.4, -14.0),
        0.90: (5.2, -18.0),
        0.95: (5.8, -10.5),
    }
    for threshold, (dx, dy) in annotation_pos.items():
        p90 = p90_stats.get(threshold)
        if p90 is None:
            continue
        color = colors[threshold]
        ax.vlines(p90, 0.0, 90.0, color=color, linewidth=1.7, linestyle=(0, (2, 3)), alpha=0.55)
        ax.scatter([p90], [90.0], s=78, color=color, edgecolor="white", linewidth=1.7, zorder=6)
        ax.annotate(
            f"{p90:.1f}%",
            xy=(p90, 90.0),
            xytext=(p90 + dx, 90.0 + dy),
            color=color,
            fontsize=COMPACT_FONT_SIZE if compact else COMPACT_FONT_SIZE,
            fontweight="bold",
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 1.5,
                "color": color,
                "shrinkA": 3,
                "shrinkB": 5,
                "mutation_scale": 13,
            },
        )

    ax.set_xlabel("Token ratio (%)", fontsize=COMPACT_FONT_SIZE if compact else 23, labelpad=8)
    ax.set_ylabel("Layer-head ratio (%)", fontsize=COMPACT_FONT_SIZE if compact else 23, labelpad=8)
    max_visible = max(60.0, float(df["required_ratio_pct"].quantile(0.99)) + 5.0)
    ax.set_xlim(0.0, min(85.0, math.ceil(max_visible / 5.0) * 5.0))
    ax.set_ylim(0.0, 100.5)
    ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 1e-9, 10.0))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis="both", labelsize=COMPACT_FONT_SIZE if compact else 21)
    ax.grid(axis="y", color=CDF_PALETTE["grid"], linewidth=1.0)
    ax.grid(axis="x", color=CDF_PALETTE["grid"], linewidth=0.8, alpha=0.45)
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        prop={"family": FONT_FAMILY, "size": COMPACT_FONT_SIZE if compact else 14},
        ncol=2,
        columnspacing=0.85,
        handlelength=2.2,
        borderaxespad=0.0,
        frameon=False,
    )


def plot_combined(threshold_csv: Path, out_dir: Path, *, basename: str = "memory_and_required_ratio") -> List[Path]:
    apply_publication_style()
    plt.rcParams.update({
        "font.family": [FONT_FAMILY, "sans-serif"],
        "mathtext.fontset": "dejavusans",
    })
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.4, 5.9),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.24},
    )

    plot_kv_cache_memory_on_ax(axes[0], legend=True, legend_top=True, compact=True)
    plot_required_ratio_cdf_on_ax(axes[1], threshold_csv, compact=True)

    axes[0].text(
        0.5,
        -0.31,
        "(a) Model + KV Cache Memory",
        transform=axes[0].transAxes,
        fontsize=COMPACT_FONT_SIZE,
        fontweight="bold",
        ha="center",
        va="top",
    )
    axes[1].text(
        0.5,
        -0.31,
        "(b) Required Token Ratio",
        transform=axes[1].transAxes,
        fontsize=COMPACT_FONT_SIZE,
        fontweight="bold",
        ha="center",
        va="top",
    )

    fig.subplots_adjust(left=0.070, right=0.985, bottom=0.25, top=0.78, wspace=0.24)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [out_dir / f"{basename}.pdf", out_dir / f"{basename}.png"]
    fig.savefig(outputs[0])
    fig.savefig(outputs[1], dpi=350)
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine total model-plus-KV-cache memory and oracle top-k required-ratio figures.")
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
    parser.add_argument("--basename", type=str, default="memory_and_required_ratio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in plot_combined(args.threshold_csv, args.out_dir, basename=args.basename):
        print(path)


if __name__ == "__main__":
    main()
