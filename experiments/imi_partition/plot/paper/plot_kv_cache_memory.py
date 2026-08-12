from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "blue_main": "#0F4D92",
    "red": "#B64342",
    "teal": "#42949E",
    "green": "#8BCF8B",
    "ink": "#202124",
    "grid": "#E8EAED",
}

MODEL_CONFIGS: Dict[str, Dict[str, float | str]] = {
    "Qwen3-8B": {
        "num_layers": 36,
        "num_kv_heads": 8,
        "head_dim": 128,
        "model_size_gb": 16.0,
        "color": PALETTE["blue_main"],
        "marker": "o",
    },
    "Llama3-8B": {
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "model_size_gb": 16.0,
        "color": PALETTE["red"],
        "marker": "s",
    },
}

GPU_MEMORY_GB = {
    "A100 40GB": 40,
}

CONTEXT_LENGTHS = np.array([4, 64, 128, 256, 512], dtype=np.int64) * 1024


def apply_publication_style() -> None:
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 2.0,
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def calculate_kv_cache_size_gb(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    context_length: int,
    precision_bytes: int = 2,
) -> float:
    size_bytes = 2 * int(num_layers) * int(num_kv_heads) * int(head_dim) * int(context_length) * int(precision_bytes)
    return float(size_bytes) / float(1024 ** 3)


def total_memory_curve(config: Dict[str, float | str], precision_bytes: int) -> List[float]:
    model_size = float(config["model_size_gb"]) * (float(precision_bytes) / 2.0)
    return [
        model_size
        + calculate_kv_cache_size_gb(
            num_layers=int(config["num_layers"]),
            num_kv_heads=int(config["num_kv_heads"]),
            head_dim=int(config["head_dim"]),
            context_length=int(ctx_len),
            precision_bytes=int(precision_bytes),
        )
        for ctx_len in CONTEXT_LENGTHS
    ]


def plot_kv_cache_memory_on_ax(
    ax,
    *,
    legend: bool = True,
    legend_top: bool = True,
    compact: bool = False,
) -> None:
    x = CONTEXT_LENGTHS / 1024
    line_width = 3.6 if compact else 3.8
    marker_size = 7.2 if compact else 8.0

    for model_name, config in MODEL_CONFIGS.items():
        color = str(config["color"])
        marker = str(config["marker"])
        ax.plot(
            x,
            total_memory_curve(config, precision_bytes=2),
            label=f"{model_name} BF16/FP16",
            linewidth=line_width,
            marker=marker,
            markersize=marker_size,
            color=color,
            linestyle="-",
            alpha=0.95,
        )
        ax.plot(
            x,
            total_memory_curve(config, precision_bytes=4),
            label=f"{model_name} FP32",
            linewidth=line_width,
            marker=marker,
            markersize=marker_size,
            color=color,
            linestyle="--",
            alpha=0.70,
        )

    gpu_colors = {
        "A100 40GB": PALETTE["teal"],
    }
    for gpu_name, memory_gb in GPU_MEMORY_GB.items():
        ax.axhline(
            y=float(memory_gb),
            color=gpu_colors[gpu_name],
            linestyle=(0, (2, 2)),
            linewidth=2.3 if compact else 2.8,
            alpha=0.85,
        )
        ax.text(
            505,
            float(memory_gb) + 1.0,
            gpu_name,
            color=gpu_colors[gpu_name],
            fontsize=22 if compact else 20,
            fontweight="bold",
            ha="right",
            va="bottom",
        )

    ax.set_xlabel("Context length", fontsize=22 if compact else 23, labelpad=8)
    ax.set_ylabel("Total memory (GB)", fontsize=22 if compact else 23, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}K" for v in x])
    ax.set_ylim(10, 185)
    ax.grid(True, axis="both", alpha=0.55, linestyle="-", linewidth=1.0, color=PALETTE["grid"])
    ax.tick_params(axis="both", labelsize=22 if compact else 20)

    if legend:
        if legend_top:
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.03),
                fontsize=22 if compact else 14,
                ncol=2,
                columnspacing=0.85,
                handlelength=1.8,
                borderaxespad=0.0,
            )
        else:
            ax.legend(loc="upper left", fontsize=12 if compact else 15)


def print_memory_table() -> None:
    print("=" * 80)
    print("Memory Analysis for Long-Context LLM Inference")
    print("=" * 80)
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n{model_name}:")
        print(f"  Model Size (BF16/FP16): {float(config['model_size_gb']):.2f} GB")
        print(f"  Architecture: {int(config['num_layers'])} layers, {int(config['num_kv_heads'])} KV heads, {int(config['head_dim'])} head dim")
        print(f"  {'Length':<10} {'BF16/FP16 total':<18} {'BF16/FP16 KV':<16} {'FP32 total':<14} {'FP32 KV':<10}")
        for ctx_len in CONTEXT_LENGTHS:
            kv_fp16 = calculate_kv_cache_size_gb(
                num_layers=int(config["num_layers"]),
                num_kv_heads=int(config["num_kv_heads"]),
                head_dim=int(config["head_dim"]),
                context_length=int(ctx_len),
                precision_bytes=2,
            )
            kv_fp32 = calculate_kv_cache_size_gb(
                num_layers=int(config["num_layers"]),
                num_kv_heads=int(config["num_kv_heads"]),
                head_dim=int(config["head_dim"]),
                context_length=int(ctx_len),
                precision_bytes=4,
            )
            print(
                f"  {ctx_len // 1024:3d}K      "
                f"{kv_fp16 + float(config['model_size_gb']):8.2f} GB       "
                f"{kv_fp16:8.2f} GB      "
                f"{kv_fp32 + float(config['model_size_gb']) * 2.0:8.2f} GB   "
                f"{kv_fp32:8.2f} GB"
            )


def plot_kv_cache_memory(out_dir: Path, *, basename: str = "kv_cache_memory_analysis") -> List[Path]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    plot_kv_cache_memory_on_ax(ax, legend=True, legend_top=True, compact=False)
    fig.tight_layout(pad=1.1)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [out_dir / f"{basename}.pdf", out_dir / f"{basename}.png"]
    fig.savefig(outputs[0])
    fig.savefig(outputs[1], dpi=350)
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot total model + KV-cache memory for Qwen3-8B and Llama3-8B.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/root/AdaptiveIMI/outputs/oracle_topk_sufficiency/plots"),
    )
    parser.add_argument("--basename", type=str, default="kv_cache_memory_analysis")
    parser.add_argument("--no-table", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = plot_kv_cache_memory(args.out_dir, basename=args.basename)
    for path in outputs:
        print(path)
    if not args.no_table:
        print_memory_table()


if __name__ == "__main__":
    main()
