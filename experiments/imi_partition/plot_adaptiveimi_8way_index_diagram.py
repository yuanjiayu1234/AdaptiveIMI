from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green": "#8BCF8B",
    "green_light": "#DDF3DE",
    "red": "#B64342",
    "red_light": "#F6CFCB",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "yellow": "#F2C14E",
    "neutral": "#CFCECE",
    "neutral_light": "#F4F6F8",
    "ink": "#202124",
    "muted": "#626A73",
    "grid": "#E8EAED",
    "white": "#FFFFFF",
}

WAY_COLORS = [
    "#0F4D92",
    "#3775BA",
    "#42949E",
    "#8BCF8B",
    "#F2C14E",
    "#E9A6A1",
    "#B64342",
    "#9A4D8E",
]


def apply_publication_style() -> None:
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 18,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def label(ax, x: float, y: float, text: str, *, size: int = 17, weight: str = "normal",
          color: str = PALETTE["ink"], ha: str = "center", va: str = "center") -> None:
    ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va=va)


def shade(color: str, factor: float) -> str:
    color = color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    r = int(max(0, min(255, r * factor)))
    g = int(max(0, min(255, g * factor)))
    b = int(max(0, min(255, b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


def arrow(ax, start: Tuple[float, float], end: Tuple[float, float], *, color: str = PALETTE["ink"], lw: float = 2.7):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=lw,
        color=color,
        shrinkA=7,
        shrinkB=7,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(patch)
    return patch


def rounded_box(ax, xy: Tuple[float, float], w: float, h: float, *, face: str, edge: str,
                lw: float = 1.8, radius: float = 0.12, alpha: float = 1.0) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def prism(ax, x: float, y: float, w: float, h: float, *, dx: float = 0.28, dy: float = 0.20,
          face: str = PALETTE["blue_secondary"], edge: str = PALETTE["ink"], lw: float = 1.3,
          alpha: float = 1.0) -> None:
    front = Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha)
    top = Polygon(
        [(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True,
        facecolor=shade(face, 1.16),
        edgecolor=edge,
        linewidth=lw,
        alpha=alpha,
    )
    side = Polygon(
        [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
        closed=True,
        facecolor=shade(face, 0.78),
        edgecolor=edge,
        linewidth=lw,
        alpha=alpha,
    )
    ax.add_patch(top)
    ax.add_patch(side)
    ax.add_patch(front)


def stage_header(ax, x: float, y: float, text: str, w: float) -> None:
    rounded_box(ax, (x, y), w, 0.48, face=PALETTE["ink"], edge=PALETTE["ink"], lw=0, radius=0.16)
    label(ax, x + w / 2, y + 0.24, text, size=18, weight="bold", color=PALETTE["white"])


def draw_kv_tensor(ax) -> None:
    x, y = 0.65, 3.05
    for i in range(5):
        prism(ax, x + 0.10 * i, y + 0.10 * i, 2.45, 1.0, dx=0.34, dy=0.22,
              face=shade(PALETTE["blue_secondary"], 1.0 + 0.04 * i), edge=PALETTE["blue_main"], lw=1.1, alpha=0.96)
    label(ax, x + 1.48, y + 1.72, "RoPE-applied K/V", size=18, weight="bold")
    label(ax, x + 1.48, y - 0.28, "[batch, KV head, token, dim]", size=14, color=PALETTE["muted"])

    rounded_box(ax, (0.78, 1.00), 2.55, 1.35, face=PALETTE["neutral_light"], edge=PALETTE["muted"], lw=1.5)
    label(ax, 2.05, 2.05, "learned 8-way\nRoPE-pair partition", size=16, weight="bold")
    label(ax, 1.05, 1.52, "per layer / KV head", size=13, color=PALETTE["muted"], ha="left")
    label(ax, 1.05, 1.20, "runtime k-values", size=13, color=PALETTE["muted"], ha="left")


def draw_eight_way(ax) -> None:
    label(ax, 5.55, 5.35, "split each key into 8 ways", size=18, weight="bold")
    x0, y0 = 4.25, 3.85
    w_total, h = 2.8, 0.82
    seg_w = w_total / 8
    for i, color in enumerate(WAY_COLORS):
        prism(ax, x0 + i * seg_w, y0, seg_w - 0.03, h, dx=0.09, dy=0.11, face=color, edge="white", lw=0.8)
        label(ax, x0 + i * seg_w + seg_w / 2, y0 - 0.25, str(i + 1), size=12, color=color, weight="bold")
    label(ax, x0 + w_total / 2, y0 + 1.18, "one key vector", size=14, color=PALETTE["muted"])

    lane_x, lane_y = 4.28, 1.05
    for i, color in enumerate(WAY_COLORS):
        yy = lane_y + (7 - i) * 0.30
        rounded_box(ax, (lane_x, yy), 2.72, 0.22, face=shade(color, 1.28), edge=color, lw=1.2, radius=0.055)
        label(ax, lane_x + 0.18, yy + 0.11, f"way {i + 1}", size=10.5, color=PALETTE["ink"], ha="left")
        label(ax, lane_x + 2.52, yy + 0.11, f"l{i + 1}", size=11.5, color=color, ha="right", weight="bold")
    label(ax, 5.62, 0.78, "8 independent k-means label streams", size=14, color=PALETTE["muted"])


def draw_cells(ax) -> None:
    rounded_box(ax, (8.10, 3.55), 3.30, 1.15, face=PALETTE["white"], edge=PALETTE["teal"], lw=2.0)
    label(ax, 9.75, 4.42, "pack 8 labels into a cell id", size=17, weight="bold")
    label(ax, 9.75, 3.94, r"cell($K_i$) = ($l_1, ..., l_8$)", size=19)

    label(ax, 9.78, 2.92, "group tokens by non-empty cells", size=18, weight="bold")
    base_x, base_y = 8.16, 1.05
    cells = [
        (0.00, 0.00, 0.58, 0.96, PALETTE["blue_secondary"]),
        (0.72, 0.16, 0.58, 0.68, PALETTE["teal"]),
        (1.36, 0.00, 0.58, 1.16, PALETTE["red"]),
        (2.08, 0.12, 0.58, 0.76, PALETTE["violet"]),
    ]
    for j, (dx, dy, w, h, color) in enumerate(cells):
        prism(ax, base_x + dx, base_y + dy, w, h, dx=0.20, dy=0.15, face=color, edge="white", lw=1.0)
        label(ax, base_x + dx + w / 2, base_y + dy - 0.18, f"cell {chr(65 + j)}", size=11.5, color=PALETTE["muted"])
    label(ax, 9.76, 0.48, "sort by cell id; keep occupied cells only", size=14, color=PALETTE["muted"])


def draw_flat_index(ax) -> None:
    x, y = 13.0, 1.02
    prism(ax, x + 0.20, y + 2.85, 3.35, 1.15, dx=0.30, dy=0.22,
          face=PALETTE["green_light"], edge=PALETTE["green"], lw=1.8)
    label(ax, x + 1.88, y + 3.66, "full cell centroids", size=17, weight="bold")
    label(ax, x + 1.88, y + 3.22, "flat centroid scoring", size=14, color=PALETTE["muted"])

    prism(ax, x + 0.20, y + 1.35, 3.35, 1.05, dx=0.30, dy=0.22,
          face=PALETTE["red_light"], edge=PALETTE["red"], lw=1.8)
    label(ax, x + 1.88, y + 2.06, "cell sizes + offsets", size=17, weight="bold")
    label(ax, x + 1.88, y + 1.67, "cell metadata", size=14, color=PALETTE["muted"])

    prism(ax, x + 0.20, y, 3.35, 0.88, dx=0.30, dy=0.22,
          face=PALETTE["neutral_light"], edge=PALETTE["muted"], lw=1.8)
    label(ax, x + 1.88, y + 0.58, "cluster-ordered K/V lists", size=17, weight="bold")
    label(ax, x + 1.88, y + 0.25, "gather selected cells", size=14, color=PALETTE["muted"])

    rounded_box(ax, (13.62, 5.05), 2.78, 0.58, face=PALETTE["white"], edge=PALETTE["blue_main"], lw=1.8, radius=0.12)
    label(ax, 15.01, 5.34, "decode query Q", size=16, weight="bold", color=PALETTE["blue_main"])
    arrow(ax, (15.0, 5.03), (14.88, 4.23), color=PALETTE["blue_main"], lw=2.4)
    arrow(ax, (14.88, 3.72), (14.86, 3.05), color=PALETTE["green"], lw=2.4)
    arrow(ax, (14.86, 2.34), (14.85, 1.96), color=PALETTE["red"], lw=2.4)


def build_figure(out_dir: Path, basename: str) -> list[Path]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(18.5, 6.8))
    ax.set_xlim(0, 18.5)
    ax.set_ylim(0, 6.8)
    ax.set_axis_off()

    stage_header(ax, 0.55, 6.05, "Input", 2.9)
    stage_header(ax, 4.05, 6.05, "8-way labeling", 3.25)
    stage_header(ax, 8.05, 6.05, "Cell construction", 3.55)
    stage_header(ax, 13.05, 6.05, "Flat runtime index", 3.85)

    draw_kv_tensor(ax)
    draw_eight_way(ax)
    draw_cells(ax)
    draw_flat_index(ax)

    arrow(ax, (3.45, 3.75), (4.12, 4.15), color=PALETTE["blue_main"])
    arrow(ax, (3.45, 1.70), (4.12, 1.72), color=PALETTE["muted"], lw=2.2)
    arrow(ax, (7.12, 2.10), (8.02, 3.95), color=PALETTE["teal"])
    arrow(ax, (9.78, 3.52), (9.78, 2.78), color=PALETTE["teal"])
    arrow(ax, (11.48, 2.00), (13.10, 2.45), color=PALETTE["red"])
    arrow(ax, (11.44, 3.98), (13.12, 4.15), color=PALETTE["green"])

    label(ax, 12.28, 4.48, "centroid table", size=13.5, color=PALETTE["green"], weight="bold")
    label(ax, 12.23, 2.42, "layout table", size=13.5, color=PALETTE["red"], weight="bold")
    label(
        ax,
        9.2,
        0.18,
        "AdaptiveIMI flat path: build compact cells from 8-way labels, score full cell centroids, then gather K/V from selected cells.",
        size=16,
        color=PALETTE["ink"],
        weight="bold",
    )

    fig.tight_layout(pad=0.35)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [out_dir / f"{basename}.pdf", out_dir / f"{basename}.png"]
    fig.savefig(outputs[0])
    fig.savefig(outputs[1], dpi=350)
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the AdaptiveIMI 8-way flat-index construction schematic.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/root/AdaptiveIMI/outputs/oracle_topk_sufficiency/plots"),
    )
    parser.add_argument("--basename", type=str, default="adaptiveimi_8way_index_diagram")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in build_figure(args.out_dir, args.basename):
        print(path)


if __name__ == "__main__":
    main()
