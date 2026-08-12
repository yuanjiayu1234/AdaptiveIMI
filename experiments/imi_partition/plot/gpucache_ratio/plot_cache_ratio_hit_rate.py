import argparse
import glob
import os
import re
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "cache_ratio_hit_rate_results.csv")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "cache_ratio_hit_rate_curve.pdf")
DEFAULT_COMBINED_OUTPUT = os.path.join(SCRIPT_DIR, "cache_ratio_hit_rate_combined.pdf")
DEFAULT_PREFETCH_LOG_DIR = os.path.join(SCRIPT_DIR, "hit_rate_logs_prefetch_focus_long_outputs")
DEFAULT_PREFETCH_DATASET = "math_find_sample00_id1"
DEFAULT_PREFETCH_CACHE_RATIO = "0.1"
DEFAULT_PREFETCH_OUTPUT = os.path.join(
    DEFAULT_PREFETCH_LOG_DIR,
    "imi_prefetch_hit_rate_math_find_sample00_id1_cache_ratio_0p1.pdf",
)
DEFAULT_CACHE_RATIO_TITLE = "Hit Rate by Cache Ratio"
DEFAULT_PREFETCH_TITLE = "Hit Rate by Hot Ratio"
PANEL_FIGSIZE = (11.4, 2.8)
COMBINED_FIGSIZE = (12.2, 3.1)
BASE_FONT_SIZE = 15
TASK_ORDER = [
    "fwe",
    "qa1",
    "vt",
    "gov_report_short",
    "gov_report_long",
    "longbook_sum_eng_short",
    "longbook_sum_eng_long",
    "NIAH_8000",
    "NIAH_16000",
    "NIAH_30000",
    "NIAH_32000",
    "NIAH_60000",
    "NIAH_64000",
    "NIAH_96000",
    "NIAH_120000",
    "NIAH_128000",
    "NIAH_240000",
    "NIAH_256000",
    "NIAH_480000",
    "NIAH_1024000",
]
LINESTYLES = {
    "AdaptiveIMI": "-",
    "ExternalBaseline": "--",
}
METHOD_LABELS = {
    "AdaptiveIMI": "AdaptiveIMI",
    "ExternalBaseline": "External Baseline",
}
METHOD_LEGEND_LABELS = {
    "AdaptiveIMI": "AdpIMI",
    "ExternalBaseline": "Baseline",
}
HITRATE_RE = re.compile(r"\[IMI hitrate\] step=(\d+) total_hit_rate=([0-9.]+)")
PREFETCH_LOG_RE = re.compile(
    r"^imi_lruplus_hot(?P<hot>[0-9.]+)_decay(?P<decay>\d+)_(?P<dataset>[A-Za-z0-9_]+)\.log$"
)
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "red": "#B64342",
    "red_light": "#E9A6A1",
    "green": "#8BCF8B",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "yellow": "#F2C14E",
    "ink": "#202124",
    "muted": "#626A73",
    "grid": "#E8EAED",
}

MODERN_TASK_PALETTE = [
    "#005AB5",  # blue
    "#DC3220",  # vermilion
    "#009E73",  # green
    "#CC79A7",  # magenta
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#7F3C8D",  # purple
    "#11A579",  # teal green
]

HOT_RATIO_PALETTE = [
    "#005AB5",
    "#DC3220",
    "#009E73",
    "#CC79A7",
    "#E69F00",
]
HOT_RATIO_LINESTYLES = {
    0.5: "--",
    0.9: "--",
}


def _hot_ratio_linestyle(value: float) -> str:
    for ratio, linestyle in HOT_RATIO_LINESTYLES.items():
        if abs(float(value) - ratio) < 1e-9:
            return linestyle
    return "-"


def apply_publication_style():
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "legend.fontsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE,
        "ytick.labelsize": BASE_FONT_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.5,
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Plot cache-ratio hit-rate curves and the selected hot-ratio comparison panel.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--combined_output", default=DEFAULT_COMBINED_OUTPUT)
    parser.add_argument(
        "--metric",
        default="mean_hit_rate_skip_first2",
        choices=["mean_hit_rate_skip_first2", "mean_hit_rate", "final_hit_rate"],
        help="Hit-rate metric to plot. Default removes the first two decode steps as cold start.",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Optional comma-separated task names to include. Default includes all available tasks.",
    )
    parser.add_argument("--title", default="")
    parser.add_argument("--xlabel", default="Cache ratio")
    parser.add_argument("--ylabel", default="Hit rate")
    parser.add_argument("--legend_cols", type=int, default=4)
    parser.add_argument("--prefetch_log_dir", default=DEFAULT_PREFETCH_LOG_DIR)
    parser.add_argument("--prefetch_dataset", default=DEFAULT_PREFETCH_DATASET)
    parser.add_argument("--prefetch_cache_ratio", default=DEFAULT_PREFETCH_CACHE_RATIO)
    parser.add_argument("--prefetch_output", default=DEFAULT_PREFETCH_OUTPUT)
    parser.add_argument("--prefetch_title", default="")
    parser.add_argument("--prefetch_legend_cols", type=int, default=3)
    return parser.parse_args()


def _task_sort_key(task_name: str):
    if task_name in TASK_ORDER:
        return (0, TASK_ORDER.index(task_name), task_name)
    if task_name.startswith("NIAH_"):
        try:
            return (1, int(task_name.split("_", 1)[1]), task_name)
        except ValueError:
            return (1, 10**18, task_name)
    return (2, 0, task_name)


def _resolve_tasks(df: pd.DataFrame, requested: str):
    available = sorted(df["task_name"].dropna().unique().tolist(), key=_task_sort_key)
    if not requested:
        return [task for task in available if task not in {"gov_report_short", "gov_report_long"}]
    requested_tasks = [item.strip() for item in requested.split(",") if item.strip()]
    requested_set = set(requested_tasks)
    return [task for task in available if task in requested_set]


def _coerce_numeric(df: pd.DataFrame, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _parse_ratio_value(value) -> float:
    text = str(value).strip()
    text = text.replace("neg", "-").replace("p", ".")
    return float(text)


def _encode_ratio_slug(value) -> str:
    normalized = format(_parse_ratio_value(value), "g")
    return normalized.replace("-", "neg").replace(".", "p")


def _normalize_output_path(path: str) -> str:
    stem, ext = os.path.splitext(path)
    if ext.lower() in {".pdf", ".png"}:
        return path
    return f"{path}.pdf"


def _save_figure(fig, output_path: str, extra_artists=None):
    normalized_path = _normalize_output_path(output_path)
    output_dir = os.path.dirname(normalized_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    extra_artists = [artist for artist in (extra_artists or []) if artist is not None]
    save_kwargs = {"dpi": 300}
    if extra_artists:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["bbox_extra_artists"] = tuple(extra_artists)
    fig.savefig(normalized_path, **save_kwargs)
    stem, ext = os.path.splitext(normalized_path)
    alt_ext = ".png" if ext.lower() == ".pdf" else ".pdf"
    alt_path = f"{stem}{alt_ext}"
    fig.savefig(alt_path, **save_kwargs)
    print(f"Saved to {normalized_path}")
    print(f"Saved to {alt_path}")
    return normalized_path, alt_path


def _format_task_label(task_name: str) -> str:
    task_aliases = {
        "gov_report_short": "Gov_Short",
        "gov_report_long": "Gov_Long",
        "longbook_sum_eng_short": "BookSumShort",
        "longbook_sum_eng_long": "BookSumLong",
    }
    if task_name in task_aliases:
        return task_aliases[task_name]
    if task_name.startswith("NIAH_"):
        suffix = task_name.split("_", 1)[1]
        try:
            value = int(suffix)
        except ValueError:
            return task_name
        if value % 1000 == 0:
            return f"NIAH_{value // 1000}k"
        return f"NIAH_{value}"
    return task_name


def _build_task_colors(tasks):
    return OrderedDict(
        (task, MODERN_TASK_PALETTE[idx % len(MODERN_TASK_PALETTE)])
        for idx, task in enumerate(tasks)
    )


def _load_cache_ratio_data(args):
    df = pd.read_csv(args.input)
    numeric_columns = [
        "cache_ratio",
        "cache_tokens",
        "cache_pages",
        "return_code",
        "mean_hit_rate_skip_first2",
        "mean_hit_rate",
        "final_hit_rate",
    ]
    df = _coerce_numeric(df, numeric_columns)

    metric = args.metric
    df["plot_metric"] = df[metric]
    if metric == "mean_hit_rate_skip_first2":
        df["plot_metric"] = df["plot_metric"].fillna(df["mean_hit_rate"])

    df = df[(df["return_code"] == 0) & df["cache_ratio"].notna() & df["plot_metric"].notna()].copy()
    if df.empty:
        raise SystemExit("No valid rows to plot after filtering return_code/cache_ratio/metric")

    tasks = _resolve_tasks(df, args.tasks)
    df = df[df["task_name"].isin(tasks)].copy()
    if df.empty:
        raise SystemExit("No rows left after applying task filter")

    tasks = sorted(df["task_name"].unique().tolist(), key=_task_sort_key)
    return df, tasks, _build_task_colors(tasks)


def _parse_hitrate_log(path: str):
    steps, rates = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = HITRATE_RE.search(line)
            if match:
                steps.append(int(match.group(1)))
                rates.append(float(match.group(2)))
    if not steps:
        return [], []
    paired = sorted(zip(steps, rates))
    steps, rates = zip(*paired)
    return list(steps), list(rates)


def _collect_prefetch_series(log_dir: str, dataset: str, cache_ratio: str):
    ratio_slug = _encode_ratio_slug(cache_ratio)
    pattern = os.path.join(
        log_dir,
        f"cache_ratio_{ratio_slug}",
        "lruplus",
        "hot_*",
        f"imi_lruplus_hot*_decay0_{dataset}.log",
    )
    series = []
    for path in sorted(glob.glob(pattern)):
        match = PREFETCH_LOG_RE.match(os.path.basename(path))
        if not match:
            continue
        try:
            hot_ratio = float(match.group("hot"))
        except ValueError:
            continue
        if abs(hot_ratio - 0.7) < 1e-9:
            continue
        steps, rates = _parse_hitrate_log(path)
        if len(steps) <= 2:
            continue
        series.append(
            {
                "hot_ratio": hot_ratio,
                "label": f"hot_ratio={hot_ratio:g}",
                "steps": steps[2:],
                "rates": rates[2:],
            }
        )
    series.sort(key=lambda item: item["hot_ratio"])
    return series


def _style_panel(ax, title: str, panel_label: str | None = None, title_below: bool = False):
    if not title_below and title:
        ax.set_title(title, fontsize=BASE_FONT_SIZE, pad=8, color=PALETTE["ink"])
    ax.tick_params(axis="both")
    ax.grid(True, axis="both", alpha=0.65, linestyle="-", linewidth=0.85, color=PALETTE["grid"])



def _draw_method_key(ax):
    x0, x1 = 0.58, 0.70
    y0, y1 = 0.19, 0.12
    ax.plot([x0, x1], [y0, y0], transform=ax.transAxes, color=PALETTE["ink"], lw=2.4, linestyle=LINESTYLES["AdaptiveIMI"], clip_on=False)
    ax.text(x1 + 0.02, y0, METHOD_LABELS["AdaptiveIMI"], transform=ax.transAxes, fontsize=BASE_FONT_SIZE, color=PALETTE["ink"], va="center", ha="left")
    ax.plot([x0, x1], [y1, y1], transform=ax.transAxes, color=PALETTE["ink"], lw=2.4, linestyle=LINESTYLES["ExternalBaseline"], clip_on=False)
    ax.text(x1 + 0.02, y1, METHOD_LABELS["ExternalBaseline"], transform=ax.transAxes, fontsize=BASE_FONT_SIZE, color=PALETTE["ink"], va="center", ha="left")

def _plot_cache_ratio_panel(ax, legend_ax, df, tasks, task_colors, args, panel_label: str | None = None, title_below: bool = False):
    for task_name in tasks:
        task_df = df[df["task_name"] == task_name]
        color = task_colors[task_name]
        for attn_type in ["AdaptiveIMI", "ExternalBaseline"]:
            sub_df = task_df[task_df["attn_type"] == attn_type].copy()
            if sub_df.empty:
                continue
            sub_df = sub_df.sort_values(by=["cache_ratio", "cache_tokens"])
            ax.plot(
                sub_df["cache_ratio"],
                sub_df["plot_metric"],
                linestyle=LINESTYLES.get(attn_type, "-"),
                color=color,
                linewidth=2.2,
                alpha=0.95,
            )

    _style_panel(
        ax,
        args.title or DEFAULT_CACHE_RATIO_TITLE,
        panel_label=panel_label,
        title_below=title_below,
    )
    ax.set_xlabel(args.xlabel, labelpad=3)
    ax.set_ylabel(args.ylabel, labelpad=8)
    ax.set_ylim(0.3, 1.02)

    dataset_handles = [
        Patch(facecolor=task_colors[task], edgecolor="none", label=_format_task_label(task))
        for task in tasks
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color=PALETTE["ink"],
            lw=2.4,
            linestyle=LINESTYLES[method],
            label=METHOD_LEGEND_LABELS[method],
        )
        for method in ("AdaptiveIMI", "ExternalBaseline")
    ]
    legend = ax.legend(
        handles=dataset_handles + method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        frameon=False,
        handlelength=1.9,
        columnspacing=0.65 if title_below else 1.0,
        handletextpad=0.45,
    )
    legend.set_in_layout(False)
    return legend


def _plot_prefetch_panel(ax, legend_ax, series, args, panel_label: str | None = None, title_below: bool = False):
    if not series:
        raise ValueError("No usable prefetch hot-ratio logs were found for the selected dataset/cache ratio")

    min_rate = 1.0
    max_rate = 0.0
    handles = []

    for idx, item in enumerate(series):
        color = HOT_RATIO_PALETTE[idx % len(HOT_RATIO_PALETTE)]
        linestyle = _hot_ratio_linestyle(item["hot_ratio"])
        ax.plot(
            item["steps"],
            item["rates"],
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            alpha=0.95,
        )
        handles.append(Line2D([0], [0], color=color, linestyle=linestyle, lw=2.2, label=item["label"]))
        min_rate = min(min_rate, min(item["rates"]))
        max_rate = max(max_rate, max(item["rates"]))

    _style_panel(
        ax,
        args.prefetch_title or DEFAULT_PREFETCH_TITLE,
        panel_label=panel_label,
        title_below=title_below,
    )
    ax.set_xlabel("Decode step", labelpad=3)
    ax.set_ylabel("Hit rate", labelpad=8)

    margin = 0.01
    y_min = max(min_rate - margin, 0.0)
    y_max = min(max_rate + margin, 1.0)
    if y_max - y_min < 0.02:
        mid = (y_max + y_min) / 2
        y_min = max(mid - 0.01, 0.0)
        y_max = min(mid + 0.01, 1.0)
    ax.set_ylim(y_min, y_max)

    legend = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=False,
        handlelength=1.9,
        columnspacing=0.65 if title_below else 1.0,
        handletextpad=0.45,
    )
    legend.set_in_layout(False)
    return legend


def _make_panel_figure(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, None


def main():
    apply_publication_style()
    args = parse_args()
    df, tasks, task_colors = _load_cache_ratio_data(args)
    prefetch_series = _collect_prefetch_series(
        args.prefetch_log_dir,
        args.prefetch_dataset,
        args.prefetch_cache_ratio,
    )

    cache_fig, cache_ax, cache_legend_ax = _make_panel_figure(PANEL_FIGSIZE)
    cache_legend = _plot_cache_ratio_panel(cache_ax, cache_legend_ax, df, tasks, task_colors, args)
    _save_figure(cache_fig, args.output, extra_artists=[cache_legend])
    plt.close(cache_fig)

    if not prefetch_series:
        print(
            "Warning: no matching hot-ratio logs found, so only the cache-ratio figure was generated."
        )
        return

    prefetch_fig, prefetch_ax, prefetch_legend_ax = _make_panel_figure(PANEL_FIGSIZE)
    prefetch_legend = _plot_prefetch_panel(prefetch_ax, prefetch_legend_ax, prefetch_series, args)
    _save_figure(prefetch_fig, args.prefetch_output, extra_artists=[prefetch_legend])
    plt.close(prefetch_fig)

    combined_fig = plt.figure(figsize=COMBINED_FIGSIZE)
    outer = combined_fig.add_gridspec(1, 2, wspace=0.24)

    combined_cache_ax = combined_fig.add_subplot(outer[0])
    combined_prefetch_ax = combined_fig.add_subplot(outer[1])

    combined_cache_legend = _plot_cache_ratio_panel(
        combined_cache_ax,
        None,
        df,
        tasks,
        task_colors,
        args,
        panel_label="(a)",
        title_below=True,
    )
    combined_prefetch_legend = _plot_prefetch_panel(
        combined_prefetch_ax,
        None,
        prefetch_series,
        args,
        panel_label="(b)",
        title_below=True,
    )

    caption_a = combined_fig.text(
        0.28,
        -0.15,
        f"(a) {args.title or DEFAULT_CACHE_RATIO_TITLE}",
        ha="center",
        va="bottom",
        fontsize=BASE_FONT_SIZE,
    )
    caption_b = combined_fig.text(
        0.74,
        -0.15,
        f"(b) {args.prefetch_title or DEFAULT_PREFETCH_TITLE}",
        ha="center",
        va="bottom",
        fontsize=BASE_FONT_SIZE,
    )

    _save_figure(combined_fig, args.combined_output, extra_artists=[combined_cache_legend, combined_prefetch_legend, caption_a, caption_b])
    plt.close(combined_fig)


if __name__ == "__main__":
    main()
