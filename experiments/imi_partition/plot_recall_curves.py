from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


METHOD_ORDER = ["full_kmeans", "learned", "pair_sequential", "sequential", "pq"]
METHOD_LABELS = {
    "full_kmeans": "KMeans",
    "learned": "Learned",
    "pair_sequential": "Sequential",
    "sequential": "Pair-Sequential",
    "pq": "PQ",
}
METHOD_LEGEND_LABELS = {
    "full_kmeans": "KMeans",
    "learned": "Learned",
    "pair_sequential": "seq.",
    "sequential": "pair-seq.",
    "pq": "PQ",
}
METHOD_COLORS = {
    "full_kmeans": "#4D4D4D",
    "learned": "#850C0A",
    "pair_sequential": "#8EAD03",
    "sequential": "#0F4D92",
    "pq": "#EA84DD",
}
METHOD_MARKERS = {
    "full_kmeans": "circle",
    "learned": "circle",
    "pair_sequential": "circle",
    "sequential": "circle",
    "pq": "circle",
}
METHOD_DASHES = {
    "full_kmeans": "2 5",
    "learned": "",
    "pair_sequential": "8 4 2 4",
    "sequential": "8 5",
    "pq": "8 3 2 3",
}
MODEL_DISPLAY_NAMES = {
    "qwen3": "Qwen3-8B",
    "llama3": "Llama3.1-8B",
}

DEFAULT_LLAMA3_DIR = Path(
    "/root/AdaptiveIMI/outputs/imi_partition/"
    "recall_llama3_longbench_r100_10k_learned33333333_seq33222222_flat"
)
DEFAULT_QWEN3_DIR = Path(
    "/root/AdaptiveIMI/outputs/imi_partition/"
    "recall_qwen3_longbench_r100_10k_learned33333333_seq33222222_flat"
)
DEFAULT_PANEL_SUMMARIES = (
    f"llama3:{DEFAULT_LLAMA3_DIR / 'recall_summary.json'},"
    f"qwen3:{DEFAULT_QWEN3_DIR / 'recall_summary.json'}"
)
DEFAULT_PANEL_OUT = Path(
    "/root/AdaptiveIMI/outputs/imi_partition/recall_llama3_qwen3_10k_learned.pdf"
)


def _method_sort_key(method: str) -> Tuple[int, str]:
    try:
        return METHOD_ORDER.index(method), method
    except ValueError:
        return len(METHOD_ORDER), method


def _points_for(
    rows: Sequence[dict],
    *,
    mode: str,
    recall_k: int,
    method: str,
    metric: str,
) -> List[Tuple[float, float]]:
    points = []
    for row in rows:
        if str(row.get("decode_mode")) != mode:
            continue
        if int(row.get("recall_k")) != int(recall_k):
            continue
        if str(row.get("method")) != method:
            continue
        points.append((float(row["retrieval_budget"]), float(row[metric])))
    return sorted(points)


def _polyline(points: Sequence[Tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _svg_marker(marker: str, x: float, y: float, size: float, color: str) -> str:
    stroke = f'stroke="{color}" stroke-width="2" fill="white"'
    if marker == "square":
        return f'<rect x="{x - size:.2f}" y="{y - size:.2f}" width="{2 * size:.2f}" height="{2 * size:.2f}" {stroke}/>'
    if marker == "triangle":
        points = [(x, y - size), (x - size, y + size), (x + size, y + size)]
        return f'<polygon points="{_polyline(points)}" {stroke}/>'
    if marker == "down_triangle":
        points = [(x, y + size), (x - size, y - size), (x + size, y - size)]
        return f'<polygon points="{_polyline(points)}" {stroke}/>'
    if marker == "diamond":
        points = [(x, y - size), (x - size, y), (x, y + size), (x + size, y)]
        return f'<polygon points="{_polyline(points)}" {stroke}/>'
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" {stroke}/>'


def _nice_metric_name(metric: str) -> str:
    return metric.replace("_mean", "").replace("_", " ")


def _metric_axis_label(metric: str, recall_k: int) -> str:
    if metric in {"candidate_recall_mean", "candidate_recall"}:
        return f"Recall@{int(recall_k)}"
    if metric in {"rerank_recall_mean", "rerank_recall"}:
        return f"Recall@{int(recall_k)}"
    return _nice_metric_name(metric)


def _draw_svg(
    rows: Sequence[dict],
    *,
    mode: str,
    metric: str,
    out_path: Path,
) -> None:
    methods = sorted({str(row["method"]) for row in rows if str(row.get("decode_mode")) == mode}, key=_method_sort_key)
    recall_ks = sorted({int(row["recall_k"]) for row in rows if str(row.get("decode_mode")) == mode})
    budgets = sorted({float(row["retrieval_budget"]) for row in rows if str(row.get("decode_mode")) == mode})
    if not methods or not recall_ks or not budgets:
        return

    panel_w = 300
    panel_h = 232
    margin_l = 62
    margin_t = 100
    plot_w = 220
    plot_h = 150
    gap = 20
    width = max(margin_l + len(recall_ks) * panel_w + 24, 560)
    height = margin_t + panel_h + 72
    max_y = max(float(row[metric]) for row in rows if str(row.get("decode_mode")) == mode)
    max_y = max(max_y, 1e-9)
    y_top = min(1.0, max_y * 1.15)

    def x_pos(panel_x: float, budget: float) -> float:
        if len(budgets) == 1:
            return panel_x + plot_w / 2
        idx = budgets.index(float(budget))
        return panel_x + idx * (plot_w / (len(budgets) - 1))

    def y_pos(panel_y: float, value: float) -> float:
        return panel_y + plot_h - (float(value) / y_top) * plot_h

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="30" font-family="Helvetica, Arial, sans-serif" font-size="19" font-weight="700">{html.escape(mode)} {_nice_metric_name(metric)}</text>',
    ]

    legend_x = 210
    legend_y = 28
    legend_right = width - 20
    for idx, method in enumerate(methods):
        color = METHOD_COLORS.get(method, "#111827")
        marker = METHOD_MARKERS.get(method, "circle")
        dash = METHOD_DASHES.get(method, "")
        label = METHOD_LABELS.get(method, method)
        entry_w = 44 + len(label) * 7.2
        if legend_x + entry_w > legend_right:
            legend_x = 210
            legend_y += 24
        x = legend_x
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"{dash_attr}/>')
        parts.append(_svg_marker(marker, x + 12, legend_y, 4.2, color))
        parts.append(f'<text x="{x + 31}" y="{legend_y + 5}" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#111827">{html.escape(label)}</text>')
        legend_x += entry_w + 18

    for panel_idx, recall_k in enumerate(recall_ks):
        origin_x = margin_l + panel_idx * panel_w
        origin_y = margin_t
        parts.append(f'<text x="{origin_x}" y="{origin_y - 14}" font-family="Helvetica, Arial, sans-serif" font-size="15" font-weight="700">recall@{recall_k}</text>')
        parts.append(f'<line x1="{origin_x}" y1="{origin_y + plot_h}" x2="{origin_x + plot_w}" y2="{origin_y + plot_h}" stroke="#111827" stroke-width="1.6"/>')
        parts.append(f'<line x1="{origin_x}" y1="{origin_y}" x2="{origin_x}" y2="{origin_y + plot_h}" stroke="#111827" stroke-width="1.6"/>')

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = origin_y + plot_h - frac * plot_h
            value = frac * y_top
            parts.append(f'<line x1="{origin_x}" y1="{y:.2f}" x2="{origin_x + plot_w}" y2="{y:.2f}" stroke="#D9D9D9" stroke-width="0.7"/>')
            parts.append(f'<text x="{origin_x - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#111827">{value:.2f}</text>')

        for budget in budgets:
            x = x_pos(origin_x, budget)
            parts.append(f'<text x="{x:.2f}" y="{origin_y + plot_h + 20}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#111827">{budget:g}</text>')

        for method in methods:
            raw_points = _points_for(rows, mode=mode, recall_k=recall_k, method=method, metric=metric)
            points = [(x_pos(origin_x, budget), y_pos(origin_y, value)) for budget, value in raw_points]
            if not points:
                continue
            color = METHOD_COLORS.get(method, "#111827")
            marker = METHOD_MARKERS.get(method, "circle")
            dash = METHOD_DASHES.get(method, "")
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(f'<polyline points="{_polyline(points)}" fill="none" stroke="{color}" stroke-width="2.8"{dash_attr} stroke-linecap="round" stroke-linejoin="round"/>')
            for x, y in points:
                parts.append(_svg_marker(marker, x, y, 4.0, color))

        if panel_idx < len(recall_ks) - 1:
            parts.append(f'<line x1="{origin_x + plot_w + gap}" y1="{origin_y - 8}" x2="{origin_x + plot_w + gap}" y2="{origin_y + plot_h + 28}" stroke="#e2e8f0"/>')

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def _pdf_escape(value: object) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_color(hex_color: str) -> Tuple[float, float, float]:
    text = str(hex_color).lstrip("#")
    return tuple(int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _write_pdf(page_w: float, page_h: float, commands: Sequence[str], out_path: Path) -> None:
    stream = "\n".join(commands).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.2f} {page_h:.2f}] "
            f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>"
        ).encode("ascii"),
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    out_path.write_bytes(bytes(out))


def _draw_circle(commands: List[str], x: float, y: float, r: float, color: Tuple[float, float, float]) -> None:
    k = 0.5522847498 * r
    commands.append(
        f"{color[0]:.4f} {color[1]:.4f} {color[2]:.4f} rg "
        f"{x + r:.2f} {y:.2f} m "
        f"{x + r:.2f} {y + k:.2f} {x + k:.2f} {y + r:.2f} {x:.2f} {y + r:.2f} c "
        f"{x - k:.2f} {y + r:.2f} {x - r:.2f} {y + k:.2f} {x - r:.2f} {y:.2f} c "
        f"{x - r:.2f} {y - k:.2f} {x - k:.2f} {y - r:.2f} {x:.2f} {y - r:.2f} c "
        f"{x + k:.2f} {y - r:.2f} {x + r:.2f} {y - k:.2f} {x + r:.2f} {y:.2f} c f"
    )


def _draw_square(commands: List[str], x: float, y: float, r: float, color: Tuple[float, float, float]) -> None:
    commands.append(f"{color[0]:.4f} {color[1]:.4f} {color[2]:.4f} rg {x - r:.2f} {y - r:.2f} {2 * r:.2f} {2 * r:.2f} re f")


def _dash_command(method: str, *, dashed: bool = False) -> str:
    if dashed:
        return "[7 5] 0 d"
    dash = METHOD_DASHES.get(method, "")
    if not dash:
        return "[] 0 d"
    return f"[{dash}] 0 d"


def _draw_pdf_marker(
    commands: List[str],
    x: float,
    y: float,
    r: float,
    color: Tuple[float, float, float],
    marker: str,
) -> None:
    stroke = f"1 1 1 rg {color[0]:.4f} {color[1]:.4f} {color[2]:.4f} RG 1.80 w"
    if marker == "square":
        commands.append(f"{stroke} {x - r:.2f} {y - r:.2f} {2 * r:.2f} {2 * r:.2f} re B")
        return
    if marker == "triangle":
        commands.append(
            f"{stroke} {x:.2f} {y + r:.2f} m {x - r:.2f} {y - r:.2f} l {x + r:.2f} {y - r:.2f} l h B"
        )
        return
    if marker == "down_triangle":
        commands.append(
            f"{stroke} {x:.2f} {y - r:.2f} m {x - r:.2f} {y + r:.2f} l {x + r:.2f} {y + r:.2f} l h B"
        )
        return
    if marker == "diamond":
        commands.append(
            f"{stroke} {x:.2f} {y + r:.2f} m {x - r:.2f} {y:.2f} l {x:.2f} {y - r:.2f} l {x + r:.2f} {y:.2f} l h B"
        )
        return

    k = 0.5522847498 * r
    commands.append(
        f"{stroke} "
        f"{x + r:.2f} {y:.2f} m "
        f"{x + r:.2f} {y + k:.2f} {x + k:.2f} {y + r:.2f} {x:.2f} {y + r:.2f} c "
        f"{x - k:.2f} {y + r:.2f} {x - r:.2f} {y + k:.2f} {x - r:.2f} {y:.2f} c "
        f"{x - r:.2f} {y - k:.2f} {x - k:.2f} {y - r:.2f} {x:.2f} {y - r:.2f} c "
        f"{x + k:.2f} {y - r:.2f} {x + r:.2f} {y - k:.2f} {x + r:.2f} {y:.2f} c B"
    )


def _draw_pdf(
    rows: Sequence[dict],
    *,
    mode: str,
    metric: str,
    out_path: Path,
) -> None:
    mode_rows = [row for row in rows if str(row.get("decode_mode")) == mode]
    methods = sorted({str(row["method"]) for row in mode_rows}, key=_method_sort_key)
    recall_ks = sorted({int(row["recall_k"]) for row in mode_rows})
    budgets = sorted({float(row["retrieval_budget"]) for row in mode_rows})
    if not methods or not recall_ks or not budgets:
        return
    if len(recall_ks) > 1:
        raise SystemExit("PDF output expects one recall_k; pass a filtered summary or use SVG for multi-panel output")

    width = 660.0
    height = 470.0
    pad_l, pad_r, pad_t, pad_b = 88.0, 34.0, 96.0, 76.0
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    max_y = max(float(row[metric]) for row in mode_rows)
    y_top = min(1.0, max(max_y * 1.12, 0.1))

    x0, y0 = pad_l, pad_t
    x1, y1 = width - pad_r, height - pad_b

    def x_pos(budget: float) -> float:
        if len(budgets) == 1:
            return x0 + plot_w / 2
        idx = budgets.index(float(budget))
        return x0 + idx * (plot_w / (len(budgets) - 1))

    def y_pos(value: float) -> float:
        return y0 + (float(value) / y_top) * plot_h

    commands: List[str] = [
        "1 1 1 rg 0 0 %.2f %.2f re f" % (width, height),
        "1 J 1 j",
    ]

    def stroke_line(
        xa: float,
        ya: float,
        xb: float,
        yb: float,
        color: Tuple[float, float, float],
        width_pt: float,
        *,
        dash: str = "",
    ) -> None:
        commands.append(
            f"{dash or '[] 0 d'} {color[0]:.4f} {color[1]:.4f} {color[2]:.4f} RG "
            f"{width_pt:.2f} w {xa:.2f} {ya:.2f} m {xb:.2f} {yb:.2f} l S [] 0 d"
        )

    def fill_text(x: float, y: float, text: object, size: int, *, bold: bool = False, align: str = "left") -> None:
        text = str(text)
        approx_w = len(text) * size * 0.52
        if align == "center":
            x -= approx_w / 2.0
        elif align == "right":
            x -= approx_w
        font = "F2" if bold else "F1"
        commands.append(f"0.0667 0.0941 0.1529 rg BT /{font} {size} Tf {x:.2f} {y:.2f} Td ({_pdf_escape(text)}) Tj ET")

    stroke_line(x0, y0, x1, y0, (0.07, 0.09, 0.13), 1.6)
    stroke_line(x0, y0, x0, y1, (0.07, 0.09, 0.13), 1.6)

    tick_size = 18
    label_size = 20
    legend_size = 18
    y_label = _metric_axis_label(metric, recall_ks[0])
    fill_text((x0 + x1) / 2.0, 24.0, "Retrieval budget", label_size, align="center")
    commands.append("q 0 1 -1 0 31.00 166.00 cm")
    commands.append(f"0.0667 0.0941 0.1529 rg BT /F1 {label_size} Tf 0 0 Td ({_pdf_escape(y_label)}) Tj ET")
    commands.append("Q")

    for i in range(5):
        frac = i / 4
        y = y0 + frac * plot_h
        value = frac * y_top
        stroke_line(x0, y, x1, y, (0.85, 0.85, 0.85), 0.7)
        fill_text(x0 - 12.0, y - 6.0, f"{value:.2f}", tick_size, align="right")

    for budget in budgets:
        x = x_pos(budget)
        stroke_line(x, y0, x, y0 - 5.0, (0.07, 0.09, 0.13), 1.2)
        fill_text(x, y0 - 28.0, f"{budget:g}", tick_size, align="center")

    legend_x = x0
    legend_y = height - 43.0
    legend_right = width - pad_r
    for method in methods:
        color = _pdf_color(METHOD_COLORS.get(method, "#111827"))
        label = METHOD_LABELS.get(method, method)
        entry_w = 58.0 + len(label) * legend_size * 0.50
        if legend_x + entry_w > legend_right:
            legend_x = x0
            legend_y -= 30.0
        dash = _dash_command(method)
        stroke_line(legend_x, legend_y + 6.0, legend_x + 34.0, legend_y + 6.0, color, 3.0, dash=dash)
        _draw_pdf_marker(commands, legend_x + 17.0, legend_y + 6.0, 4.6, color, METHOD_MARKERS.get(method, "circle"))
        fill_text(legend_x + 44.0, legend_y, label, legend_size)
        legend_x += entry_w + 22.0

    for method in methods:
        raw_points = _points_for(rows, mode=mode, recall_k=recall_ks[0], method=method, metric=metric)
        points = [(x_pos(budget), y_pos(value)) for budget, value in raw_points]
        if not points:
            continue
        color = _pdf_color(METHOD_COLORS.get(method, "#111827"))
        if len(points) > 1:
            path = [f"{_dash_command(method)} {color[0]:.4f} {color[1]:.4f} {color[2]:.4f} RG 3.0 w"]
            path.append(f"{points[0][0]:.2f} {points[0][1]:.2f} m")
            path.extend(f"{x:.2f} {y:.2f} l" for x, y in points[1:])
            path.append("S [] 0 d")
            commands.append(" ".join(path))
        for x, y in points:
            _draw_pdf_marker(commands, x, y, 4.6, color, METHOD_MARKERS.get(method, "circle"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pdf(width, height, commands, out_path)


def _parse_compare_summaries(value: str) -> List[Tuple[str, Path]]:
    specs: List[Tuple[str, Path]] = []
    for item in str(value).replace(",", " ").split():
        if not item.strip():
            continue
        if ":" in item:
            label, path = item.split(":", 1)
        elif "=" in item:
            label, path = item.split("=", 1)
        else:
            path_obj = Path(item)
            label = path_obj.parent.name.rsplit("_", 1)[-1]
            path = item
        label = label.strip()
        specs.append((MODEL_DISPLAY_NAMES.get(label, label), Path(path)))
    if len(specs) < 2:
        raise SystemExit("--compare-summaries expects at least two label:path entries")
    return specs


def _draw_panel_pdf(
    series: Sequence[Tuple[str, Sequence[dict]]],
    *,
    mode: str,
    metric: str,
    out_path: Path,
) -> None:
    methods = [method for method in METHOD_ORDER if any(
        str(row.get("method")) == method and str(row.get("decode_mode")) == mode
        for _, rows in series
        for row in rows
    )]
    recall_ks = sorted({int(row["recall_k"]) for _, rows in series for row in rows if str(row.get("decode_mode")) == mode})
    if not methods:
        raise SystemExit(f"no methods found for mode={mode!r}")
    if len(recall_ks) != 1:
        raise SystemExit("panel PDF expects exactly one recall_k")

    width = 11.4 * 72.0
    height = 4.1 * 72.0
    pad_l = 76.0
    pad_r = 24.0
    panel_gap = 70.0
    plot_w = (width - pad_l - pad_r - panel_gap) / 2.0
    plot_h = 174.0
    y0 = 54.0
    y1 = y0 + plot_h
    x_starts = [pad_l, pad_l + plot_w + panel_gap]
    all_values = [
        float(row[metric])
        for _, rows in series
        for row in rows
        if str(row.get("decode_mode")) == mode and metric in row
    ]
    if not all_values:
        raise SystemExit(f"metric {metric!r} missing from panel rows")
    y_top = min(1.0, max(max(all_values) * 1.10, 0.1))

    commands: List[str] = [
        "1 1 1 rg 0 0 %.2f %.2f re f" % (width, height),
        "1 J 1 j",
    ]

    def stroke_line(
        xa: float,
        ya: float,
        xb: float,
        yb: float,
        color: Tuple[float, float, float],
        width_pt: float,
        *,
        dash: str = "",
    ) -> None:
        commands.append(
            f"{dash or '[] 0 d'} {color[0]:.4f} {color[1]:.4f} {color[2]:.4f} RG "
            f"{width_pt:.2f} w {xa:.2f} {ya:.2f} m {xb:.2f} {yb:.2f} l S [] 0 d"
        )

    def fill_text(
        x: float,
        y: float,
        text: object,
        size: int,
        *,
        bold: bool = False,
        align: str = "left",
    ) -> None:
        text = str(text)
        approx_w = len(text) * size * 0.52
        if align == "center":
            x -= approx_w / 2.0
        elif align == "right":
            x -= approx_w
        font = "F2" if bold else "F1"
        commands.append(f"0.0667 0.0941 0.1529 rg BT /{font} {size} Tf {x:.2f} {y:.2f} Td ({_pdf_escape(text)}) Tj ET")

    label_size = 15
    tick_size = 15
    title_size = 15
    legend_size = 15

    for panel_idx, (title, rows) in enumerate(series):
        panel_rows = [row for row in rows if str(row.get("decode_mode")) == mode]
        budgets = sorted({float(row["retrieval_budget"]) for row in panel_rows})
        if not budgets:
            raise SystemExit(f"no rows found for panel {title!r}")
        x0 = x_starts[panel_idx]
        x1 = x0 + plot_w

        def x_pos(budget: float) -> float:
            if len(budgets) == 1:
                return x0 + plot_w / 2.0
            return x0 + budgets.index(float(budget)) * (plot_w / (len(budgets) - 1))

        def y_pos(value: float) -> float:
            return y0 + (float(value) / y_top) * plot_h

        stroke_line(x0, y0, x1, y0, (0.07, 0.09, 0.13), 1.7)
        stroke_line(x0, y0, x0, y1, (0.07, 0.09, 0.13), 1.7)

        for i in range(5):
            frac = i / 4.0
            y = y0 + frac * plot_h
            stroke_line(x0, y, x1, y, (0.85, 0.85, 0.85), 0.7)
            fill_text(x0 - 10.0, y - 5.5, f"{frac * y_top:.2f}", tick_size, align="right")

        for budget in budgets:
            x = x_pos(budget)
            stroke_line(x, y0, x, y0 - 5.0, (0.07, 0.09, 0.13), 1.2)
            fill_text(x, y0 - 31.0, f"{budget:g}", tick_size, align="center")

        fill_text((x0 + x1) / 2.0, 5.0, "Retrieval budget", label_size, align="center")
        fill_text((x0 + x1) / 2.0, y1 + 18.0, title, title_size, align="center")
        if panel_idx == 0:
            commands.append("q 0 1 -1 0 18.00 134.00 cm")
            commands.append(f"0.0667 0.0941 0.1529 rg BT /F1 {label_size} Tf 0 0 Td ({_pdf_escape(_metric_axis_label(metric, recall_ks[0]))}) Tj ET")
            commands.append("Q")

        for method in methods:
            raw_points = _points_for(panel_rows, mode=mode, recall_k=recall_ks[0], method=method, metric=metric)
            if not raw_points:
                continue
            points = [(x_pos(budget), y_pos(value)) for budget, value in raw_points]
            color = _pdf_color(METHOD_COLORS.get(method, "#111827"))
            if len(points) > 1:
                path = [f"{_dash_command(method)} {color[0]:.4f} {color[1]:.4f} {color[2]:.4f} RG 3.0 w"]
                path.append(f"{points[0][0]:.2f} {points[0][1]:.2f} m")
                path.extend(f"{x:.2f} {y:.2f} l" for x, y in points[1:])
                path.append("S [] 0 d")
                commands.append(" ".join(path))
            for x, y in points:
                _draw_pdf_marker(commands, x, y, 4.8, color, METHOD_MARKERS.get(method, "circle"))

    legend_y = y1 + 42.0
    legend_entries = []
    total_w = 0.0
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        entry_w = 50.0 + len(label) * legend_size * 0.52
        legend_entries.append((method, label, entry_w))
        total_w += entry_w
    total_w += max(len(legend_entries) - 1, 0) * 18.0
    legend_x = (width - total_w) / 2.0
    for method, label, entry_w in legend_entries:
        color = _pdf_color(METHOD_COLORS.get(method, "#111827"))
        stroke_line(legend_x, legend_y + 5.0, legend_x + 30.0, legend_y + 5.0, color, 2.6, dash=_dash_command(method))
        _draw_pdf_marker(commands, legend_x + 15.0, legend_y + 5.0, 4.2, color, METHOD_MARKERS.get(method, "circle"))
        fill_text(legend_x + 39.0, legend_y - 1.0, label, legend_size)
        legend_x += entry_w + 18.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pdf(width, height, commands, out_path)


def _draw_compare_pdf(
    series: Sequence[Tuple[str, Sequence[dict]]],
    *,
    mode: str,
    metric: str,
    out_path: Path,
) -> None:
    methods = [method for method in METHOD_ORDER if any(
        str(row.get("method")) == method and str(row.get("decode_mode")) == mode
        for _, rows in series
        for row in rows
    )]
    budgets = sorted({float(row["retrieval_budget"]) for _, rows in series for row in rows if str(row.get("decode_mode")) == mode})
    recall_ks = sorted({int(row["recall_k"]) for _, rows in series for row in rows if str(row.get("decode_mode")) == mode})
    if not methods or not budgets or not recall_ks:
        raise SystemExit(f"no rows found for mode={mode!r}")
    if len(recall_ks) > 1:
        raise SystemExit("compare PDF expects one recall_k")

    width = 430.0
    height = 330.0
    plot_w = 300.0
    plot_h = 165.0
    x0, y0 = 78.0, 56.0
    x1, y1 = x0 + plot_w, y0 + plot_h
    y_top = min(1.0, max(float(row[metric]) for _, rows in series for row in rows if str(row.get("decode_mode")) == mode) * 1.10)
    y_top = max(y_top, 0.1)

    def x_pos(budget: float) -> float:
        if len(budgets) == 1:
            return x0 + plot_w / 2.0
        return x0 + budgets.index(float(budget)) * (plot_w / (len(budgets) - 1))

    def y_pos(value: float) -> float:
        return y0 + (float(value) / y_top) * plot_h

    commands: List[str] = [
        "1 1 1 rg 0 0 %.2f %.2f re f" % (width, height),
        "1 J 1 j",
    ]

    def stroke_line(
        xa: float,
        ya: float,
        xb: float,
        yb: float,
        color: Tuple[float, float, float],
        width_pt: float,
        dash: bool = False,
    ) -> None:
        dash_cmd = "[7 4] 0 d " if dash else "[] 0 d "
        commands.append(
            f"{dash_cmd}{color[0]:.4f} {color[1]:.4f} {color[2]:.4f} RG "
            f"{width_pt:.2f} w {xa:.2f} {ya:.2f} m {xb:.2f} {yb:.2f} l S [] 0 d"
        )

    def fill_text(x: float, y: float, text: object, size: int, *, align: str = "left") -> None:
        text = str(text)
        approx_w = len(text) * size * 0.52
        if align == "center":
            x -= approx_w / 2.0
        elif align == "right":
            x -= approx_w
        commands.append(f"0.0667 0.0941 0.1529 rg BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({_pdf_escape(text)}) Tj ET")

    stroke_line(x0, y0, x1, y0, (0.07, 0.09, 0.13), 1.5)
    stroke_line(x0, y0, x0, y1, (0.07, 0.09, 0.13), 1.5)

    label_size = 18
    tick_size = label_size
    legend_size = label_size
    fill_text((x0 + x1) / 2.0, 16.0, "Retrieval budget", label_size, align="center")
    commands.append("q 0 1 -1 0 24.00 122.00 cm")
    commands.append(f"0.0667 0.0941 0.1529 rg BT /F1 {label_size} Tf 0 0 Td ({_pdf_escape('Recall@100')}) Tj ET")
    commands.append("Q")

    for i in range(5):
        frac = i / 4.0
        y = y0 + frac * plot_h
        stroke_line(x0, y, x1, y, (0.85, 0.85, 0.85), 0.7)
        fill_text(x0 - 9.0, y - 5.0, f"{frac * y_top:.2f}", tick_size, align="right")

    for budget in budgets:
        x = x_pos(budget)
        stroke_line(x, y0, x, y0 - 4.0, (0.07, 0.09, 0.13), 1.2)
        fill_text(x, y0 - 21.0, f"{budget:g}", tick_size, align="center")

    method_positions = {
        "full_kmeans": (18.0, height - 52.0),
        "learned": (150.0, height - 52.0),
        "pair_sequential": (275.0, height - 52.0),
        "sequential": (54.0, height - 84.0),
        "pq": (54.0, height - 84.0),
    }
    fallback_positions = [
        (18.0, height - 52.0),
        (150.0, height - 52.0),
        (275.0, height - 52.0),
        (54.0, height - 84.0),
    ]
    for idx, method in enumerate(methods):
        color = _pdf_color(METHOD_COLORS.get(method, "#111827"))
        label = METHOD_LEGEND_LABELS.get(method, METHOD_LABELS.get(method, method))
        legend_x, legend_y = method_positions.get(method, fallback_positions[min(idx, len(fallback_positions) - 1)])
        stroke_line(legend_x, legend_y + 5.0, legend_x + 28.0, legend_y + 5.0, color, 2.5)
        _draw_pdf_marker(commands, legend_x + 14.0, legend_y + 5.0, 4.0, color, METHOD_MARKERS.get(method, "circle"))
        fill_text(legend_x + 38.0, legend_y, label, legend_size)

    marker_y = height - 84.0
    marker_x = 205.0
    for idx, (label, _) in enumerate(series):
        color = (0.12, 0.16, 0.22)
        dash = idx == 1
        stroke_line(marker_x, marker_y + 5.0, marker_x + 30.0, marker_y + 5.0, color, 2.5, dash=dash)
        if idx == 0:
            _draw_pdf_marker(commands, marker_x + 15.0, marker_y + 5.0, 4.0, color, "circle")
        else:
            _draw_pdf_marker(commands, marker_x + 15.0, marker_y + 5.0, 4.0, color, "square")
        fill_text(marker_x + 40.0, marker_y, label, legend_size)
        marker_x += 92.0

    for length_idx, (label, rows) in enumerate(series):
        dash = length_idx == 1
        for method in methods:
            raw_points = _points_for(rows, mode=mode, recall_k=recall_ks[0], method=method, metric=metric)
            if not raw_points:
                continue
            points = [(x_pos(budget), y_pos(value)) for budget, value in raw_points]
            color = _pdf_color(METHOD_COLORS.get(method, "#111827"))
            if len(points) > 1:
                dash_cmd = "[7 4] 0 d " if dash else "[] 0 d "
                path = [f"{dash_cmd}{color[0]:.4f} {color[1]:.4f} {color[2]:.4f} RG 2.8 w"]
                path.append(f"{points[0][0]:.2f} {points[0][1]:.2f} m")
                path.extend(f"{x:.2f} {y:.2f} l" for x, y in points[1:])
                path.append("S [] 0 d")
                commands.append(" ".join(path))
            for x, y in points:
                if length_idx == 0:
                    _draw_pdf_marker(commands, x, y, 4.2, color, "circle")
                else:
                    _draw_pdf_marker(commands, x, y, 4.2, color, "square")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pdf(width, height, commands, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot recall curves from recall_summary.json.")
    parser.add_argument("summary", type=Path, nargs="?")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--out-file", type=Path, default=None)
    parser.add_argument("--metrics", type=str, default="candidate_recall_mean")
    parser.add_argument("--modes", type=str, default="flat")
    parser.add_argument("--formats", type=str, default="pdf", help="Comma/space separated: pdf,svg")
    parser.add_argument("--compare-summaries", type=str, default="", help="Comma/space separated label:path entries for one combined PDF.")
    parser.add_argument(
        "--panel-summaries",
        type=str,
        default=DEFAULT_PANEL_SUMMARIES,
        help="Comma/space separated label:path entries for one 1xN panel PDF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.panel_summaries:
        specs = _parse_compare_summaries(args.panel_summaries)
        series = []
        for label, path in specs:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data.get("summary", [])
            if not rows:
                raise SystemExit(f"no summary rows in {path}")
            series.append((label, rows))
        if len(series) != 2:
            raise SystemExit("--panel-summaries currently expects exactly two label:path entries")
        metrics = [item.strip() for item in str(args.metrics).replace(",", " ").split() if item.strip()]
        modes = [item.strip() for item in str(args.modes).replace(",", " ").split() if item.strip()]
        if len(metrics) != 1 or len(modes) != 1:
            raise SystemExit("--panel-summaries expects exactly one metric and one mode")
        if args.out_file is not None:
            out_path = args.out_file
        else:
            out_path = DEFAULT_PANEL_OUT
        _draw_panel_pdf(series, mode=modes[0], metric=metrics[0], out_path=out_path)
        print(f"[plot] wrote {out_path}")
        return

    if args.compare_summaries:
        specs = _parse_compare_summaries(args.compare_summaries)
        series = []
        for label, path in specs:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data.get("summary", [])
            if not rows:
                raise SystemExit(f"no summary rows in {path}")
            series.append((label, rows))
        metrics = [item.strip() for item in str(args.metrics).replace(",", " ").split() if item.strip()]
        modes = [item.strip() for item in str(args.modes).replace(",", " ").split() if item.strip()]
        if len(metrics) != 1 or len(modes) != 1:
            raise SystemExit("--compare-summaries expects exactly one metric and one mode")
        if args.out_file is not None:
            out_path = args.out_file
        else:
            first_path = specs[0][1]
            labels = "_".join(label for label, _ in specs)
            out_path = first_path.parent.parent / f"recall_compare_{labels}_{metrics[0]}_{modes[0]}.pdf"
        _draw_compare_pdf(series, mode=modes[0], metric=metrics[0], out_path=out_path)
        print(f"[plot] wrote {out_path}")
        return

    if args.summary is None:
        raise SystemExit("provide summary or --compare-summaries")
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = data.get("summary", [])
    if not rows:
        raise SystemExit(f"no summary rows in {args.summary}")
    out_dir = args.out_dir or args.summary.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [item.strip() for item in str(args.metrics).replace(",", " ").split() if item.strip()]
    available_modes = sorted({str(row["decode_mode"]) for row in rows})
    modes = [item.strip() for item in str(args.modes).replace(",", " ").split() if item.strip()]
    if not modes:
        modes = available_modes
    formats = [item.strip().lower() for item in str(args.formats).replace(",", " ").split() if item.strip()]
    for metric in metrics:
        for row in rows:
            if metric not in row:
                raise SystemExit(f"metric {metric!r} missing from summary rows")
        for mode in modes:
            if mode not in available_modes:
                raise SystemExit(f"mode {mode!r} missing from summary rows; available: {available_modes}")
            for fmt in formats:
                if fmt == "svg":
                    out_path = out_dir / f"{metric}_{mode}.svg"
                    _draw_svg(rows, mode=mode, metric=metric, out_path=out_path)
                elif fmt == "pdf":
                    out_path = out_dir / f"{metric}_{mode}.pdf"
                    _draw_pdf(rows, mode=mode, metric=metric, out_path=out_path)
                else:
                    raise SystemExit(f"unsupported format {fmt!r}; supported: pdf,svg")
                print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
