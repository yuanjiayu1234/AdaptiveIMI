from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import resolve_model_path
from experiments.imi_partition.eval.evaluate_two_stage_recall import (
    _parse_layers,
    _read_jsonl,
    _resolve_modeling_module,
)


DETAIL_FIELDS = [
    "prompt_idx",
    "layer",
    "kv_head",
    "q_head",
    "group_size",
    "seq_len",
    "key_start",
    "key_end",
    "search_tokens",
    "budget_kind",
    "budget_label",
    "budget_value",
    "budget_tokens",
    "budget_ratio",
    "attention_mass",
    "missing_mass",
    "relative_output_error",
    "output_cosine",
    "full_entropy",
]

THRESHOLD_FIELDS = [
    "prompt_idx",
    "layer",
    "kv_head",
    "q_head",
    "group_size",
    "seq_len",
    "key_start",
    "key_end",
    "search_tokens",
    "mass_threshold",
    "required_tokens",
    "required_ratio",
    "full_entropy",
]


@dataclass
class LayerQKV:
    queries: torch.Tensor  # [q_heads, tokens, dim], CPU fp16/bf16
    keys: torch.Tensor  # [kv_heads, tokens, dim], CPU fp16/bf16
    values: torch.Tensor  # [kv_heads, tokens, dim], CPU fp16/bf16
    key_start: int
    key_end: int
    seq_len: int


class LastTokenQKVCollector:
    def __init__(
        self,
        *,
        layer_ids: Sequence[int],
        static_start: int,
        static_end: int,
        max_search_tokens: int,
    ) -> None:
        self.layer_ids = set(int(layer) for layer in layer_ids)
        self.static_start = max(int(static_start), 0)
        self.static_end = max(int(static_end), 0)
        self.max_search_tokens = max(int(max_search_tokens), 0)
        self.active_layer: Optional[int] = None
        self.records: Dict[int, LayerQKV] = {}
        self._pending_values: Dict[int, torch.Tensor] = {}

    def clear(self) -> None:
        self.records.clear()
        self._pending_values.clear()

    def _search_range(self, seq_len: int) -> Tuple[int, int]:
        key_start = min(self.static_start, seq_len - 1)
        key_end = seq_len - self.static_end if self.static_end > 0 else seq_len - 1
        key_end = min(max(key_end, key_start), seq_len - 1)
        if self.max_search_tokens > 0 and key_end - key_start > self.max_search_tokens:
            key_start = key_end - self.max_search_tokens
        return int(key_start), int(key_end)

    def add_values(self, layer_idx: int, value_states: torch.Tensor) -> None:
        layer_idx = int(layer_idx)
        if layer_idx not in self.layer_ids:
            return
        if value_states.dim() != 4 or value_states.size(0) != 1:
            return
        seq_len = int(value_states.size(2))
        if seq_len < 2:
            return
        key_start, key_end = self._search_range(seq_len)
        if key_end <= key_start:
            return
        self._pending_values[layer_idx] = (
            value_states[0, :, key_start:key_end, :].detach().to(device="cpu").contiguous()
        )

    def add_qk(self, layer_idx: int, query_states: torch.Tensor, key_states: torch.Tensor) -> None:
        layer_idx = int(layer_idx)
        if layer_idx not in self.layer_ids:
            return
        if query_states.dim() != 4 or key_states.dim() != 4:
            return
        if query_states.size(0) != 1 or key_states.size(0) != 1:
            raise ValueError("oracle top-k sufficiency evaluator expects batch_size=1")

        seq_len = int(key_states.size(2))
        if seq_len < 2:
            return
        key_start, key_end = self._search_range(seq_len)
        if key_end <= key_start:
            return

        values = self._pending_values.get(layer_idx)
        if values is None:
            return
        q_all = query_states[0].detach().to(device="cpu").contiguous()
        k_search = key_states[0, :, key_start:key_end, :].detach().to(device="cpu").contiguous()
        if int(values.size(1)) != int(k_search.size(1)):
            return
        self.records[layer_idx] = LayerQKV(
            queries=q_all,
            keys=k_search,
            values=values,
            key_start=key_start,
            key_end=key_end,
            seq_len=seq_len,
        )


def _parse_ints(value: str) -> List[int]:
    return [int(item) for item in value.replace(",", " ").split() if item.strip()]


def _parse_floats(value: str) -> List[float]:
    return [float(item) for item in value.replace(",", " ").split() if item.strip()]


def _pct_label(value: float) -> str:
    pct = float(value) * 100.0
    text = ("%g" % pct).replace(".", "p")
    return f"{text}%"


def _iter_prompts(args: argparse.Namespace) -> Iterable[str]:
    if args.jsonl is None:
        raise ValueError("provide --jsonl")
    yield from _read_jsonl(args.jsonl, args.text_field, args.limit)


def _value_projection_to_kv_states(attn_module, hidden_states: torch.Tensor) -> torch.Tensor:
    if not hasattr(attn_module, "v_proj"):
        raise AttributeError("attention module has no v_proj")
    value_states = attn_module.v_proj(hidden_states)
    bsz, seq_len, _ = value_states.shape
    kv_heads = int(
        getattr(
            attn_module,
            "num_key_value_heads",
            getattr(getattr(attn_module, "config", None), "num_key_value_heads", 0),
        )
    )
    head_dim = int(getattr(attn_module, "head_dim", 0))
    if kv_heads <= 0:
        hidden_size = int(getattr(getattr(attn_module, "config", None), "hidden_size", 0))
        num_heads = int(getattr(getattr(attn_module, "config", None), "num_attention_heads", 0))
        if hidden_size <= 0 or num_heads <= 0:
            raise ValueError("cannot infer num_key_value_heads for value projection")
        head_dim = head_dim or hidden_size // num_heads
        kv_heads = int(value_states.size(-1)) // head_dim
    if head_dim <= 0:
        head_dim = int(value_states.size(-1)) // kv_heads
    return value_states.view(bsz, seq_len, kv_heads, head_dim).transpose(1, 2).contiguous()


def _budget_specs(n_tokens: int, fixed_ks: Sequence[int], pct_budgets: Sequence[float]) -> List[Tuple[str, str, float, int]]:
    specs: List[Tuple[str, str, float, int]] = []
    seen = set()
    for k in fixed_ks:
        kk = max(1, min(int(k), int(n_tokens)))
        key = ("fixed", kk)
        if key not in seen:
            seen.add(key)
            specs.append(("fixed", f"top{kk}", float(kk), kk))
    for pct in pct_budgets:
        kk = max(1, min(int(round(float(pct) * float(n_tokens))), int(n_tokens)))
        label = _pct_label(float(pct))
        key = ("percent", label, kk)
        if key not in seen:
            seen.add(key)
            specs.append(("percent", label, float(pct), kk))
    return specs


def evaluate_record(
    *,
    prompt_idx: int,
    layer_idx: int,
    record: LayerQKV,
    fixed_ks: Sequence[int],
    pct_budgets: Sequence[float],
    mass_thresholds: Sequence[float],
    device: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    queries = record.queries.to(device=device)
    keys_by_kv = record.keys.to(device=device)
    values_by_kv = record.values.to(device=device)
    q_heads = int(queries.size(0))
    kv_heads = int(keys_by_kv.size(0))
    n_tokens = int(keys_by_kv.size(1))
    head_dim = int(keys_by_kv.size(2))
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads must be divisible by kv_heads, got {q_heads=} {kv_heads=}")
    group_size = q_heads // kv_heads

    detail_rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []
    scale = 1.0 / math.sqrt(float(head_dim))

    for q_head in range(q_heads):
        kv_head = q_head // group_size
        query = queries[q_head, int(record.seq_len) - 1, :].float()
        keys = keys_by_kv[kv_head].float()
        values = values_by_kv[kv_head].float()
        scores = torch.mv(keys, query) * scale
        probs = torch.softmax(scores, dim=0)
        ordered = torch.argsort(scores, descending=True)
        ordered_probs = probs.index_select(0, ordered)
        cdf = torch.cumsum(ordered_probs, dim=0)
        full_out = torch.mv(values.transpose(0, 1), probs)
        full_norm = torch.linalg.vector_norm(full_out).clamp_min(1e-8)
        entropy = float((-(probs * probs.clamp_min(1e-30).log()).sum()).item())

        for threshold in mass_thresholds:
            threshold_f = float(threshold)
            pos = torch.searchsorted(cdf, torch.tensor(threshold_f, device=cdf.device), right=False)
            required = min(int(pos.item()) + 1, n_tokens)
            threshold_rows.append({
                "prompt_idx": int(prompt_idx),
                "layer": int(layer_idx),
                "kv_head": int(kv_head),
                "q_head": int(q_head),
                "group_size": int(group_size),
                "seq_len": int(record.seq_len),
                "key_start": int(record.key_start),
                "key_end": int(record.key_end),
                "search_tokens": int(n_tokens),
                "mass_threshold": threshold_f,
                "required_tokens": int(required),
                "required_ratio": float(required) / float(max(n_tokens, 1)),
                "full_entropy": entropy,
            })

        for budget_kind, budget_label, budget_value, budget_tokens in _budget_specs(n_tokens, fixed_ks, pct_budgets):
            top_idx = ordered[:budget_tokens]
            mass = float(probs.index_select(0, top_idx).sum().item())
            sparse_scores = scores.index_select(0, top_idx)
            sparse_probs = torch.softmax(sparse_scores, dim=0)
            sparse_values = values.index_select(0, top_idx)
            sparse_out = torch.mv(sparse_values.transpose(0, 1), sparse_probs)
            rel_err = float((torch.linalg.vector_norm(sparse_out - full_out) / full_norm).item())
            cosine = float(torch.nn.functional.cosine_similarity(sparse_out, full_out, dim=0).item())
            detail_rows.append({
                "prompt_idx": int(prompt_idx),
                "layer": int(layer_idx),
                "kv_head": int(kv_head),
                "q_head": int(q_head),
                "group_size": int(group_size),
                "seq_len": int(record.seq_len),
                "key_start": int(record.key_start),
                "key_end": int(record.key_end),
                "search_tokens": int(n_tokens),
                "budget_kind": budget_kind,
                "budget_label": budget_label,
                "budget_value": float(budget_value),
                "budget_tokens": int(budget_tokens),
                "budget_ratio": float(budget_tokens) / float(max(n_tokens, 1)),
                "attention_mass": mass,
                "missing_mass": 1.0 - mass,
                "relative_output_error": rel_err,
                "output_cosine": cosine,
                "full_entropy": entropy,
            })
    return detail_rows, threshold_rows


def _init_csv(path: Path, fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()


def _append_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        for row in rows:
            writer.writerow(row)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _summarize_details(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault(str(row["budget_label"]), []).append(row)
    out = []
    for label, items in sorted(buckets.items(), key=lambda kv: (str(kv[1][0]["budget_kind"]), float(kv[1][0]["budget_tokens"]))):
        masses = [float(item["attention_mass"]) for item in items]
        errors = [float(item["relative_output_error"]) for item in items]
        cosines = [float(item["output_cosine"]) for item in items]
        ratios = [float(item["budget_ratio"]) for item in items]
        out.append({
            "budget_label": label,
            "budget_kind": str(items[0]["budget_kind"]),
            "samples": len(items),
            "budget_ratio_mean": _mean(ratios),
            "attention_mass_mean": _mean(masses),
            "missing_mass_mean": 1.0 - _mean(masses),
            "relative_output_error_mean": _mean(errors),
            "output_cosine_mean": _mean(cosines),
        })
    return out


def _summarize_thresholds(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[float, List[Dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault(float(row["mass_threshold"]), []).append(row)
    out = []
    for threshold, items in sorted(buckets.items()):
        tokens = [float(item["required_tokens"]) for item in items]
        ratios = [float(item["required_ratio"]) for item in items]
        out.append({
            "mass_threshold": float(threshold),
            "samples": len(items),
            "required_tokens_mean": _mean(tokens),
            "required_ratio_mean": _mean(ratios),
        })
    return out


def _summarize_layer_budgets(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[int, str], List[Dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault((int(row["layer"]), str(row["budget_label"])), []).append(row)
    out = []
    for (layer, label), items in sorted(buckets.items()):
        masses = [float(item["attention_mass"]) for item in items]
        errors = [float(item["relative_output_error"]) for item in items]
        cosines = [float(item["output_cosine"]) for item in items]
        ratios = [float(item["budget_ratio"]) for item in items]
        tokens = [float(item["budget_tokens"]) for item in items]
        out.append({
            "layer": int(layer),
            "budget_label": label,
            "budget_kind": str(items[0]["budget_kind"]),
            "samples": len(items),
            "budget_tokens_mean": _mean(tokens),
            "budget_ratio_mean": _mean(ratios),
            "attention_mass_mean": _mean(masses),
            "missing_mass_mean": 1.0 - _mean(masses),
            "relative_output_error_mean": _mean(errors),
            "output_cosine_mean": _mean(cosines),
        })
    return out


def _summarize_layer_thresholds(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[int, float], List[Dict[str, object]]] = {}
    for row in rows:
        buckets.setdefault((int(row["layer"]), float(row["mass_threshold"])), []).append(row)
    out = []
    for (layer, threshold), items in sorted(buckets.items()):
        tokens = [float(item["required_tokens"]) for item in items]
        ratios = [float(item["required_ratio"]) for item in items]
        out.append({
            "layer": int(layer),
            "mass_threshold": float(threshold),
            "samples": len(items),
            "required_tokens_mean": _mean(tokens),
            "required_ratio_mean": _mean(ratios),
        })
    return out


def write_plots(detail_csv: Path, threshold_csv: Path, out_dir: Path) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[oracle-topk] plotting skipped: {exc}", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(detail_csv)
    thresholds = pd.read_csv(threshold_csv)
    if detail.empty:
        return

    labels = list(dict.fromkeys(detail["budget_label"].astype(str).tolist()))
    mass_data = [detail.loc[detail["budget_label"].astype(str) == label, "attention_mass"].to_numpy() for label in labels]
    err_data = [detail.loc[detail["budget_label"].astype(str) == label, "relative_output_error"].to_numpy() for label in labels]

    plt.figure(figsize=(max(8, len(labels) * 0.8), 4.5))
    plt.boxplot(mass_data, labels=labels, showfliers=False)
    plt.ylabel("Attention mass retained")
    plt.xlabel("Oracle exact top-k budget")
    plt.ylim(0.0, 1.02)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "attention_mass_box.png", dpi=200)
    plt.close()

    plt.figure(figsize=(max(8, len(labels) * 0.8), 4.5))
    plt.boxplot(err_data, labels=labels, showfliers=False)
    plt.ylabel("Relative attention output error")
    plt.xlabel("Oracle exact top-k budget")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "output_error_box.png", dpi=200)
    plt.close()

    grouped = detail.groupby(["search_tokens", "budget_label"], as_index=False)["attention_mass"].mean()
    plt.figure(figsize=(7, 4.5))
    for label in labels:
        sub = grouped[grouped["budget_label"].astype(str) == label].sort_values("search_tokens")
        if not sub.empty:
            plt.plot(sub["search_tokens"], sub["attention_mass"], marker="o", label=label)
    plt.xlabel("Searchable context tokens")
    plt.ylabel("Mean attention mass retained")
    plt.ylim(0.0, 1.02)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "attention_mass_by_context.png", dpi=200)
    plt.close()

    if not thresholds.empty:
        plt.figure(figsize=(7, 4.5))
        for threshold in sorted(thresholds["mass_threshold"].unique()):
            vals = thresholds.loc[thresholds["mass_threshold"] == threshold, "required_ratio"].sort_values().to_numpy()
            if len(vals) == 0:
                continue
            y = [(idx + 1) / len(vals) for idx in range(len(vals))]
            plt.plot(vals, y, label=f"mass {threshold:g}")
        plt.xlabel("Required token ratio")
        plt.ylabel("CDF")
        plt.xlim(left=0.0)
        plt.ylim(0.0, 1.02)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "required_ratio_cdf.png", dpi=200)
        plt.close()

        threshold_pivot = thresholds.pivot_table(
            index="layer",
            columns="mass_threshold",
            values="required_ratio",
            aggfunc="mean",
        ).sort_index()
        plt.figure(figsize=(7.5, max(4, 0.28 * len(threshold_pivot.index))))
        image = plt.imshow(threshold_pivot.to_numpy(), aspect="auto", cmap="magma")
        plt.colorbar(image, label="Mean required token ratio")
        plt.yticks(range(len(threshold_pivot.index)), [str(v) for v in threshold_pivot.index])
        plt.xticks(range(len(threshold_pivot.columns)), [f"{v:g}" for v in threshold_pivot.columns])
        plt.xlabel("Attention mass threshold")
        plt.ylabel("Layer")
        plt.tight_layout()
        plt.savefig(out_dir / "required_ratio_by_layer_heatmap.png", dpi=200)
        plt.close()

    mass_pivot = detail.pivot_table(
        index="layer",
        columns="budget_label",
        values="attention_mass",
        aggfunc="mean",
    ).reindex(columns=labels).sort_index()
    plt.figure(figsize=(max(8, 0.75 * len(labels)), max(4, 0.28 * len(mass_pivot.index))))
    image = plt.imshow(mass_pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(image, label="Mean attention mass retained")
    plt.yticks(range(len(mass_pivot.index)), [str(v) for v in mass_pivot.index])
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.xlabel("Oracle exact top-k budget")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(out_dir / "attention_mass_by_layer_heatmap.png", dpi=200)
    plt.close()

    err_pivot = detail.pivot_table(
        index="layer",
        columns="budget_label",
        values="relative_output_error",
        aggfunc="mean",
    ).reindex(columns=labels).sort_index()
    plt.figure(figsize=(max(8, 0.75 * len(labels)), max(4, 0.28 * len(err_pivot.index))))
    image = plt.imshow(err_pivot.to_numpy(), aspect="auto", cmap="inferno")
    plt.colorbar(image, label="Mean relative output error")
    plt.yticks(range(len(err_pivot.index)), [str(v) for v in err_pivot.index])
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.xlabel("Oracle exact top-k budget")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(out_dir / "output_error_by_layer_heatmap.png", dpi=200)
    plt.close()

    hardest = (
        detail.groupby("layer", as_index=False)
        .agg(attention_mass_mean=("attention_mass", "mean"), output_error_mean=("relative_output_error", "mean"))
        .sort_values("layer")
    )
    plt.figure(figsize=(8, 4.5))
    plt.plot(hardest["layer"], 1.0 - hardest["attention_mass_mean"], marker="o", label="Missing attention mass")
    plt.plot(hardest["layer"], hardest["output_error_mean"], marker="o", label="Output error")
    plt.xlabel("Layer")
    plt.ylabel("Mean difficulty")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "layer_difficulty_mean.png", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oracle exact top-k attention sufficiency study: mass retained and output error."
    )
    parser.add_argument("--model", type=str, default="llama-3.1-8b")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--text-field", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--static-start", type=int, default=0)
    parser.add_argument("--static-end", type=int, default=0)
    parser.add_argument("--max-search-tokens", type=int, default=0)
    parser.add_argument("--fixed-ks", type=str, default="32,64,128,256,512")
    parser.add_argument("--pct-budgets", type=str, default="0.01,0.025,0.05,0.10,0.20")
    parser.add_argument("--mass-thresholds", type=str, default="0.8,0.9,0.95")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-device", type=str, default="")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-threshold-csv", type=Path, default=None)
    parser.add_argument("--out-summary", type=Path, default=None)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    eval_device = args.eval_device or args.device
    fixed_ks = _parse_ints(args.fixed_ks)
    pct_budgets = _parse_floats(args.pct_budgets)
    mass_thresholds = _parse_floats(args.mass_thresholds)

    model_path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(args.device)
    model.eval()

    num_layers = int(model.config.num_hidden_layers)
    layer_ids = _parse_layers(args.layers, num_layers)
    collector = LastTokenQKVCollector(
        layer_ids=layer_ids,
        static_start=args.static_start,
        static_end=args.static_end,
        max_search_tokens=args.max_search_tokens,
    )

    modeling_module = _resolve_modeling_module(model_path, args.model)
    original_apply_rope = modeling_module.apply_rotary_pos_emb
    original_forwards = []

    def wrapped_apply_rotary_pos_emb(*rope_args, **rope_kwargs):
        q_embed, k_embed = original_apply_rope(*rope_args, **rope_kwargs)
        if collector.active_layer is not None:
            collector.add_qk(collector.active_layer, q_embed, k_embed)
        return q_embed, k_embed

    modeling_module.apply_rotary_pos_emb = wrapped_apply_rotary_pos_emb
    for layer_idx, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        original_forward = attention.forward
        original_forwards.append((attention, original_forward))

        def make_forward(attn_module, fn, idx):
            def wrapped_forward(*forward_args, **forward_kwargs):
                hidden_states = forward_kwargs.get("hidden_states")
                if hidden_states is None and forward_args:
                    hidden_states = forward_args[0]
                collector.active_layer = idx
                try:
                    if hidden_states is not None and idx in collector.layer_ids:
                        values = _value_projection_to_kv_states(attn_module, hidden_states)
                        collector.add_values(idx, values)
                    return fn(*forward_args, **forward_kwargs)
                finally:
                    collector.active_layer = None

            return wrapped_forward

        attention.forward = make_forward(attention, original_forward, layer_idx)

    threshold_csv = args.out_threshold_csv
    if threshold_csv is None:
        threshold_csv = args.out_csv.with_name(args.out_csv.stem + "_thresholds.csv")
    _init_csv(args.out_csv, DETAIL_FIELDS)
    _init_csv(threshold_csv, THRESHOLD_FIELDS)
    print(f"[oracle-topk] streaming detail csv to {args.out_csv}", flush=True)
    print(f"[oracle-topk] streaming threshold csv to {threshold_csv}", flush=True)

    all_detail_rows: List[Dict[str, object]] = []
    all_threshold_rows: List[Dict[str, object]] = []
    prompt_idx = 0
    try:
        with torch.inference_mode():
            for prompt in _iter_prompts(args):
                collector.clear()
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                )
                input_ids = encoded["input_ids"].to(args.device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(args.device)
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                token_count = int(input_ids.size(1))
                print(
                    f"[oracle-topk] prompt={prompt_idx} tokens={token_count} collected_layers={len(collector.records)}",
                    flush=True,
                )
                for layer_idx in layer_ids:
                    record = collector.records.get(layer_idx)
                    if record is None:
                        continue
                    detail_rows, threshold_rows = evaluate_record(
                        prompt_idx=prompt_idx,
                        layer_idx=layer_idx,
                        record=record,
                        fixed_ks=fixed_ks,
                        pct_budgets=pct_budgets,
                        mass_thresholds=mass_thresholds,
                        device=eval_device,
                    )
                    all_detail_rows.extend(detail_rows)
                    all_threshold_rows.extend(threshold_rows)
                    _append_csv(args.out_csv, DETAIL_FIELDS, detail_rows)
                    _append_csv(threshold_csv, THRESHOLD_FIELDS, threshold_rows)
                    print(
                        f"[oracle-topk] prompt={prompt_idx} layer={layer_idx} detail_rows={len(detail_rows)} threshold_rows={len(threshold_rows)}",
                        flush=True,
                    )
                prompt_idx += 1
    finally:
        modeling_module.apply_rotary_pos_emb = original_apply_rope
        for attention, original_forward in original_forwards:
            attention.forward = original_forward

    summary = {
        "metadata": {
            "model": args.model,
            "model_path": model_path,
            "jsonl": str(args.jsonl),
            "layers": layer_ids,
            "limit": int(args.limit),
            "max_length": int(args.max_length),
            "static_start": int(args.static_start),
            "static_end": int(args.static_end),
            "max_search_tokens": int(args.max_search_tokens),
            "fixed_ks": fixed_ks,
            "pct_budgets": pct_budgets,
            "mass_thresholds": mass_thresholds,
            "dtype": str(args.dtype),
            "device": str(args.device),
            "eval_device": str(eval_device),
        },
        "budget_summary": _summarize_details(all_detail_rows),
        "threshold_summary": _summarize_thresholds(all_threshold_rows),
        "layer_budget_summary": _summarize_layer_budgets(all_detail_rows),
        "layer_threshold_summary": _summarize_layer_thresholds(all_threshold_rows),
    }
    if args.out_summary is not None:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[oracle-topk] wrote {args.out_summary}", flush=True)

    plot_dir = args.plot_dir
    if plot_dir is None and not args.no_plots:
        plot_dir = args.out_csv.with_suffix("")
    if plot_dir is not None and not args.no_plots:
        write_plots(args.out_csv, threshold_csv, plot_dir)
        print(f"[oracle-topk] wrote plots to {plot_dir}", flush=True)


if __name__ == "__main__":
    main()
