from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_head_groups(groups: object, *, head_dim: int, m: int, head_idx: int) -> List[List[int]]:
    if not isinstance(groups, list) or len(groups) != m:
        raise ValueError(f"head {head_idx} must contain exactly {m} subspace groups")
    if head_dim % 2 != 0:
        raise ValueError("learned 8-way partition currently expects even head_dim")

    half = head_dim // 2
    seen = set()
    parsed: List[List[int]] = []
    for group_idx, group in enumerate(groups):
        if not isinstance(group, list) or not group:
            raise ValueError(f"head {head_idx} group {group_idx} is empty")
        parsed_group = [int(item) for item in group]
        for pair in parsed_group:
            if pair < 0 or pair >= half:
                raise ValueError(f"head {head_idx} pair id {pair} out of range [0, {half})")
            if pair in seen:
                raise ValueError(f"head {head_idx} pair id {pair} appears more than once")
            seen.add(pair)
        parsed.append(parsed_group)

    if len(seen) != half:
        missing = sorted(set(range(half)) - seen)
        raise ValueError(f"head {head_idx} partition does not cover all RoPE pairs; missing sample={missing[:8]}")
    return parsed


def load_hybrid_partition_specs(
    partition_path: str | Path,
    *,
    layer_num: int,
    kv_heads: int,
    head_dim: int,
    required_m: int = 8,
) -> List[Optional[Dict[str, object]]]:
    payload = _load_json(partition_path)
    layer_groups = payload.get("layer_groups") or payload.get("groups")
    if not isinstance(layer_groups, list) or not layer_groups:
        raise ValueError("hybrid 8-way partition file must contain a non-empty layer_groups list")

    specs: List[Optional[Dict[str, object]]] = [None] * int(layer_num)
    for group in layer_groups:
        if not isinstance(group, dict):
            raise ValueError("each layer group must be an object")

        layer_start = int(group.get("layer_start", group.get("start", 0)))
        layer_end = int(group.get("layer_end", group.get("end", layer_num)))
        if bool(group.get("layer_end_inclusive", False)):
            layer_end += 1
        layer_start = max(layer_start, 0)
        layer_end = min(layer_end, int(layer_num))
        if layer_start >= layer_end:
            continue

        m = int(group.get("M", group.get("subspace_parts", 0)) or 0)
        if m != required_m:
            raise ValueError(f"hybrid v2 requires M={required_m}, got M={m}")

        heads_obj = group.get("heads")
        if not isinstance(heads_obj, dict):
            raise ValueError("hybrid partition layer group must contain heads dict")

        heads: Dict[str, Dict[str, object]] = {}
        for head_idx in range(int(kv_heads)):
            raw_head = heads_obj.get(str(head_idx), heads_obj.get(head_idx))
            if raw_head is None:
                raise ValueError(f"hybrid partition missing head {head_idx}")
            if isinstance(raw_head, dict) and raw_head.get("dim_groups") is not None:
                raise ValueError(
                    "hybrid_8way_v2 currently accepts layer-head RoPE pair 'groups' only; "
                    f"head {head_idx} provided dim_groups"
                )
            raw_groups = raw_head.get("groups") if isinstance(raw_head, dict) else raw_head
            heads[str(head_idx)] = {
                "groups": _validate_head_groups(raw_groups, head_dim=head_dim, m=required_m, head_idx=head_idx)
            }

        spec: Dict[str, object] = {
            "enabled": True,
            "subspace_parts": required_m,
            "M": required_m,
            "heads": heads,
        }

        for layer_idx in range(layer_start, layer_end):
            specs[layer_idx] = spec

    missing_layers = [idx for idx, spec in enumerate(specs) if spec is None]
    if missing_layers:
        raise ValueError(f"hybrid partition does not cover layers; missing sample={missing_layers[:8]}")
    return specs
