from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


@dataclass
class DependencyResult:
    dependency: torch.Tensor
    corr: torch.Tensor
    gaussian_mi: torch.Tensor
    num_tokens: int
    head_dim: int
    num_pairs: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "dependency": self.dependency.cpu(),
            "corr": self.corr.cpu(),
            "gaussian_mi": self.gaussian_mi.cpu(),
            "num_tokens": int(self.num_tokens),
            "head_dim": int(self.head_dim),
            "num_pairs": int(self.num_pairs),
        }


def layer_groups(num_layers: int, group_count: int = 3) -> List[Tuple[int, int]]:
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    groups: List[Tuple[int, int]] = []
    for group_idx in range(group_count):
        start = (num_layers * group_idx) // group_count
        end = (num_layers * (group_idx + 1)) // group_count
        if start < end:
            groups.append((start, end))
    return groups


def normalize_offdiag(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if matrix.dim() != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError("matrix must be square")
    out = matrix.float().clone()
    n = out.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=out.device)
    scale = out[mask].mean().clamp_min(eps)
    out = out / scale
    out.fill_diagonal_(0.0)
    return out


def _pair_view(keys: torch.Tensor) -> torch.Tensor:
    if keys.dim() != 2:
        raise ValueError("keys must have shape [tokens, head_dim]")
    head_dim = int(keys.size(-1))
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    front = keys[:, :half].float()
    back = keys[:, half:].float()
    return torch.stack((front, back), dim=-1)


def _whiten_pairs(pair_values: torch.Tensor, eps: float) -> torch.Tensor:
    # pair_values: [tokens, pairs, 2]
    centered = pair_values - pair_values.mean(dim=0, keepdim=True)
    denom = max(int(centered.size(0)) - 1, 1)
    cov = torch.einsum("npd,npe->pde", centered, centered) / float(denom)
    eye = torch.eye(2, dtype=cov.dtype, device=cov.device).expand_as(cov)
    cov = cov + eps * eye
    eigvals, eigvecs = torch.linalg.eigh(cov)
    inv_sqrt = eigvecs @ torch.diag_embed(torch.rsqrt(eigvals.clamp_min(eps))) @ eigvecs.transpose(-1, -2)
    return torch.einsum("npd,pde->npe", centered, inv_sqrt)


def compute_pair_dependency(
    keys: torch.Tensor,
    *,
    eps: float = 1e-5,
    mi_weight: float = 0.25,
    normalize: bool = True,
) -> DependencyResult:
    """Estimate RoPE-pair dependency for one layer/head or one aggregated scope.

    Args:
        keys: RoPE-applied keys with shape [tokens, head_dim]. For NeoX-style
            RoPE the pair unit is (u, u + head_dim / 2).
        eps: Numerical regularizer for whitening and logdet.
        mi_weight: Extra weight for Gaussian-MI dependency. The returned
            dependency is corr + mi_weight * gaussian_mi after normalization.
        normalize: Normalize corr and MI by off-diagonal means before combining.
    """

    if keys.numel() == 0:
        raise ValueError("keys must be non-empty")
    pairs = _pair_view(keys)
    num_tokens = int(pairs.size(0))
    num_pairs = int(pairs.size(1))
    if num_tokens < 4:
        raise ValueError("at least 4 tokens are required to estimate dependency")

    whitened = _whiten_pairs(pairs, eps=eps)
    denom = max(num_tokens - 1, 1)
    cross = torch.einsum("nua,nvb->uvab", whitened, whitened) / float(denom)
    corr = torch.linalg.matrix_norm(cross, ord="fro", dim=(-2, -1))
    corr.fill_diagonal_(0.0)

    cc_t = cross @ cross.transpose(-1, -2)
    eye = torch.eye(2, dtype=cc_t.dtype, device=cc_t.device)
    stable = eye - cc_t
    stable = stable + eps * eye
    det = stable[..., 0, 0] * stable[..., 1, 1] - stable[..., 0, 1] * stable[..., 1, 0]
    gaussian_mi = -0.5 * torch.log(det.clamp_min(eps))
    gaussian_mi = torch.nan_to_num(gaussian_mi, nan=0.0, posinf=0.0, neginf=0.0)
    gaussian_mi.fill_diagonal_(0.0)

    corr_for_score = normalize_offdiag(corr, eps=eps) if normalize else corr
    mi_for_score = normalize_offdiag(gaussian_mi, eps=eps) if normalize else gaussian_mi
    dependency = corr_for_score + float(mi_weight) * mi_for_score
    dependency = 0.5 * (dependency + dependency.transpose(0, 1))
    dependency.fill_diagonal_(0.0)

    return DependencyResult(
        dependency=dependency.cpu(),
        corr=corr.cpu(),
        gaussian_mi=gaussian_mi.cpu(),
        num_tokens=num_tokens,
        head_dim=int(keys.size(-1)),
        num_pairs=num_pairs,
    )


def tensor_to_lhtd(keys: torch.Tensor, layout: str) -> torch.Tensor:
    layout = layout.lower()
    if layout == "lhtd":
        if keys.dim() != 4:
            raise ValueError("layout lhtd expects tensor [layers, heads, tokens, dim]")
        return keys
    if layout == "slhtd":
        if keys.dim() != 5:
            raise ValueError("layout slhtd expects tensor [samples, layers, heads, tokens, dim]")
        s, l, h, t, d = keys.shape
        return keys.permute(1, 2, 0, 3, 4).reshape(l, h, s * t, d)
    if layout == "htd":
        if keys.dim() != 3:
            raise ValueError("layout htd expects tensor [heads, tokens, dim]")
        return keys.unsqueeze(0)
    if layout == "td":
        if keys.dim() != 2:
            raise ValueError("layout td expects tensor [tokens, dim]")
        return keys.view(1, 1, keys.size(0), keys.size(1))
    if layout == "auto":
        if keys.dim() == 5:
            return tensor_to_lhtd(keys, "slhtd")
        if keys.dim() == 4:
            return tensor_to_lhtd(keys, "lhtd")
        if keys.dim() == 3:
            return tensor_to_lhtd(keys, "htd")
        if keys.dim() == 2:
            return tensor_to_lhtd(keys, "td")
    raise ValueError(f"unsupported key tensor layout: {layout}")


def iter_scope_keys(
    keys_lhtd: torch.Tensor,
    *,
    scope: str,
    group_count: int = 3,
) -> Iterable[Tuple[str, torch.Tensor, Dict[str, int]]]:
    """Yield [tokens, dim] tensors for the requested partition granularity."""

    if keys_lhtd.dim() != 4:
        raise ValueError("keys_lhtd must have shape [layers, heads, tokens, dim]")
    num_layers, num_heads, _, _ = keys_lhtd.shape
    scope = scope.lower()

    if scope == "global":
        values = keys_lhtd.reshape(-1, keys_lhtd.size(-1))
        yield (
            "global",
            values,
            {
                "layer_start": 0,
                "layer_end": int(num_layers),
                "head_count": int(num_heads),
                "scope_kind": "global",
            },
        )
        return

    if scope == "layer_group":
        for group_idx, (start, end) in enumerate(layer_groups(num_layers, group_count)):
            values = keys_lhtd[start:end, :, :, :].reshape(-1, keys_lhtd.size(-1))
            yield (
                f"layer_group{group_idx}_layers{start}-{end - 1}",
                values,
                {
                    "layer_group": int(group_idx),
                    "layer_start": int(start),
                    "layer_end": int(end),
                    "head_count": int(num_heads),
                    "scope_kind": "layer_group",
                },
            )
        return

    if scope == "layer":
        for layer in range(num_layers):
            values = keys_lhtd[layer, :, :, :].reshape(-1, keys_lhtd.size(-1))
            yield (
                f"layer{layer}",
                values,
                {
                    "layer": int(layer),
                    "layer_start": int(layer),
                    "layer_end": int(layer + 1),
                    "head_count": int(num_heads),
                    "scope_kind": "layer",
                },
            )
        return

    if scope == "global_head":
        for head in range(num_heads):
            values = keys_lhtd[:, head, :, :].reshape(-1, keys_lhtd.size(-1))
            yield f"global_head{head}", values, {"head": int(head), "layer_group": -1, "scope_kind": "global_head"}
        return

    if scope == "layer_group_head":
        for group_idx, (start, end) in enumerate(layer_groups(num_layers, group_count)):
            for head in range(num_heads):
                values = keys_lhtd[start:end, head, :, :].reshape(-1, keys_lhtd.size(-1))
                yield (
                    f"layer_group{group_idx}_layers{start}-{end - 1}_head{head}",
                    values,
                    {
                        "head": int(head),
                        "layer_group": int(group_idx),
                        "layer_start": int(start),
                        "layer_end": int(end),
                        "scope_kind": "layer_group_head",
                    },
                )
        return

    if scope == "layer_head":
        for layer in range(num_layers):
            for head in range(num_heads):
                yield (
                    f"layer{layer}_head{head}",
                    keys_lhtd[layer, head, :, :],
                    {"head": int(head), "layer": int(layer), "layer_group": -1, "scope_kind": "layer_head"},
                )
        return

    raise ValueError("scope must be one of: global, layer_group, layer, global_head, layer_group_head, layer_head")
