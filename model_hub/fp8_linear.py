from __future__ import annotations

import torch


class Fp8Linear:
    """Compressed-tensors W8A8 FP8 linear for AdaptiveIMI runtime.

    Checkpoints store weights as [out_features, in_features] float8_e4m3fn
    with channel-wise [out_features, 1] dequantization scales. vLLM's FP8
    kernel expects a transposed [in_features, out_features] weight matrix.
    """

    def __init__(
        self,
        *,
        weight_t: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        out_features: int,
        device: str,
    ) -> None:
        self.weight = weight_t
        self.weight_scale = weight_scale
        self.bias = bias
        self.out_features = int(out_features)
        self.device = device
        self._cutlass_fp8_supported: bool | None = None

    @classmethod
    def from_tensors(
        cls,
        *,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        device: str,
        dtype: torch.dtype,
    ) -> "Fp8Linear":
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError(f"FP8 linear expects float8_e4m3fn weight, got {weight.dtype}")
        if weight_scale is None:
            raise ValueError("FP8 linear requires channel-wise weight_scale")
        out_features = int(weight.size(0))
        if tuple(weight_scale.shape) != (out_features, 1):
            raise ValueError(
                f"expected weight_scale shape {(out_features, 1)}, got {tuple(weight_scale.shape)}"
            )

        # vLLM's FP8 scaled-mm path expects [in_features, out_features]
        # with column-major strides.  Do not call contiguous() after the
        # transpose: that would turn the matrix row-major and cuBLASLt
        # rejects the row-major x row-major combination.
        weight_t = weight.t().to(device=device, non_blocking=True)
        scale = weight_scale.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        if bias is not None:
            bias = bias.to(device=device, dtype=dtype, non_blocking=True).contiguous()

        return cls(
            weight_t=weight_t,
            weight_scale=scale,
            bias=bias,
            out_features=out_features,
            device=device,
        )

    def _kernel_supported(self) -> bool:
        if self._cutlass_fp8_supported is None:
            try:
                from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
                    cutlass_fp8_supported,
                )

                self._cutlass_fp8_supported = bool(cutlass_fp8_supported())
            except Exception as exc:
                raise RuntimeError(
                    "vLLM FP8 runtime is required for compressed-tensors FP8 AdaptiveIMI. "
                    "Install/use a vLLM build with FP8 CUTLASS support."
                ) from exc
        return self._cutlass_fp8_supported

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import apply_fp8_linear

        if hidden_states.device != self.weight.device:
            hidden_states = hidden_states.to(self.weight.device, non_blocking=True)
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        return apply_fp8_linear(
            input=hidden_states,
            weight=self.weight,
            weight_scale=self.weight_scale,
            input_scale=None,
            bias=self.bias,
            cutlass_fp8_supported=self._kernel_supported(),
            use_per_token_if_dynamic=True,
        )
