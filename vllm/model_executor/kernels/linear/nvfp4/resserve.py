# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in FlashInfer ResServe calls for an NVFP4 linear layer.

These functions are not registered with the NVFP4 kernel selector.
``backend="auto"`` does not reach them. They target
https://github.com/flashinfer-ai/flashinfer/pull/6889.

MEXT and KEXT are separate paths.

MEXT does not change weight loading. The checkpoint stays base-K NVFP4:
``weight`` is uint8 ``[N, K/2]`` and ``weight_scale`` is the float8 swizzle
``(padded_N, K_tiles * 4)``. Do not pass the CUTLASS K-padded copies; the
swizzle cannot be sliced back to base K. The activation quantizer returns
paired rows, and the GEMM folds them.

KEXT has no GEMM of its own. The checkpoint weight must already append the
selected channels, in ascending index order, before it is quantized. Each
tensor-parallel rank loads ``K / tp + num_salient / tp`` along K, on scale-group
boundaries. ``channel_mask_from_indices`` turns that rank's indices into the
mask. The activation quantizer appends the same channels, and the existing
``flashinfer_scaled_fp4_mm`` consumes the extended pair.

``input_global_scale_inv`` is the quantization multiplier already given to
``scaled_fp4_quant``. ``alpha`` is ``layer.alpha``.
"""

from __future__ import annotations

import torch

from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm


def channel_mask_from_indices(
    salient_indices: torch.Tensor, num_channels: int
) -> torch.Tensor:
    """Pack base-K channel indices into ``uint8[K/8]``.

    Bit 0 of byte ``i`` selects channel ``8 * i``. ``num_channels`` is the
    rank-local base K.
    """
    if num_channels % 8:
        raise ValueError("num_channels must be a multiple of 8")
    masks = torch.zeros(
        num_channels // 8, dtype=torch.uint8, device=salient_indices.device
    )
    if salient_indices.numel() == 0:
        return masks
    masks.index_add_(
        0,
        salient_indices // 8,
        (1 << (salient_indices % 8)).to(dtype=torch.uint8),
    )
    return masks


def _restore(out: torch.Tensor, x: torch.Tensor, bias: torch.Tensor | None):
    if bias is not None:
        out = out + bias
    return out.view(*x.shape[:-1], out.shape[-1])


def apply_mext(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    alpha: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fold a base-K NVFP4 weight with row-pair activations.

    ``x`` is FP16 or BF16 ``[..., K]``. ``weight`` is unpadded uint8
    ``[N, K/2]``.
    """
    from flashinfer.resserve import mm_nvfp4_residue, nvfp4_quantize_residue

    x2 = x.reshape(-1, x.shape[-1])
    packed, act_scale = nvfp4_quantize_residue(x2, input_global_scale_inv)
    out = mm_nvfp4_residue(
        packed,
        weight,
        act_scale,
        weight_scale,
        alpha,
        out_dtype=x.dtype,
    )
    return _restore(out, x, bias)


def apply_kext(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    alpha: torch.Tensor,
    channel_mask: torch.Tensor,
    residue_per_8: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize selected-channel residues and call the stock NVFP4 GEMM.

    ``weight`` and ``weight_scale`` are the extended checkpoint tensors in the
    layout ``flashinfer_scaled_fp4_mm`` already accepts. ``channel_mask``
    matches the channels appended to that weight. ``residue_per_8`` is 1, 2,
    or 4.
    """
    from flashinfer.resserve import nvfp4_quantize_kext

    x2 = x.reshape(-1, x.shape[-1])
    packed, act_scale = nvfp4_quantize_kext(
        x2, input_global_scale_inv, channel_mask, residue_per_8
    )
    out = flashinfer_scaled_fp4_mm(
        packed,
        weight,
        act_scale,
        weight_scale,
        alpha,
        x.dtype,
        backend="cute-dsl",
    )
    return _restore(out, x, bias)
