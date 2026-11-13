# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelOpt NVFP4 linear method for a ResServe sidecar.

``extended_k`` widens the checkpoint weight and runs ``nvfp4_quantize_kext``
plus the stock NVFP4 GEMM. ``mext_r1`` keeps the base-K weight and runs
``nvfp4_quantize_residue`` plus ``mm_nvfp4_residue``. Layers the sidecar does
not name stay on ``ModelOptLinearMethod``. MoE experts are unchanged.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.linear import (
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization.modelopt import (
    LINEAR_METHOD_BUILDERS,
    ModelOptLinearMethod,
    resolve,
)
from vllm.model_executor.layers.quantization.resserve_load import (
    ResidueLoadError,
    channel_mask_from_indices,
    extended_dim,
    interleave_extended,
    layer_record,
    plan_from_partition,
    residue_per_8,
    runtime_mode,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    nvfp4_weight_padding_bytes,
    swizzle_blockscale,
)


def _tp_rank(sharded: bool) -> int:
    if not sharded:
        return 0
    from vllm.distributed import get_tensor_model_parallel_rank

    return get_tensor_model_parallel_rank()


@register_weight_loader_v2_supported_method
class ResServeNvFp4LinearMethod(ModelOptLinearMethod):
    """NVFP4 linear method that consumes ``residue_kernel_metadata.json``."""

    def __init__(self, quant_config, prefix: str, mode: str) -> None:
        spec, ctx, format_scheme = resolve("NVFP4", quant_config, prefix)
        super().__init__(spec, ctx, format_scheme)
        self.quant_config = quant_config
        self.prefix = prefix
        self.mode = mode

    def create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        layer_name = getattr(layer, "prefix", "") or self.prefix
        record = layer_record(self.quant_config, layer_name) or {}
        width = extended_dim(self.quant_config, layer_name)
        plan = None
        if self.mode == "extended_k":
            if width is None:
                raise ResidueLoadError(f"{layer_name} is extended_k but has no extended K")
            num_salient = int(record.get("num_salient", 0))
            plan = plan_from_partition(
                extended_dim=width,
                input_size_per_partition=input_size_per_partition,
                input_size=input_size,
                num_salient=num_salient,
                tp_rank=_tp_rank(input_size_per_partition < input_size),
            )
            if plan is None:
                input_size_per_partition = width
                input_size = width
            else:
                input_size_per_partition = plan.k_ext_shard
                input_size = width
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        if plan is not None:
            layer._resserve_plan = plan
            _wrap_loaders(layer, plan)

    def process_weights_after_loading(self, layer) -> None:
        raw_weight = raw_scale = None
        if self.mode == "mext_r1":
            # Fold reads the original base-K pair. The stock loader may pad K
            # afterwards for CUTLASS, so keep the copies from before that.
            raw_weight = layer.weight.detach().clone()
            raw_scale = layer.weight_scale.detach().clone()
        super().process_weights_after_loading(layer)
        if self.mode == "mext_r1":
            layer.register_buffer("_resserve_weight", raw_weight, persistent=False)
            layer.register_buffer(
                "_resserve_weight_scale",
                swizzle_blockscale(raw_scale),
                persistent=False,
            )
            return
        if nvfp4_weight_padding_bytes(layer):
            raise ResidueLoadError(
                f"{self.prefix} extended K was padded by the stock NVFP4 loader; "
                "the KEXT GEMM expects the exported width"
            )
        plan = getattr(layer, "_resserve_plan", None)
        record = layer_record(self.quant_config, getattr(layer, "prefix", "") or self.prefix) or {}
        indices = record.get("salient_indices")
        num_salient = int(record.get("num_salient", 0))
        if not indices or len(indices) != num_salient:
            raise ResidueLoadError(
                f"{self.prefix} extended_k sidecar needs salient_indices of length {num_salient}"
            )
        full = torch.tensor(indices, dtype=torch.int64, device=layer.weight.device)
        if plan is None:
            local = full
            k_base = layer.weight.shape[1] * 2 - num_salient
        else:
            local = plan.local_salient_indices(full)
            k_base = plan.base_shard
        count = residue_per_8(local.numel(), k_base)
        layer.register_buffer(
            "channel_mask",
            channel_mask_from_indices(local, k_base),
            persistent=False,
        )
        layer.residue_per_8 = count

    def apply(self, layer, x, bias=None):
        from vllm.model_executor.kernels.linear.nvfp4.resserve import (
            apply_kext,
            apply_mext,
        )

        if self.mode == "mext_r1":
            out = apply_mext(
                x,
                layer._resserve_weight,
                layer._resserve_weight_scale,
                layer.input_global_scale_inv,
                layer.alpha,
                bias,
            )
        else:
            out = apply_kext(
                x,
                layer.weight,
                layer.weight_scale,
                layer.input_global_scale_inv,
                layer.alpha,
                layer.channel_mask,
                layer.residue_per_8,
                bias,
            )
        return out[..., : layer.output_size_per_partition]


def _wrap_loaders(layer, plan) -> None:
    """Permute extended columns before vLLM's contiguous shard narrow."""
    for name, channels_per_element in (("weight", 2), ("weight_scale", 16)):
        param = getattr(layer, name, None)
        orig = getattr(param, "weight_loader", None)
        if orig is None:
            continue

        def make(orig_loader, channels):
            expected = plan.k_ext // channels

            def loader(param_, loaded_weight, *args, **kwargs):
                # shortcut: only the full checkpoint row is permuted. If vLLM
                # starts passing an already-narrowed shard, drop this wrap and
                # gather that shard explicitly.
                if loaded_weight.shape[-1] == expected:
                    loaded_weight = interleave_extended(loaded_weight, plan, channels)
                return orig_loader(param_, loaded_weight, *args, **kwargs)

            return loader

        param.weight_loader = make(orig, channels_per_element)


def register_resserve_nvfp4() -> None:
    """Install the sidecar-aware builder. Plain NVFP4 layers stay generic."""

    def build(config, prefix: str):
        mode = runtime_mode(config, prefix)
        if mode == "standard":
            spec, ctx, format_scheme = resolve("NVFP4", config, prefix)
            return ModelOptLinearMethod(spec, ctx, format_scheme)
        if mode not in ("extended_k", "mext_r1"):
            raise ResidueLoadError(f"{prefix} has unknown runtime_mode {mode!r}")
        return ResServeNvFp4LinearMethod(config, prefix, mode)

    LINEAR_METHOD_BUILDERS["NVFP4"] = build
