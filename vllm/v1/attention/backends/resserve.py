# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ResServe page32 decode with stock FlashInfer prefill and KV writes."""

from dataclasses import replace

import torch

from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    nvfp4_split_data_scale,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferImpl,
    FlashInferMetadataBuilder,
    FlashInferTrtllmAPIDecode,
)


class ResServeMetadataBuilder(FlashInferMetadataBuilder):
    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        if vllm_config.parallel_config.decode_context_parallel_size != 1:
            raise NotImplementedError("ResServe decode does not support DCP")
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        self._capture_max_seq_len = vllm_config.model_config.max_model_len

    def build_for_cudagraph_capture(self, common_attn_metadata):
        md = self.build(0, common_attn_metadata)
        if md.num_decodes:
            capacity = md.decode.block_tables.shape[1] * 32
            md.decode.max_seq_len = min(self._capture_max_seq_len, capacity)
        return md


class ResServeFlashInferImpl(FlashInferImpl):
    can_return_lse_for_decode = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            not self.is_kvcache_nvfp4
            or self.head_size not in (128, 256)
            or self.num_heads % self.num_kv_heads
            or self.num_heads // self.num_kv_heads not in (4, 6, 8)
        ):
            raise NotImplementedError("ResServe requires NVFP4 KV, D128/D256, GQA4/6/8")
        if (
            self.alibi_slopes is not None
            or self.window_left != -1
            or self.logits_soft_cap not in (None, 0.0)
            or self.sinks is not None
            or self.kv_sharing_target_layer_name is not None
        ):
            raise NotImplementedError("Unsupported ResServe attention variant")
        self.supports_quant_query_input = False

    def fused_output_quant_supported(self, quant_key):
        return False

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("ResServe page32 returns BF16 output")
        if attn_metadata is None:
            return output.zero_()
        if query.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise ValueError("ResServe requires original BF16 Q and BF16 output")
        md = attn_metadata
        if md.use_cascade or not md.causal:
            raise NotImplementedError("ResServe requires causal, non-cascade attention")
        n, end = md.num_decode_tokens, md.num_actual_tokens
        if md.num_prefill_tokens:
            # Keep the stock prefill and its precision; only decode is replaced.
            from vllm import _custom_ops

            prefill_md = replace(
                md,
                num_actual_tokens=md.num_prefill_tokens,
                num_decodes=0,
                num_decode_tokens=0,
                decode=None,
                slot_mapping=md.slot_mapping[n:end],
            )
            q_pre = query[n:end].reshape(-1, self.num_heads * self.head_size)
            q8, _ = _custom_ops.scaled_fp8_quant(q_pre, scale=layer._q_scale)
            q8 = q8.view(-1, self.num_heads, self.head_size)
            super().forward(
                layer,
                q8,
                key[n:end] if key is not None else None,
                value[n:end] if value is not None else None,
                kv_cache,
                prefill_md,
                output[n:end],
            )
        if n:
            if n != md.num_decodes or not isinstance(
                md.decode, FlashInferTrtllmAPIDecode
            ):
                raise NotImplementedError(
                    "ResServe requires one token per decode request"
                )
            from flashinfer.resserve import paged_decode_residue

            k, v = kv_cache.split(self.num_kv_heads, dim=1)
            order = FlashInferBackend.get_kv_cache_stride_order()
            k = canonicalize_singleton_dim_strides(k.permute(*order))
            v = canonicalize_singleton_dim_strides(v.permute(*order))
            kd, ks = nvfp4_split_data_scale(k)
            vd, vs = nvfp4_split_data_scale(v)
            if kd.shape[2] != 32:
                raise ValueError("ResServe requires --block-size 32")
            pages, heads = kd.shape[:2]
            paged_decode_residue(
                query[:n].reshape(n, self.num_heads, self.head_size),
                kd,
                vd,
                ks,
                vs.view(pages, heads, 8, self.head_size // 16, 4),
                md.decode.block_tables,
                md.decode.seq_lens,
                sm_scale=self.scale,
                key_scale=layer._k_scale_float,
                value_scale=layer._v_scale_float,
                max_seq_len=max(1, md.decode.max_seq_len),
                out=output[:n].view(n, self.num_heads, self.head_size),
            )
        return output


class ResServeFlashInferBackend(FlashInferBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["nvfp4"]

    @staticmethod
    def get_name():
        return "FLASHINFER_RESSERVE"

    @staticmethod
    def get_impl_cls():
        return ResServeFlashInferImpl

    @staticmethod
    def get_builder_cls():
        return ResServeMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [32]

    @classmethod
    def get_supported_head_sizes(cls):
        return [128, 256]

    @classmethod
    def supports_sink(cls):
        return False
