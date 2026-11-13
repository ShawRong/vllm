# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read a ResServe NVFP4 sidecar and shard an extended-K checkpoint.

The sidecar is ``residue_kernel_metadata.json`` next to the checkpoint.
Layer records use ``runtime_mode`` (``standard``, ``extended_k``, ``mext_r1``),
``num_salient``, ``salient_indices``, and ``extended_linear_dims``. Channel
selection is already finished in that file. This module only turns it into
widths, a tensor-parallel plan, and a ``uint8[K/8]`` mask.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import torch

_FUSED = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}
_SALIENT_BLOCK = 8
_SF_VEC = 16
_SF_SWIZZLE_BLOCKS = 4
_RESIDUE_SF_GROUP = 128


class ResidueLoadError(ValueError):
    """A sidecar or shard shape this loader will not guess."""


@dataclass(frozen=True)
class ShardPlan:
    """One rank's cut of ``[base K | salient]``."""

    k_base: int
    num_salient: int
    tp_size: int
    tp_rank: int

    @property
    def k_ext(self) -> int:
        return self.k_base + self.num_salient

    @property
    def base_shard(self) -> int:
        return self.k_base // self.tp_size

    @property
    def salient_shard(self) -> int:
        return self.num_salient // self.tp_size

    @property
    def k_ext_shard(self) -> int:
        return self.base_shard + self.salient_shard

    def validate(self) -> None:
        tp = self.tp_size
        if tp <= 0 or self.tp_rank not in range(tp):
            raise ResidueLoadError(f"bad tp_rank={self.tp_rank} for tp_size={tp}")
        if self.k_base % tp or self.num_salient % tp:
            raise ResidueLoadError(
                f"K_base={self.k_base} and num_salient={self.num_salient} "
                f"must both divide tp_size={tp}"
            )
        if self.base_shard % _RESIDUE_SF_GROUP:
            raise ResidueLoadError(
                f"K_base/tp={self.base_shard} must be a multiple of "
                f"{_RESIDUE_SF_GROUP} so a rank boundary does not split a "
                "residue scale group"
            )
        grid = _SF_VEC * _SF_SWIZZLE_BLOCKS
        for name, width in (
            ("K_base/tp", self.base_shard),
            ("num_salient/tp", self.salient_shard),
        ):
            if width % grid:
                raise ResidueLoadError(f"{name}={width} must be a multiple of {grid}")
        if self.base_shard % _SALIENT_BLOCK:
            raise ResidueLoadError(
                f"K_base/tp={self.base_shard} must be a multiple of {_SALIENT_BLOCK}"
            )

    def ranges(self, channels_per_element: int) -> tuple[tuple[int, int], tuple[int, int]]:
        if channels_per_element <= 0:
            raise ResidueLoadError("channels_per_element must be positive")
        for name, value in (
            ("k_base", self.k_base),
            ("base_shard", self.base_shard),
            ("salient_shard", self.salient_shard),
        ):
            if value % channels_per_element:
                raise ResidueLoadError(
                    f"{name}={value} is not divisible by {channels_per_element}"
                )
        base = (
            self.tp_rank * self.base_shard // channels_per_element,
            self.base_shard // channels_per_element,
        )
        salient = (
            (self.k_base + self.tp_rank * self.salient_shard) // channels_per_element,
            self.salient_shard // channels_per_element,
        )
        return base, salient

    def gather(self, full: torch.Tensor, channels_per_element: int, dim: int = -1):
        (base_off, base_len), (sal_off, sal_len) = self.ranges(channels_per_element)
        return torch.cat(
            (
                full.narrow(dim, base_off, base_len),
                full.narrow(dim, sal_off, sal_len),
            ),
            dim=dim,
        )

    def local_salient_indices(self, full_indices: torch.Tensor) -> torch.Tensor:
        lo = self.tp_rank * self.base_shard
        hi = lo + self.base_shard
        mine = full_indices[(full_indices >= lo) & (full_indices < hi)]
        if mine.numel() != self.salient_shard:
            raise ResidueLoadError(
                f"rank {self.tp_rank} has {mine.numel()} salient channels, "
                f"expected {self.salient_shard}"
            )
        run = full_indices.narrow(0, self.tp_rank * self.salient_shard, self.salient_shard)
        if not torch.equal(mine, run):
            raise ResidueLoadError("salient indices are not sorted into contiguous rank runs")
        return mine - lo


def plan_from_partition(
    *,
    extended_dim: int,
    input_size_per_partition: int,
    input_size: int,
    num_salient: int,
    tp_rank: int,
) -> ShardPlan | None:
    """Return None when the input is not sharded (column parallel)."""
    if input_size_per_partition >= input_size:
        return None
    if input_size % input_size_per_partition:
        raise ResidueLoadError(
            f"input_size={input_size} is not a multiple of "
            f"input_size_per_partition={input_size_per_partition}"
        )
    k_base = extended_dim - num_salient
    if k_base != input_size:
        raise ResidueLoadError(
            f"extended_dim={extended_dim} minus num_salient={num_salient} "
            f"is {k_base}, not input_size={input_size}"
        )
    plan = ShardPlan(
        k_base=k_base,
        num_salient=num_salient,
        tp_size=input_size // input_size_per_partition,
        tp_rank=tp_rank,
    )
    plan.validate()
    return plan


def interleave_extended(full: torch.Tensor, plan: ShardPlan, channels_per_element: int):
    """Reorder ``[base | salient]`` into ``[base_0 | sal_0 | base_1 | ...]``."""
    pieces = [
        replace(plan, tp_rank=rank).gather(full, channels_per_element)
        for rank in range(plan.tp_size)
    ]
    return torch.cat(pieces, dim=-1).contiguous()


def channel_mask_from_indices(salient_indices: torch.Tensor, num_channels: int) -> torch.Tensor:
    if num_channels % 8:
        raise ResidueLoadError("num_channels must be a multiple of 8")
    masks = torch.zeros(num_channels // 8, dtype=torch.uint8, device=salient_indices.device)
    if salient_indices.numel() == 0:
        return masks
    masks.index_add_(
        0,
        salient_indices // 8,
        (1 << (salient_indices % 8)).to(dtype=torch.uint8),
    )
    return masks


def residue_per_8(num_salient: int, k_base: int) -> int:
    if k_base <= 0 or num_salient * 8 % k_base:
        raise ResidueLoadError(f"num_salient={num_salient} is not 1, 2, or 4 per 8 channels of {k_base}")
    count = num_salient * 8 // k_base
    if count not in (1, 2, 4):
        raise ResidueLoadError(f"residue_per_8={count} is not 1, 2, or 4")
    return count


def _aliases(layer_name: str) -> list[str]:
    names = [layer_name]
    for source, target in (
        ("language_model.model.", "model.language_model."),
        ("model.language_model.", "language_model.model."),
    ):
        if layer_name.startswith(source):
            names.append(target + layer_name[len(source) :])
            break
    parent, _, leaf = layer_name.rpartition(".")
    if leaf in _FUSED:
        prefix = f"{parent}." if parent else ""
        names.extend(prefix + suffix for suffix in _FUSED[leaf])
    return names


def _same_record(left: dict, right: dict) -> bool:
    keys = ("runtime_mode", "num_salient", "salient_indices")
    return all(left.get(key) == right.get(key) for key in keys)


def load_sidecar(model_dir: str | os.PathLike[str]) -> dict:
    path = Path(model_dir) / "residue_kernel_metadata.json"
    if not path.is_file():
        return {}
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ResidueLoadError(f"residue metadata must be an object: {path}")
    return data


def cached_sidecar(quant_config) -> dict:
    data = getattr(quant_config, "_resserve_metadata", None)
    if data is not None:
        return data
    model_dir = os.environ.get("RESIDUE_VLLM_MODEL_DIR")
    if not model_dir:
        from vllm.config import get_current_vllm_config_or_none

        cfg = get_current_vllm_config_or_none()
        model = cfg.model_config.model if cfg is not None and cfg.model_config else None
        if model and Path(model).is_dir():
            model_dir = model
    data = load_sidecar(model_dir) if model_dir else {}
    quant_config._resserve_metadata = data
    return data


def layer_record(quant_config, layer_name: str) -> dict | None:
    data = cached_sidecar(quant_config)
    layers = data.get("layers", {})
    found = [layers[name] for name in _aliases(layer_name) if name in layers]
    if not found:
        return None
    first = found[0]
    if any(not _same_record(first, other) for other in found[1:]):
        raise ResidueLoadError(f"fused shards of {layer_name} disagree in the residue sidecar")
    return first


def extended_dim(quant_config, layer_name: str) -> int | None:
    dims = cached_sidecar(quant_config).get("extended_linear_dims", {})
    hits = [dims[name] for name in _aliases(layer_name) if name in dims]
    if not hits:
        return None
    if any(hit != hits[0] for hit in hits[1:]):
        raise ResidueLoadError(f"fused shards of {layer_name} disagree on extended K")
    return int(hits[0])


def runtime_mode(quant_config, layer_name: str) -> str:
    record = layer_record(quant_config, layer_name)
    if record is None:
        return "standard"
    return record.get("runtime_mode", "extended_k")
