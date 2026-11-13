# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the ResServe sidecar shard rule."""

import json
from pathlib import Path

import torch

from vllm.model_executor.layers.quantization.resserve_load import (
    ResidueLoadError,
    channel_mask_from_indices,
    extended_dim,
    interleave_extended,
    layer_record,
    load_sidecar,
    plan_from_partition,
    residue_per_8,
    runtime_mode,
)


def _plan():
    # K_base 256, two ranks: each base shard is 128 and each salient shard is 64.
    return plan_from_partition(
        extended_dim=256 + 128,
        input_size_per_partition=128,
        input_size=256,
        num_salient=128,
        tp_rank=1,
    )


def test_plan_reorders_base_then_salient():
    plan = _plan()
    assert plan is not None
    full = torch.arange(384)
    rank1 = plan.gather(full, 1)
    assert torch.equal(rank1[:128], torch.arange(128, 256))
    assert torch.equal(rank1[128:], torch.arange(256 + 64, 256 + 128))
    interleaved = interleave_extended(full, plan, 1)
    # Rank 0 occupies the first k_ext_shard channels of the permuted tensor.
    assert torch.equal(interleaved[:128], torch.arange(128))
    assert torch.equal(interleaved[128:192], torch.arange(256, 256 + 64))


def test_local_indices_and_mask():
    plan = _plan()
    indices = torch.arange(0, 256, 2)
    local = plan.local_salient_indices(indices)
    assert local.numel() == 64
    assert int(local[0]) == 0
    mask = channel_mask_from_indices(local, plan.base_shard)
    assert mask.shape == (16,)
    assert int(mask[0]) == 0b01010101
    assert residue_per_8(local.numel(), plan.base_shard) == 4


def test_column_parallel_has_no_plan():
    assert (
        plan_from_partition(
            extended_dim=384,
            input_size_per_partition=256,
            input_size=256,
            num_salient=128,
            tp_rank=0,
        )
        is None
    )


def test_bad_group_boundary_is_rejected():
    try:
        plan_from_partition(
            extended_dim=192,
            input_size_per_partition=64,
            input_size=128,
            num_salient=64,
            tp_rank=0,
        )
    except ResidueLoadError:
        return
    raise AssertionError("a shard that splits a residue scale group must fail")


def test_sidecar_fused_names(tmp_path: Path):
    payload = {
        "extended_linear_dims": {
            "model.layers.0.q_proj": 384,
            "model.layers.0.k_proj": 384,
            "model.layers.0.v_proj": 384,
        },
        "layers": {
            "model.layers.0.q_proj": {
                "runtime_mode": "extended_k",
                "num_salient": 128,
                "salient_indices": [0, 1],
            },
            "model.layers.0.k_proj": {
                "runtime_mode": "extended_k",
                "num_salient": 128,
                "salient_indices": [0, 1],
            },
            "model.layers.0.v_proj": {
                "runtime_mode": "extended_k",
                "num_salient": 128,
                "salient_indices": [0, 1],
            },
            "model.layers.0.o_proj": {"runtime_mode": "mext_r1"},
        },
    }
    path = tmp_path / "residue_kernel_metadata.json"
    path.write_text(json.dumps(payload))
    data = load_sidecar(tmp_path)
    config = type("Cfg", (), {"_resserve_metadata": data})()
    assert runtime_mode(config, "model.layers.0.qkv_proj") == "extended_k"
    assert extended_dim(config, "model.layers.0.qkv_proj") == 384
    assert layer_record(config, "model.layers.0.o_proj")["runtime_mode"] == "mext_r1"
    assert runtime_mode(config, "model.layers.0.down_proj") == "standard"
