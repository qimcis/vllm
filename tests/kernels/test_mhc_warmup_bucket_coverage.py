# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin mHC TileLang warmup coverage of every reachable split-k bucket."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("MHC TileLang warmup tests require CUDA", allow_module_level=True)

from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    _MHC_PRE_BIG_FUSE_TILELANG_KERNEL,
    compute_num_split,
    mhc_fused_post_pre_split_config,
)
from vllm.utils.math_utils import cdiv

H100_SMS = 132
HIDDEN_SIZE = 4096
HC_MULT = 4
HC_HIDDEN_SIZE = HIDDEN_SIZE * HC_MULT
MAX_TOKENS = 4096


@pytest.fixture(autouse=True)
def h100_sm_count(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(multi_processor_count=H100_SMS),
    )
    compute_num_split.cache_clear()
    yield
    compute_num_split.cache_clear()


def _registration(
    *,
    max_tokens: int = MAX_TOKENS,
    include_pre_gemm_splits: bool = True,
    include_broadcast_splits: bool = False,
) -> list:
    return _MHC_PRE_BIG_FUSE_TILELANG_KERNEL.get_warmup_keys(
        SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens)
        ),
        hidden_size=HIDDEN_SIZE,
        hc_mult=HC_MULT,
        use_norm_weight=True,
        include_pre_gemm_splits=include_pre_gemm_splits,
        include_broadcast_splits=include_broadcast_splits,
        rms_eps=1e-5,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
        norm_eps=(1e-5, 1e-5),
        broadcast_norm_eps=1e-5,
    )


def _runtime_keys(*, max_tokens: int = MAX_TOKENS) -> set:
    keys = set()
    for num_tokens in range(1, max_tokens + 1):
        fused_config = mhc_fused_post_pre_split_config(num_tokens, HIDDEN_SIZE, HC_MULT)
        n_splits = (
            fused_config[1]
            if fused_config is not None
            else compute_num_split(64, HC_HIDDEN_SIZE, cdiv(num_tokens, 64))
        )
        keys.add(
            _MHC_PRE_BIG_FUSE_TILELANG_KERNEL.dispatch(
                hidden_size=HIDDEN_SIZE,
                hc_mult=HC_MULT,
                n_splits=n_splits,
                is_broadcast=False,
                use_norm_weight=True,
                rms_eps=1e-5,
                hc_pre_eps=1e-6,
                hc_sinkhorn_eps=1e-6,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=20,
                norm_eps=1e-5,
                broadcast_norm_eps=1e-5,
            )
        )
    return keys


def test_glm5next_registration_covers_every_runtime_key():
    warmup_keys = set(_registration())
    runtime_keys = _runtime_keys()
    missing = runtime_keys - warmup_keys
    assert not missing, f"runtime keys not warmed: {missing}"


def test_one_warmup_key_per_split_bucket():
    warmup_keys = _registration()
    reachable_splits = {
        compute_num_split(64, HC_HIDDEN_SIZE, grid)
        for grid in range(1, cdiv(MAX_TOKENS, 64) + 1)
    }
    warmed_splits = {key.n_splits for key in warmup_keys if not key.is_broadcast}
    assert reachable_splits <= warmed_splits
    assert warmed_splits - reachable_splits == {1}, (
        "beyond the non-deep-gemm n_splits=1 fallback, exactly one key per bucket"
    )


def test_warmup_covers_the_3_cta_bucket_of_a_132_sm_part():
    assert compute_num_split(64, HC_HIDDEN_SIZE, 3) == 44
    warmup_keys = _registration()
    assert any(key.n_splits == 44 and not key.is_broadcast for key in warmup_keys), (
        "the 129-192 token bucket (3 CTAs, n_splits=44) must be warmed"
    )


def test_deepseek_registration_keeps_broadcast_keys_covered():
    warmup_keys = set(_registration(include_broadcast_splits=True))
    broadcast_runtime_keys = set()
    for num_tokens in (1, MAX_TOKENS):
        n_splits = compute_num_split(64, HIDDEN_SIZE, cdiv(num_tokens, 64))
        broadcast_runtime_keys.add(
            _MHC_PRE_BIG_FUSE_TILELANG_KERNEL.dispatch(
                hidden_size=HIDDEN_SIZE,
                hc_mult=HC_MULT,
                n_splits=n_splits,
                is_broadcast=True,
                use_norm_weight=True,
                rms_eps=1e-5,
                hc_pre_eps=1e-6,
                hc_sinkhorn_eps=1e-6,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=20,
                norm_eps=1e-5,
                broadcast_norm_eps=1e-5,
            )
        )
    assert broadcast_runtime_keys <= warmup_keys


def test_without_deep_gemm_only_fused_and_single_splits_warm():
    warmup_keys = _registration(include_pre_gemm_splits=False)
    assert {key.n_splits for key in warmup_keys} == {1, 8}
