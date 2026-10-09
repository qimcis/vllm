# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Lamport TP all-reduce + mHC post for the glm5next decoder.

Mirrors ``vllm/models/deepseek_v41/nvidia/ops/mhc.py`` (the ``AllReduceMHC``
kernel it builds is shared): a TP-partial sublayer output publishes into a
three-generation Lamport mailbox, and a PDL-chained consumer reduces it and
applies the mHC post-mix while fragments land. glm5next's seam collapses the
post-mixed streams with mixes computed from them (its reference math), so the
consumer stops after the post-mix and the layer's hc_pre does the collapse,
the RMSNorm and the next mixes.
"""

from typing import TYPE_CHECKING, cast

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )
    from vllm.models.deepseek_v41.nvidia.ops.cute_dsl import AllReduceMHC

logger = init_logger(__name__)

# Fused path pays only at small token counts.
MHC_ALL_REDUCE_MAX_TOKENS = 16

# The fused all-reduce + mHC post kernel, built once by init_mhc_all_reduce.
_all_reduce_mhc: "AllReduceMHC | None" = None


def supports_mhc_all_reduce(vllm_config: VllmConfig) -> bool:
    """Check the model shape, the SM100-family kernel, and the mailbox."""
    spec = vllm_config.speculative_config
    if spec is not None and spec.method in (
        "eagle3",
        "dflash",
        "dspark",
        "extract_hidden_states",
    ):
        # These drafters read raw decoder-layer outputs as aux states, which
        # the fused seam leaves partial.
        return False
    parallel = vllm_config.parallel_config
    config = vllm_config.model_config.hf_config
    if (
        not config.mhc
        or parallel.tensor_parallel_size == 1
        or parallel.enable_expert_parallel
        or parallel.use_sequence_parallel_moe
        or config.hidden_size != 4096
        or config.hc_mult != 4
        or not current_platform.is_device_capability_family(100)
    ):
        return False
    # The custom all-reduce's MNNVL buffers succeed exactly when NVLink
    # multicast does, which the fused kernel's own mailbox needs too.
    comm = cast("CudaCommunicator", get_tp_group().device_communicator).ca_comm
    return comm is not None and bool(comm.mnnvl_lamport_ag_multicast_ptr)


def init_mhc_all_reduce(vllm_config: VllmConfig) -> None:
    """Build the fused all-reduce + mHC post kernel.

    Collective over the TP group, so every rank must call it, and only when
    ``supports_mhc_all_reduce`` holds.
    """
    global _all_reduce_mhc
    from vllm.models.deepseek_v41.nvidia.ops.cute_dsl import AllReduceMHC

    config = vllm_config.model_config.hf_config
    _all_reduce_mhc = AllReduceMHC(
        hidden_size=config.hidden_size,
        hc_mult=config.hc_mult,
        max_num_tokens=MHC_ALL_REDUCE_MAX_TOKENS,
        top_k=config.num_experts_per_tok,
        device=current_platform.current_device(),
        # glm5next collapses with mixes computed from the reduced streams and
        # its monolithic TRTLLM router emits BF16 routing weights.
        collapse=False,
        fp32_weights=False,
    )
    logger.info_once(
        "glm5next mHC: CuTe DSL Lamport all-reduce fused with the mHC post for "
        "up to %d tokens.",
        MHC_ALL_REDUCE_MAX_TOKENS,
    )


def mhc_all_reduce_post(
    x: torch.Tensor | MoEOutput,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Return the post-mixed residual streams for a TP-partial sublayer output.

    An unfinalized MoE output publishes through the finalize variant, which
    folds the top-k reduction and the shared-expert add into the publish.
    """
    assert _all_reduce_mhc is not None, "init_mhc_all_reduce was not called"
    if isinstance(x, MoEOutput):
        routed = x.routed
        assert isinstance(routed, UnfinalizedMoEOutput)
        assert x.shared_output is not None
        return _all_reduce_mhc.finalize_post(
            routed.gemm2_permuted,
            routed.expert_weights,
            routed.expanded_idx_to_permuted_idx,
            x.shared_output,
            residual,
            post,
            comb,
        )
    return _all_reduce_mhc.reduce_post(x, residual, post, comb)
