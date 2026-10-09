# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    prepare_inputs_to_capture,
)
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model

from vllm.v1.worker.gpu.input_batch import InputBatch


class MTPSpeculator(TargetDependentARSpeculator):
    share_mtp_topk_indices: bool = False

    def __init__(self, vllm_config, device: torch.device):
        super().__init__(vllm_config, device)
        # Split step-0 for uniform spec-verify shapes: precompute KV for all
        # rows; run attention/MoE/indexer scoring on the last valid row only.
        self._step0_split = False
        self._step0_region2_states: dict = {}
        self._step0_req_arange: torch.Tensor | None = None
        self._capturing_step0 = False

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_model = load_eagle_model(target_model, self.vllm_config)
        spec_config = self.vllm_config.speculative_config
        draft_hf_config = (
            spec_config.draft_model_config.hf_config
            if spec_config is not None
            else None
        )
        # Detect index_share_for_mtp_iteration. When True, the proposer
        # toggles skip_topk so step 0 computes MTP's own indices and
        # steps 1+ reuse them.
        self.share_mtp_topk_indices = (
            self.vllm_config.parallel_config.prefill_context_parallel_size == 1
            and getattr(draft_hf_config, "index_share_for_mtp_iteration", False)
            and hasattr(draft_model.model, "set_skip_topk")
            and hasattr(draft_model.model, "compact_topk_indices")
        )
        return draft_model

    def _supports_step0_split(self) -> bool:
        return (
            self.dp_size == 1
            and self.dcp_size == 1
            and self.pcp_manager is None
            and hasattr(self.model, "precompute_and_store_draft_kv")
            and hasattr(self.model, "forward_step0_last_rows")
        )

    def capture(self):
        # Bake the split mode only into graphs whose shape it always serves.
        self._step0_split = False
        self._capturing_step0 = True
        try:
            manager = self.prefill_cudagraph_manager
            if manager is not None and self._supports_step0_split():
                shared = set()
                for mode in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE):
                    for desc in manager._capture_descs.get(mode, []):
                        if (
                            mode == CUDAGraphMode.FULL
                            and desc.uniform_token_count is not None
                        ):
                            continue
                        num_reqs = (
                            desc.num_reqs
                            if desc.num_reqs is not None
                            else min(desc.num_tokens, self.max_num_reqs)
                        )
                        shared.add((num_reqs, desc.num_tokens))
                for desc in manager._capture_descs.get(CUDAGraphMode.FULL, []):
                    if (
                        desc.uniform_token_count is None
                        or desc.num_reqs is None
                        or desc.num_ubatches != 1
                    ):
                        continue
                    key = (desc.num_reqs, desc.num_tokens)
                    if key in shared or key in self._step0_region2_states:
                        continue
                    self._step0_region2_states[key] = prepare_inputs_to_capture(
                        desc.num_reqs,
                        desc.num_reqs,
                        self.model_state,
                        self.input_buffers,
                        self.block_tables,
                        self.attn_groups,
                        self.kv_cache_config,
                        full_cudagraph=True,
                        max_query_len=1,
                    )
            super().capture()
        finally:
            self._capturing_step0 = False

    def _prepare_step0_prefill(
        self, batch_desc: BatchExecutionDescriptor, input_batch: InputBatch
    ) -> None:
        """Per-batch prep for the split step-0 replay: seq_lens/query_start_loc
        for the 1-query-per-request last-row attention, then rebuild the draft's
        decode attention metadata (region 2) so its persistent buffers hold this
        batch before the captured graph replays."""
        self._step0_split = False
        if (
            batch_desc.cg_mode != CUDAGraphMode.FULL
            or batch_desc.uniform_token_count is None
            or batch_desc.num_reqs is None
            or not self._supports_step0_split()
        ):
            return
        key = (batch_desc.num_reqs, batch_desc.num_tokens)
        if key not in self._step0_region2_states:
            return
        num_reqs = input_batch.num_reqs
        num_reqs_padded = batch_desc.num_reqs
        # The last-row attention window: position of the last valid row + 1
        # (padded requests attend nothing).
        self.input_buffers.seq_lens[:num_reqs_padded] = (
            self.input_buffers.positions.index_select(
                0, self.last_token_indices[:num_reqs_padded]
            )
            + 1
        ).to(self.input_buffers.seq_lens.dtype)
        self.input_buffers.seq_lens[num_reqs:num_reqs_padded].zero_()
        if self._step0_req_arange is None:
            self._step0_req_arange = torch.arange(
                self.max_num_reqs + 1,
                dtype=self.input_buffers.query_start_loc.dtype,
                device=self.device,
            )
        self.input_buffers.query_start_loc[: num_reqs_padded + 1] = (
            self._step0_req_arange[: num_reqs_padded + 1]
        )
        # Same builder buffers the captured region-2 kernels read.
        self._build_uniform_attn_metadata(
            num_reqs=num_reqs,
            batch_desc=BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=num_reqs_padded,
                num_reqs=num_reqs_padded,
                uniform_token_count=1,
            ),
            num_query_per_req=1,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            step=0,
        )
        self._step0_split = True

    def _prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs=None,
    ) -> None:
        if (
            self._capturing_step0
            and mm_inputs is None
            and (num_reqs, num_tokens) in self._step0_region2_states
        ):
            self._prefill_step0_split(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
            return
        super()._prefill(
            num_reqs,
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            mm_inputs,
        )

    def _prefill_step0_split(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode: CUDAGraphMode,
    ) -> None:
        """Split step-0: KV-write-only precompute over the padded batch under
        the target's attention metadata, then attention/MoE/indexer scoring on
        each request's last valid row only under the draft's decode metadata."""
        state = self._step0_region2_states[(num_reqs, num_tokens)]
        last_token_indices = self.last_token_indices[:num_reqs]
        positions = self.input_buffers.positions[last_token_indices]
        # The output hidden state at position P (= positions) and the token id
        # at P+1 are used to draft the token at P+2. Sampling keys a draw by the
        # position before the sampled token, so the net adjustment is +1.
        sample_src_positions = positions + 1
        idx_mapping = self.idx_mapping[:num_reqs]

        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
        ):
            self.model.precompute_and_store_draft_kv(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
            )

        with set_forward_context(
            state.attn_metadata,
            self.vllm_config,
            num_tokens=num_reqs,
            slot_mapping=state.slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_reqs),
        ):
            last_hidden_states = self.model.forward_step0_last_rows(
                last_token_indices, positions
            )

        self.draft_tokens[:num_reqs, 0] = self.sample_draft(
            last_hidden_states[:num_reqs],
            sample_src_positions,
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )
        self.hidden_states[:num_reqs] = last_hidden_states[:num_reqs]
        self.input_buffers.positions[:num_reqs] = positions
        self.sample_src_positions[:num_reqs] = sample_src_positions

    def on_prefill_begin(self, num_reqs: int) -> None:
        # Step 0 computes its own top-k. Unconditional, so a step that died
        # midway cannot leave reuse mode on.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)

    def on_prefill_end(self, num_reqs: int) -> None:
        # Step 0 (prefill) wrote topk indices for every query token in the
        # multi-token batch. Compact them down to each request's last token so
        # steps 1+ can reuse them from the shared buffer. In the split path the
        # last-row scoring already wrote those rows at the buffer front.
        if (
            self.share_mtp_topk_indices
            and self.num_speculative_steps > 1
            and not self._step0_split
        ):
            self.model.model.compact_topk_indices(self.last_token_indices[:num_reqs])

    def on_multi_step_decode_begin(self, num_reqs: int) -> None:
        # Switch to reuse mode so draft steps 1+ skip the indexer op and read
        # the indices that step 0 wrote into the shared buffer.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(True)

    def on_multi_step_decode_end(self, num_reqs: int) -> None:
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)
