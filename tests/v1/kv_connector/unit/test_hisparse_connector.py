# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import (
    HiSparseConnector,
    HiSparseConnectorMetadata,
    HiSparseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import (
    HiSparseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import (
    MooncakeStoreConnector,
    _mooncake_cache_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
from vllm.v1.hisparse.runtime import HiSparseCacheHandle
from vllm.v1.kv_cache_interface import HiSparseResidentSpec, KVCacheGroupRole
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


@pytest.mark.parametrize("nested", [False, True])
def test_cache_manager_binding_preserves_hisparse_and_legacy_pool_hooks(nested):
    """Composite binding must reach HiSparse and existing pool-only connectors."""
    hisparse = object.__new__(HiSparseConnector)
    hisparse._role = KVConnectorRole.SCHEDULER
    hisparse.connector_scheduler = HiSparseConnectorScheduler(async_speculative=False)
    legacy = object.__new__(ExampleConnector)
    legacy.bind_gpu_block_pool = MagicMock()
    connector = object.__new__(MultiConnector)
    connector._connectors = [hisparse, legacy]
    if nested:
        parent = object.__new__(MultiConnector)
        parent._connectors = [connector]
        connector = parent
    from tests.v1.core.test_prefix_caching import make_hisparse_kv_cache_manager

    manager = make_hisparse_kv_cache_manager(16, 16)

    connector.bind_kv_cache_manager(manager)

    assert hisparse.connector_scheduler.coordinator is get_hisparse_coordinator(manager)
    legacy.bind_gpu_block_pool.assert_called_once_with(manager.block_pool)


def test_hisparse_requires_block_outermost_device_layout():
    assert HiSparseConnector.get_required_kvcache_layout(MagicMock()) == "BLHNC"


def test_mooncake_stores_hisparse_host_source_while_nixl_uses_resident_pages():
    from tests.v1.core.test_prefix_caching import make_hisparse_kv_cache_manager

    config = make_hisparse_kv_cache_manager(
        16, 16, transfer_device_cache=True
    ).kv_cache_config
    mooncake_config = _mooncake_cache_config(config)

    assert any(
        isinstance(group.kv_cache_spec, HiSparseResidentSpec)
        for group in config.transfer_groups
    )
    assert not any(
        group.role is KVCacheGroupRole.HISPARSE_SOURCE
        for group in config.transfer_groups
    )
    assert any(
        group.role is KVCacheGroupRole.HISPARSE_SOURCE
        for group in mooncake_config.prefix_cacheable_groups
    )
    assert not any(
        isinstance(group.kv_cache_spec, HiSparseResidentSpec)
        for group in mooncake_config.transfer_groups
    )


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def test_mooncake_connector_uses_hisparse_store_projection(role):
    from tests.v1.core.test_prefix_caching import make_hisparse_kv_cache_manager

    config = make_hisparse_kv_cache_manager(
        16, 16, transfer_device_cache=True
    ).kv_cache_config
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both", kv_connector_extra_config={}
        )
    )
    module = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector"
    with (
        patch.object(MooncakeStoreConnector, "_validate_kv_cache_config"),
        patch(f"{module}.MooncakeStoreScheduler") as scheduler,
        patch(f"{module}.MooncakeStoreWorker") as worker,
    ):
        connector = MooncakeStoreConnector(vllm_config, role, config)

    child = scheduler if role == KVConnectorRole.SCHEDULER else worker
    projected = child.call_args.args[1]
    assert connector._kv_cache_config is projected
    assert any(
        group.role is KVCacheGroupRole.HISPARSE_SOURCE
        for group in projected.prefix_cacheable_groups
    )
    assert all(
        not isinstance(group.kv_cache_spec, HiSparseResidentSpec)
        for group in projected.transfer_groups
    )


def test_mooncake_only_hit_keeps_hisparse_host_landing():
    from tests.v1.core.test_prefix_caching import make_hisparse_kv_cache_manager
    from tests.v1.kv_connector.unit.utils import create_request, make_nixl_scheduler

    coordinator = get_hisparse_coordinator(make_hisparse_kv_cache_manager(16, 16))
    nixl = make_nixl_scheduler()
    nixl.hisparse = coordinator
    mooncake = MagicMock()
    mooncake.get_num_new_matched_tokens.return_value = (16, True)
    hisparse = MagicMock()
    hisparse.get_num_new_matched_tokens.return_value = (0, False)
    composite = object.__new__(MultiConnector)
    composite._connectors = [nixl, mooncake, hisparse]
    composite._requests_to_connector = {}
    request = create_request(num_tokens=32)

    assert composite.get_num_new_matched_tokens(request, 0) == (16, True)
    assert composite._requests_to_connector[request.request_id] == 1
    assert coordinator.imports_to_host(request.request_id)


def test_no_forward_enqueues_deferred_hisparse_transfers():
    """A zero-token step must still enqueue deferred post-forward transfers."""
    connector = object.__new__(ActiveKVConnector)
    connector._disabled = False
    connector.pre_forward = MagicMock()
    connector.finish_forward = MagicMock()
    connector.post_forward = MagicMock(return_value=None)

    scheduler_output = SimpleNamespace(finished_req_ids=set())
    connector.no_forward(scheduler_output)

    connector.pre_forward.assert_called_once_with(scheduler_output)
    connector.finish_forward.assert_called_once_with()


def test_full_graph_step_prepares_host_mirror_outside_model():
    """Graph replay must restore host-mirror state cleared at step start."""
    runtime = SimpleNamespace(
        is_group_leader=True,
        eager_host_mirror=True,
        begin_forward=MagicMock(),
        invalidate_written_slots=MagicMock(),
    )
    handle = HiSparseCacheHandle(runtime)
    handle.mirror_slot_mapping = torch.tensor([4, 5])
    worker = object.__new__(HiSparseConnectorWorker)
    worker.cache_layer_names = ["layer"]
    worker.cache_handles = [handle]
    worker._group_leaders = (("layer", handle),)
    worker._per_layer_mirrored = set()
    worker._submitted_mirror_layers = set()
    worker.is_host_writer = True
    worker._enqueue_row_dma = MagicMock()
    worker.start_step = MagicMock(
        side_effect=lambda *_args, **_kwargs: worker._clear_forward_mirror_state()
    )

    connector = object.__new__(HiSparseConnector)
    connector.connector_worker = worker
    connector._get_connector_metadata = MagicMock(
        return_value=HiSparseConnectorMetadata(None, (), (), {}, True)
    )
    req_id_per_token = torch.tensor([0, 1], dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        num_actual_tokens=2,
        num_decode_tokens=2,
        num_reqs=2,
        max_query_len=1,
        req_id_per_token=req_id_per_token,
    )

    connector.start_load_kv(
        SimpleNamespace(),
        request_state_indices=None,
        request_ids=[],
        attn_metadata={"layer": attn_metadata},
    )

    worker._enqueue_host_mirror()

    worker._enqueue_row_dma.assert_called_once_with((0,), ready_event=None)
    runtime.invalidate_written_slots.assert_called_once()
