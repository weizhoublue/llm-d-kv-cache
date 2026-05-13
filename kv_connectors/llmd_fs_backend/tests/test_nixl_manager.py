# Copyright 2025 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import struct
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import OffloadKey, make_offload_key

from llmd_fs_backend.file_mapper import FileMapper
from llmd_nixl.manager import (
    LOOKUP_MODE_DICT,
    NixlStorageOffloadingManager,
)


def get_prefix_hash(token_ids: Iterable[int]) -> BlockHash:
    buf = bytearray()
    for token_id in token_ids:
        buf += struct.pack("<I", int(token_id) & 0xFFFFFFFF)
    digest_int = int.from_bytes(hashlib.sha256(buf).digest()[:8], "big")
    return BlockHash((digest_int & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))


def get_offload_key(token_ids: Iterable[int], group_idx: int = 0) -> OffloadKey:
    return make_offload_key(get_prefix_hash(token_ids), group_idx)


def test_nixl_manager_dict_lookup_uses_single_offload_key(tmp_path):
    file_mapper = FileMapper(
        root_dir=str(tmp_path),
        model_name="test-model",
        gpu_block_size=16,
        gpu_blocks_per_file=1,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        rank=0,
        dtype="torch.float16",
    )
    manager = NixlStorageOffloadingManager(
        file_mapper=file_mapper,
        extra_config={"lookup_mode": LOOKUP_MODE_DICT},
    )
    key = get_offload_key(range(100, 117), group_idx=1)

    assert manager.lookup(key, req_context=None) is False

    manager.complete_store([key])

    assert manager.lookup(key, req_context=None) is True
