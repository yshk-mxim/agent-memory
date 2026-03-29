# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT safetensors cache persistence adapter.

Numpy-based KV cache persistence for the TRT backend. Uses safetensors.numpy
for I/O — no MLX dependency. Implements the same CachePersistencePort as the
MLX SafetensorsCacheAdapter but with numpy arrays instead of mx.array.

KV cache format on disk: Q4 quantized (via CacheQuantizationPort) or raw FP16.
"""

import json
import logging
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from agent_memory.domain.entities import AgentBlocks, KVBlock
from agent_memory.domain.errors import AgentNotFoundError, CachePersistenceError
from agent_memory.ports.outbound import CacheQuantizationPort

logger = logging.getLogger(__name__)

_VALID_AGENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class TRTSafetensorsCacheAdapter:
    """Numpy-based cache persistence for TRT backend.

    Uses safetensors.numpy for I/O. No MLX dependency.
    """

    def __init__(
        self,
        cache_dir: Path,
        kv_bits: int = 4,
        kv_group_size: int = 64,
        quantizer: CacheQuantizationPort | None = None,
    ) -> None:
        """Initialize adapter.

        Args:
            cache_dir: Directory for safetensors files.
            kv_bits: Quantization bits for disk format.
            kv_group_size: Quantization group size.
            quantizer: Optional quantization port for Q4/Q8 compression.
        """
        self._kv_bits = kv_bits
        self._kv_group_size = kv_group_size
        self._quantizer = quantizer
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise CachePersistenceError(
                f"Failed to create cache directory {self.cache_dir}: {e}"
            ) from e

    def _validate_agent_id(self, agent_id: str) -> None:
        """Validate agent_id is safe for filesystem use."""
        max_len = 256
        if not agent_id or len(agent_id) > max_len:
            raise CachePersistenceError(f"Invalid agent_id length: {len(agent_id)}")
        if not _VALID_AGENT_ID_PATTERN.match(agent_id):
            raise CachePersistenceError(f"Invalid agent_id characters: {agent_id}")

    def save(
        self,
        agent_id: str,
        blocks: AgentBlocks,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save KV cache to disk as safetensors.

        Args:
            agent_id: Agent identifier (used as filename).
            blocks: AgentBlocks containing per-layer KV data.
            metadata: Optional metadata dict (stored in safetensors header).

        Returns:
            Path to the saved file.
        """
        self._validate_agent_id(agent_id)
        cache_path = self.cache_dir / f"{agent_id}.safetensors"
        tmp_path = self.cache_dir / f"{agent_id}.tmp.safetensors"

        tensors: dict[str, np.ndarray] = {}

        for layer_id in sorted(blocks.blocks.keys()):
            for block_idx, block in enumerate(blocks.blocks[layer_id]):
                if block.layer_data is None:
                    continue

                k_data, v_data = block.layer_data

                quantized_tuple_len = 3  # (weights, scales, biases)
                if isinstance(k_data, tuple) and len(k_data) == quantized_tuple_len:
                    # Already quantized: (weights, scales, biases)
                    kw, ks, kb = k_data
                    vw, vs, vb = v_data
                    tensors[f"L{layer_id}_B{block_idx}_K_weights"] = np.asarray(kw)
                    tensors[f"L{layer_id}_B{block_idx}_K_scales"] = np.asarray(ks)
                    tensors[f"L{layer_id}_B{block_idx}_K_biases"] = np.asarray(kb)
                    tensors[f"L{layer_id}_B{block_idx}_V_weights"] = np.asarray(vw)
                    tensors[f"L{layer_id}_B{block_idx}_V_scales"] = np.asarray(vs)
                    tensors[f"L{layer_id}_B{block_idx}_V_biases"] = np.asarray(vb)
                else:
                    # FP16 arrays — quantize if quantizer provided
                    k_arr = np.asarray(k_data, dtype=np.float16)
                    v_arr = np.asarray(v_data, dtype=np.float16)

                    if self._quantizer is not None:
                        kw, ks, kb = self._quantizer.quantize(
                            k_arr.reshape(-1),
                            bits=self._kv_bits,
                            group_size=self._kv_group_size,
                        )
                        vw, vs, vb = self._quantizer.quantize(
                            v_arr.reshape(-1),
                            bits=self._kv_bits,
                            group_size=self._kv_group_size,
                        )
                        tensors[f"L{layer_id}_B{block_idx}_K_weights"] = kw
                        tensors[f"L{layer_id}_B{block_idx}_K_scales"] = ks
                        tensors[f"L{layer_id}_B{block_idx}_K_biases"] = kb
                        tensors[f"L{layer_id}_B{block_idx}_V_weights"] = vw
                        tensors[f"L{layer_id}_B{block_idx}_V_scales"] = vs
                        tensors[f"L{layer_id}_B{block_idx}_V_biases"] = vb
                    else:
                        tensors[f"L{layer_id}_B{block_idx}_K"] = k_arr
                        tensors[f"L{layer_id}_B{block_idx}_V"] = v_arr

        # Encode metadata into safetensors header
        meta_dict = {}
        if metadata:
            meta_dict = {k: str(v) for k, v in metadata.items()}

        save_file(tensors, str(tmp_path), metadata=meta_dict)
        tmp_path.rename(cache_path)  # Atomic rename

        return cache_path

    def load(self, agent_id: str) -> tuple[AgentBlocks, dict[str, Any]]:
        """Load KV cache from safetensors file.

        Args:
            agent_id: Agent identifier.

        Returns:
            Tuple of (AgentBlocks, metadata dict).
        """
        self._validate_agent_id(agent_id)
        cache_path = self.cache_dir / f"{agent_id}.safetensors"

        if not cache_path.exists():
            raise AgentNotFoundError(f"No cache for agent {agent_id}")

        # Read metadata from safetensors header
        metadata = self._read_metadata(cache_path)

        # Load tensors
        tensors = load_file(str(cache_path))

        # Reconstruct blocks
        blocks_dict: dict[int, list[KVBlock]] = {}
        layer_idx = 0
        while True:
            k_key = f"L{layer_idx}_B0_K"
            kw_key = f"L{layer_idx}_B0_K_weights"

            if k_key in tensors:
                # Raw FP16 format
                k = tensors[k_key]
                v = tensors[f"L{layer_idx}_B0_V"]
                block = KVBlock(
                    block_id=layer_idx * 1_000_000,
                    layer_id=layer_idx,
                    token_count=int(metadata.get("total_tokens", 0)),
                    layer_data=(k, v),
                )
                blocks_dict[layer_idx] = [block]
            elif kw_key in tensors:
                # Quantized format
                kw = tensors[kw_key]
                ks = tensors[f"L{layer_idx}_B0_K_scales"]
                kb = tensors[f"L{layer_idx}_B0_K_biases"]
                vw = tensors[f"L{layer_idx}_B0_V_weights"]
                vs = tensors[f"L{layer_idx}_B0_V_scales"]
                vb = tensors[f"L{layer_idx}_B0_V_biases"]
                block = KVBlock(
                    block_id=layer_idx * 1_000_000,
                    layer_id=layer_idx,
                    token_count=int(metadata.get("total_tokens", 0)),
                    layer_data=((kw, ks, kb), (vw, vs, vb)),
                )
                blocks_dict[layer_idx] = [block]
            else:
                break

            layer_idx += 1

        total_tokens = int(metadata.get("total_tokens", 0))
        agent_blocks = AgentBlocks(
            agent_id=agent_id,
            blocks=blocks_dict,
            total_tokens=total_tokens,
        )

        return agent_blocks, metadata

    def exists(self, agent_id: str) -> bool:
        """Check if cache file exists on disk."""
        return (self.cache_dir / f"{agent_id}.safetensors").exists()

    def delete(self, agent_id: str) -> None:
        """Delete cache file from disk."""
        path = self.cache_dir / f"{agent_id}.safetensors"
        if path.exists():
            path.unlink()

    def list_cached_agents(self) -> list[str]:
        """List all agent IDs with caches on disk."""
        return [f.stem for f in self.cache_dir.glob("*.safetensors")]

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, str]:
        """Read metadata from safetensors header without loading tensors."""
        with path.open("rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_size)
        header = json.loads(header_bytes)
        result: dict[str, str] = header.get("__metadata__", {})
        return result
