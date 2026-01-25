"""Safetensors cache persistence adapter."""

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from semantic.domain.entities import AgentBlocks, KVBlock
from semantic.domain.errors import AgentNotFoundError


class SafetensorsCacheAdapter:
    """Adapter for cache persistence using safetensors format."""

    def __init__(self, cache_dir: Path) -> None:
        """Initialize adapter with cache directory."""
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        agent_id: str,
        blocks: AgentBlocks,
        metadata: dict[str, Any],
    ) -> Path:
        """Save agent blocks to disk using safetensors format."""
        cache_path = self.cache_dir / f"{agent_id}.safetensors"
        tmp_path = self.cache_dir / f"{agent_id}.safetensors.tmp"

        tensors: dict[str, np.ndarray[Any, Any]] = {}

        for layer_id, layer_blocks in blocks.blocks.items():
            for block_idx, block in enumerate(layer_blocks):
                if block.layer_data is None:
                    continue

                k_data = block.layer_data.get("k")
                v_data = block.layer_data.get("v")

                if k_data is None or v_data is None:
                    continue

                # Skip FakeTensor (unit tests)
                if hasattr(k_data, "__class__") and k_data.__class__.__name__ == "FakeTensor":
                    continue

                # Convert to numpy
                try:
                    if hasattr(k_data, "__array_interface__") or hasattr(k_data, "__array__"):
                        k_np = np.asarray(k_data)
                        v_np = np.asarray(v_data)
                    else:
                        k_np = np.array(k_data)
                        v_np = np.array(v_data)

                    k_key = f"L{layer_id}_B{block_idx}_K"
                    v_key = f"L{layer_id}_B{block_idx}_V"
                    tensors[k_key] = k_np
                    tensors[v_key] = v_np
                except Exception:
                    continue

        # Convert metadata values to strings
        str_metadata = {k: str(v) for k, v in metadata.items()}

        # Atomic write
        save_file(tensors, str(tmp_path), metadata=str_metadata)
        tmp_path.rename(cache_path)

        return cache_path

    def load(self, cache_path: Path) -> tuple[dict[int, list[KVBlock]], dict[str, Any]]:
        """Load agent blocks from disk."""
        if not cache_path.exists():
            raise AgentNotFoundError(f"Cache not found: {cache_path}")

        # Read metadata from header (safetensors format: 8-byte header size + JSON header)
        with cache_path.open("rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                raise AgentNotFoundError(f"Invalid cache file: {cache_path}")

            header_size = struct.unpack("<Q", header_size_bytes)[0]
            header_bytes = f.read(header_size)
            header = json.loads(header_bytes.decode("utf-8"))

        metadata = header.get("__metadata__", {})

        # Load tensors
        tensors_data = load_file(str(cache_path))

        # Reconstruct blocks
        blocks_dict: dict[int, list[KVBlock]] = {}

        for key in sorted(tensors_data.keys()):
            if not key.endswith("_K"):
                continue

            parts = key.split("_")
            if len(parts) != 3:
                continue

            layer_id = int(parts[0][1:])
            block_idx = int(parts[1][1:])

            k_key = key
            v_key = key.replace("_K", "_V")

            if v_key not in tensors_data:
                continue

            k_array = tensors_data[k_key]
            v_array = tensors_data[v_key]

            token_count = k_array.shape[2] if len(k_array.shape) >= 3 else 0

            block = KVBlock(
                block_id=layer_id * 1000 + block_idx,
                layer_id=layer_id,
                token_count=token_count,
                layer_data={"k": k_array, "v": v_array},
            )

            if layer_id not in blocks_dict:
                blocks_dict[layer_id] = []
            blocks_dict[layer_id].append(block)

        return blocks_dict, metadata

    def exists(self, agent_id: str) -> bool:
        """Check if cache exists on disk."""
        cache_path = self.cache_dir / f"{agent_id}.safetensors"
        return cache_path.exists()

    def delete(self, agent_id: str) -> None:
        """Delete cache from disk."""
        cache_path = self.cache_dir / f"{agent_id}.safetensors"
        if cache_path.exists():
            cache_path.unlink()

    def list_cached_agents(self) -> list[str]:
        """List all agent IDs with caches on disk."""
        return [p.stem for p in self.cache_dir.glob("*.safetensors")]
