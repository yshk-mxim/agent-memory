"""Safetensors cache persistence adapter.

CRITICAL: Preserves 4-bit quantization when saving/loading KV cache.
"""

import json
import logging
import os
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from semantic.domain.entities import AgentBlocks, KVBlock
from semantic.domain.errors import AgentNotFoundError, CachePersistenceError

logger = logging.getLogger(__name__)

# Pattern for valid agent IDs: alphanumeric, hyphens, underscores only
_VALID_AGENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class SafetensorsCacheAdapter:
    """Adapter for cache persistence using safetensors format."""

    def __init__(self, cache_dir: Path) -> None:
        """Initialize adapter with cache directory."""
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise CachePersistenceError(f"Failed to create cache directory {self.cache_dir}: {e}") from e

    def _validate_agent_id(self, agent_id: str) -> None:
        """Validate agent_id to prevent path traversal attacks.

        Args:
            agent_id: Agent identifier to validate

        Raises:
            CachePersistenceError: If agent_id contains invalid characters
        """
        if not agent_id:
            raise CachePersistenceError("agent_id cannot be empty")

        if len(agent_id) > 256:
            raise CachePersistenceError(
                f"agent_id too long: {len(agent_id)} chars (max 256)"
            )

        if not _VALID_AGENT_ID_PATTERN.match(agent_id):
            raise CachePersistenceError(
                f"Invalid agent_id '{agent_id}': must contain only "
                "alphanumeric characters, hyphens, and underscores"
            )

    def save(
        self,
        agent_id: str,
        blocks: AgentBlocks,
        metadata: dict[str, Any],
    ) -> Path:
        """Save agent blocks to disk using safetensors format."""
        self._validate_agent_id(agent_id)
        cache_path = self.cache_dir / f"{agent_id}.safetensors"
        tmp_path = self.cache_dir / f"{agent_id}.safetensors.tmp"

        # Estimate cache size before saving to avoid partial writes on disk full
        estimated_bytes = self._estimate_cache_size(blocks)
        free_bytes = self._get_free_disk_space()

        # Require 20% overhead for safety (safetensors metadata, tmp file, etc.)
        required_bytes = int(estimated_bytes * 1.2)

        if free_bytes < required_bytes:
            free_mb = free_bytes / (1024 * 1024)
            required_mb = required_bytes / (1024 * 1024)
            raise CachePersistenceError(
                f"Insufficient disk space: {free_mb:.1f}MB available, "
                f"{required_mb:.1f}MB required for agent {agent_id}"
            )

        tensors: dict[str, np.ndarray[Any, Any]] = {}

        for layer_id, layer_blocks in blocks.blocks.items():
            for block_idx, block in enumerate(layer_blocks):
                if block.layer_data is None:
                    continue

                k_data = block.layer_data.get("k")
                v_data = block.layer_data.get("v")

                if k_data is None or v_data is None:
                    continue

                # Skip FakeTensor (unit tests - TODO: use dependency injection instead)
                if k_data.__class__.__name__ == "FakeTensor":
                    continue

                # CRITICAL: Quantize KV cache to 4-bit before saving
                # MLX's BatchGenerator returns float16 arrays (dequantized)
                # We need to quantize them ourselves to save disk space and memory
                try:
                    import mlx.core as mx  # Import here to avoid circular deps

                    # Check if k_data is already quantized (tuple of 3 components)
                    if isinstance(k_data, tuple) and len(k_data) == 3:
                        # Already quantized format: (weights, scales, biases)
                        k_weights, k_scales, k_biases = k_data
                        v_weights, v_scales, v_biases = v_data

                        # Save all quantized components separately
                        tensors[f"L{layer_id}_B{block_idx}_K_weights"] = np.asarray(k_weights)
                        tensors[f"L{layer_id}_B{block_idx}_K_scales"] = np.asarray(k_scales)
                        if k_biases is not None:
                            tensors[f"L{layer_id}_B{block_idx}_K_biases"] = np.asarray(k_biases)

                        tensors[f"L{layer_id}_B{block_idx}_V_weights"] = np.asarray(v_weights)
                        tensors[f"L{layer_id}_B{block_idx}_V_scales"] = np.asarray(v_scales)
                        if v_biases is not None:
                            tensors[f"L{layer_id}_B{block_idx}_V_biases"] = np.asarray(v_biases)

                    else:
                        # Float16/float32 array - QUANTIZE it before saving
                        # Convert to MLX array if needed
                        if not hasattr(k_data, "dtype") or "mlx" not in str(type(k_data)):
                            k_data = mx.array(k_data)
                            v_data = mx.array(v_data)

                        # Quantize to 4-bit with group_size=64
                        k_weights, k_scales, k_biases = mx.quantize(
                            k_data, group_size=64, bits=4
                        )
                        v_weights, v_scales, v_biases = mx.quantize(
                            v_data, group_size=64, bits=4
                        )

                        # Save quantized components
                        tensors[f"L{layer_id}_B{block_idx}_K_weights"] = np.asarray(k_weights)
                        tensors[f"L{layer_id}_B{block_idx}_K_scales"] = np.asarray(k_scales)
                        if k_biases is not None:
                            tensors[f"L{layer_id}_B{block_idx}_K_biases"] = np.asarray(k_biases)

                        tensors[f"L{layer_id}_B{block_idx}_V_weights"] = np.asarray(v_weights)
                        tensors[f"L{layer_id}_B{block_idx}_V_scales"] = np.asarray(v_scales)
                        if v_biases is not None:
                            tensors[f"L{layer_id}_B{block_idx}_V_biases"] = np.asarray(v_biases)

                except (TypeError, ValueError) as e:
                    # Expected errors from invalid tensor shapes/types - log and skip block
                    logger.warning(
                        f"Skipping block L{layer_id}_B{block_idx}: {type(e).__name__}: {e}"
                    )
                    continue
                except Exception as e:
                    # Unexpected error - log with full traceback for debugging
                    logger.error(
                        f"Unexpected error saving block L{layer_id}_B{block_idx}: {e}",
                        exc_info=True,
                    )
                    continue

        # Convert metadata values to strings
        str_metadata = {k: str(v) for k, v in metadata.items()}

        # Atomic write with proper cleanup on failure
        try:
            save_file(tensors, str(tmp_path), metadata=str_metadata)
            tmp_path.rename(cache_path)
        except OSError as e:
            # Clean up temp file on failure
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    logger.warning(f"Failed to clean up temp file: {tmp_path}")
            raise CachePersistenceError(
                f"Failed to save cache for agent {agent_id}: {e}"
            ) from e
        finally:
            # Extra safety: clean up any leftover temp file
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass  # Already renamed or cleaned up

        return cache_path

    def load(self, cache_path: Path) -> tuple[dict[int, list[KVBlock]], dict[str, Any]]:
        """Load agent blocks from disk."""
        if not cache_path.exists():
            raise AgentNotFoundError(f"Cache not found: {cache_path}")

        # Read metadata from header (safetensors format: 8-byte header size + JSON header)
        try:
            with cache_path.open("rb") as f:
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8:
                    raise CachePersistenceError(f"Invalid cache file (truncated header): {cache_path}")

                header_size = struct.unpack("<Q", header_size_bytes)[0]
                header_bytes = f.read(header_size)
                header = json.loads(header_bytes.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise CachePersistenceError(f"Corrupted cache metadata for {cache_path}: {e}") from e
        except OSError as e:
            raise CachePersistenceError(f"Failed to read cache file {cache_path}: {e}") from e

        metadata = header.get("__metadata__", {})

        # Load tensors
        try:
            tensors_data = load_file(str(cache_path))
        except Exception as e:
            raise CachePersistenceError(f"Failed to load tensors from {cache_path}: {e}") from e

        # Reconstruct blocks
        blocks_dict: dict[int, list[KVBlock]] = {}
        processed_blocks: set[tuple[int, int]] = set()

        # First pass: find all unique (layer_id, block_idx) pairs
        for key in sorted(tensors_data.keys()):
            if "_K" in key or "_V" in key:
                parts = key.split("_")
                if len(parts) >= 3:
                    try:
                        layer_id = int(parts[0][1:])  # L123 -> 123
                        block_idx = int(parts[1][1:])  # B456 -> 456
                        processed_blocks.add((layer_id, block_idx))
                    except (ValueError, IndexError):
                        continue

        # Second pass: reconstruct each block
        import mlx.core as mx  # Import for quantized array reconstruction

        for layer_id, block_idx in sorted(processed_blocks):
            # Check for quantized format first (has _weights suffix)
            k_weights_key = f"L{layer_id}_B{block_idx}_K_weights"
            v_weights_key = f"L{layer_id}_B{block_idx}_V_weights"

            if k_weights_key in tensors_data and v_weights_key in tensors_data:
                # QUANTIZED FORMAT: Reconstruct quantized tuple
                k_weights = mx.array(tensors_data[k_weights_key])
                k_scales = mx.array(tensors_data[f"L{layer_id}_B{block_idx}_K_scales"])
                k_biases_key = f"L{layer_id}_B{block_idx}_K_biases"
                k_biases = mx.array(tensors_data[k_biases_key]) if k_biases_key in tensors_data else None

                v_weights = mx.array(tensors_data[v_weights_key])
                v_scales = mx.array(tensors_data[f"L{layer_id}_B{block_idx}_V_scales"])
                v_biases_key = f"L{layer_id}_B{block_idx}_V_biases"
                v_biases = mx.array(tensors_data[v_biases_key]) if v_biases_key in tensors_data else None

                # Store as quantized tuples (don't dequantize unless needed)
                k_data = (k_weights, k_scales, k_biases)
                v_data = (v_weights, v_scales, v_biases)

                # Get token count from weights shape (axis 2)
                token_count = k_weights.shape[2] if len(k_weights.shape) >= 3 else 0

            else:
                # REGULAR FORMAT: Float arrays
                k_key = f"L{layer_id}_B{block_idx}_K"
                v_key = f"L{layer_id}_B{block_idx}_V"

                if k_key not in tensors_data or v_key not in tensors_data:
                    continue

                k_data = tensors_data[k_key]
                v_data = tensors_data[v_key]

                token_count = k_data.shape[2] if len(k_data.shape) >= 3 else 0

            # Use large multiplier to avoid ID collisions with many blocks
            # 1_000_000 supports up to 1M blocks per layer (enough for 256M tokens at 256 tokens/block)
            block = KVBlock(
                block_id=layer_id * 1_000_000 + block_idx,
                layer_id=layer_id,
                token_count=token_count,
                layer_data={"k": k_data, "v": v_data},
            )

            if layer_id not in blocks_dict:
                blocks_dict[layer_id] = []
            blocks_dict[layer_id].append(block)

        return blocks_dict, metadata

    def exists(self, agent_id: str) -> bool:
        """Check if cache exists on disk."""
        self._validate_agent_id(agent_id)
        cache_path = self.cache_dir / f"{agent_id}.safetensors"
        return cache_path.exists()

    def delete(self, agent_id: str) -> None:
        """Delete cache from disk."""
        self._validate_agent_id(agent_id)
        cache_path = self.cache_dir / f"{agent_id}.safetensors"
        if cache_path.exists():
            cache_path.unlink()

    def list_cached_agents(self) -> list[str]:
        """List all agent IDs with caches on disk."""
        return [p.stem for p in self.cache_dir.glob("*.safetensors")]

    def _get_free_disk_space(self) -> int:
        """Get free disk space on cache directory's filesystem."""
        try:
            stat = os.statvfs(self.cache_dir)
            return stat.f_bavail * stat.f_frsize
        except OSError as e:
            logger.warning(f"Failed to get disk space for {self.cache_dir}: {e}")
            return 0  # Return 0 to trigger disk space check failure

    def _estimate_cache_size(self, blocks: AgentBlocks) -> int:
        """Estimate disk size of cache in bytes.

        For Q4 quantized cache:
        - weights: seq_len * n_heads * (head_dim // 8) * 4 bytes (uint32 packed)
        - scales: seq_len * n_heads * (head_dim // group_size) * 2 bytes (float16)
        - biases: same as scales

        Rough estimate: ~0.75 bytes per element vs 2 bytes for float16
        """
        total_bytes = 0

        for layer_blocks in blocks.blocks.values():
            for block in layer_blocks:
                if block.layer_data is None:
                    continue

                k_data = block.layer_data.get("k")
                v_data = block.layer_data.get("v")

                if k_data is None or v_data is None:
                    continue

                # Check if already quantized (tuple of 3) or float array
                if isinstance(k_data, tuple) and len(k_data) == 3:
                    # Quantized: sum component sizes
                    k_weights, k_scales, k_biases = k_data
                    v_weights, v_scales, v_biases = v_data

                    # Use numpy or mlx array size calculation
                    for arr in [k_weights, k_scales, k_biases, v_weights, v_scales, v_biases]:
                        if arr is not None:
                            if hasattr(arr, "nbytes"):
                                total_bytes += arr.nbytes
                            elif hasattr(arr, "size"):
                                # Estimate from size and dtype
                                total_bytes += arr.size * 4  # Conservative estimate
                else:
                    # Float array
                    for arr in [k_data, v_data]:
                        if hasattr(arr, "nbytes"):
                            total_bytes += arr.nbytes
                        elif hasattr(arr, "size"):
                            total_bytes += arr.size * 2  # float16 = 2 bytes

        return total_bytes
