#!/usr/bin/env python3
"""Benchmark: safetensors vs zstd-compressed safetensors for KV cache persistence.

Generates synthetic Q4 KV cache data matching Gemma 3 12B geometry at 32K tokens,
then compares plain safetensors (current approach) vs zstd-compressed safetensors.

Metrics: file size, save time, load time, peak memory delta.
"""

import gc
import json
import os
import struct
import time
import tracemalloc
from pathlib import Path

import mlx.core as mx
import zstandard as zstd

# ── Gemma 3 12B Q4 geometry ──────────────────────────────────────────────
N_LAYERS = 48
N_KV_HEADS = 8
HEAD_DIM = 256
BLOCK_TOKENS = 256
KV_BITS = 4
KV_GROUP_SIZE = 64

TARGET_TOKENS = 32_768
BLOCKS_PER_LAYER = TARGET_TOKENS // BLOCK_TOKENS  # 128

# Derived Q4 packing dimensions
# uint32 packs 8 x 4-bit values, so last dim = HEAD_DIM // 8
PACKED_DIM = HEAD_DIM // 8  # 32
N_GROUPS = (N_KV_HEADS * BLOCK_TOKENS * HEAD_DIM) // KV_GROUP_SIZE

# Zstd compression levels to test
ZSTD_LEVELS = [1, 3, 6, 9]

CACHE_DIR = Path("/tmp/claude/benchmark_zstd")


def generate_synthetic_q4_block():
    """Generate one block's worth of Q4 KV data matching Gemma 3 geometry.

    Returns dict of tensor_key -> mx.array for one block.
    """
    # K and V each have: weights (uint32), scales (bfloat16), biases (bfloat16)
    # Shape matches what mx.quantize produces for (N_KV_HEADS, BLOCK_TOKENS, HEAD_DIM)
    k_weights = mx.random.randint(0, 2**31, shape=(N_KV_HEADS, BLOCK_TOKENS, PACKED_DIM)).astype(mx.uint32)
    k_scales = mx.random.normal(shape=(N_KV_HEADS, BLOCK_TOKENS, HEAD_DIM // KV_GROUP_SIZE)).astype(mx.bfloat16)
    k_biases = mx.random.normal(shape=(N_KV_HEADS, BLOCK_TOKENS, HEAD_DIM // KV_GROUP_SIZE)).astype(mx.bfloat16)

    v_weights = mx.random.randint(0, 2**31, shape=(N_KV_HEADS, BLOCK_TOKENS, PACKED_DIM)).astype(mx.uint32)
    v_scales = mx.random.normal(shape=(N_KV_HEADS, BLOCK_TOKENS, HEAD_DIM // KV_GROUP_SIZE)).astype(mx.bfloat16)
    v_biases = mx.random.normal(shape=(N_KV_HEADS, BLOCK_TOKENS, HEAD_DIM // KV_GROUP_SIZE)).astype(mx.bfloat16)

    return k_weights, k_scales, k_biases, v_weights, v_scales, v_biases


def build_full_cache_tensors():
    """Build all tensors for 32K tokens across 48 layers (128 blocks/layer).

    Returns dict of tensor_key -> mx.array, matching our safetensors naming convention.
    """
    tensors = {}
    for layer_id in range(N_LAYERS):
        for block_idx in range(BLOCKS_PER_LAYER):
            kw, ks, kb, vw, vs, vb = generate_synthetic_q4_block()
            prefix = f"L{layer_id}_B{block_idx}"
            tensors[f"{prefix}_K_weights"] = kw
            tensors[f"{prefix}_K_scales"] = ks
            tensors[f"{prefix}_K_biases"] = kb
            tensors[f"{prefix}_V_weights"] = vw
            tensors[f"{prefix}_V_scales"] = vs
            tensors[f"{prefix}_V_biases"] = vb
    # Force materialization
    mx.eval(*tensors.values())
    return tensors


def metadata_for_benchmark():
    """Create realistic metadata dict."""
    return {
        "model_id": "mlx-community/gemma-3-12b-it-4bit",
        "total_tokens": str(TARGET_TOKENS),
        "n_layers": str(N_LAYERS),
        "n_kv_heads": str(N_KV_HEADS),
        "head_dim": str(HEAD_DIM),
        "kv_bits": str(KV_BITS),
        "kv_group_size": str(KV_GROUP_SIZE),
    }


# ── Safetensors (plain) ──────────────────────────────────────────────────

def save_plain(tensors: dict, path: Path, metadata: dict) -> float:
    """Save via mx.save_safetensors. Returns wall-clock seconds."""
    stem = path.with_suffix("")  # mx.save_safetensors auto-appends .safetensors
    t0 = time.perf_counter()
    mx.save_safetensors(str(stem), tensors, metadata=metadata)
    dt = time.perf_counter() - t0
    return dt


def load_plain(path: Path) -> tuple[dict, float]:
    """Load via mx.load. Returns (tensors, wall-clock seconds)."""
    t0 = time.perf_counter()
    data = mx.load(str(path))
    dt = time.perf_counter() - t0
    return data, dt


# ── Zstd-compressed safetensors ──────────────────────────────────────────

def save_zstd(tensors: dict, path: Path, metadata: dict, level: int = 3) -> float:
    """Save safetensors then compress with zstd. Returns wall-clock seconds."""
    # Step 1: save to temp safetensors
    tmp_stem = path.with_suffix(".plain_tmp")
    tmp_safetensors = Path(str(tmp_stem) + ".safetensors")

    t0 = time.perf_counter()
    mx.save_safetensors(str(tmp_stem), tensors, metadata=metadata)

    # Step 2: read, compress, write
    with open(tmp_safetensors, "rb") as f:
        raw = f.read()

    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(raw)

    with open(path, "wb") as f:
        f.write(compressed)

    dt = time.perf_counter() - t0

    # Cleanup temp
    tmp_safetensors.unlink(missing_ok=True)
    return dt


def load_zstd(path: Path) -> tuple[dict, float]:
    """Decompress zstd then load safetensors. Returns (tensors, wall-clock seconds)."""
    tmp_path = path.with_suffix(".load_tmp.safetensors")

    t0 = time.perf_counter()

    with open(path, "rb") as f:
        compressed = f.read()

    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(compressed)

    with open(tmp_path, "wb") as f:
        f.write(raw)

    data = mx.load(str(tmp_path))
    dt = time.perf_counter() - t0

    tmp_path.unlink(missing_ok=True)
    return data, dt


def save_zstd_streaming(tensors: dict, path: Path, metadata: dict, level: int = 3) -> float:
    """Save safetensors then streaming-compress with zstd (lower peak memory)."""
    tmp_stem = path.with_suffix(".plain_tmp")
    tmp_safetensors = Path(str(tmp_stem) + ".safetensors")

    t0 = time.perf_counter()
    mx.save_safetensors(str(tmp_stem), tensors, metadata=metadata)

    cctx = zstd.ZstdCompressor(level=level)
    with open(tmp_safetensors, "rb") as ifh, open(path, "wb") as ofh:
        cctx.copy_stream(ifh, ofh)

    dt = time.perf_counter() - t0
    tmp_safetensors.unlink(missing_ok=True)
    return dt


def load_zstd_streaming(path: Path) -> tuple[dict, float]:
    """Streaming-decompress zstd then load. Lower peak memory than bulk."""
    tmp_path = path.with_suffix(".load_tmp.safetensors")

    t0 = time.perf_counter()
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as ifh, open(tmp_path, "wb") as ofh:
        dctx.copy_stream(ifh, ofh)

    data = mx.load(str(tmp_path))
    dt = time.perf_counter() - t0

    tmp_path.unlink(missing_ok=True)
    return data, dt


def measure_peak_memory(fn, *args, **kwargs):
    """Run fn(*args) while tracking peak memory. Returns (result, peak_bytes)."""
    gc.collect()
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    result = fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak - baseline


def format_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.2f} KB"
    return f"{n} B"


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("KV Cache Persistence Benchmark: safetensors vs zstd")
    print("=" * 72)
    print(f"Model geometry: Gemma 3 12B Q4")
    print(f"  Layers: {N_LAYERS}, KV heads: {N_KV_HEADS}, head_dim: {HEAD_DIM}")
    print(f"  Block tokens: {BLOCK_TOKENS}, Q4 group_size: {KV_GROUP_SIZE}")
    print(f"  Target: {TARGET_TOKENS:,} tokens = {BLOCKS_PER_LAYER} blocks/layer")
    print()

    # ── Generate synthetic data ────────────────────────────────────────
    print("Generating synthetic Q4 KV cache data...")
    t0 = time.perf_counter()
    tensors = build_full_cache_tensors()
    gen_time = time.perf_counter() - t0

    n_tensors = len(tensors)
    total_bytes = sum(t.nbytes for t in tensors.values())
    print(f"  {n_tensors} tensors, {format_bytes(total_bytes)} in-memory")
    print(f"  Generation time: {gen_time:.1f}s")
    print()

    metadata = metadata_for_benchmark()
    results = {}

    # ── Plain safetensors ──────────────────────────────────────────────
    print("─" * 72)
    print("PLAIN SAFETENSORS (current approach)")
    print("─" * 72)

    plain_path = CACHE_DIR / "cache_plain.safetensors"

    # Save
    save_dt, save_peak = measure_peak_memory(save_plain, tensors, plain_path, metadata)
    plain_size = plain_path.stat().st_size
    print(f"  Save: {save_dt:.3f}s | file: {format_bytes(plain_size)} | peak mem delta: {format_bytes(save_peak)}")

    # Load
    (loaded, load_dt), load_peak = measure_peak_memory(load_plain, plain_path)
    print(f"  Load: {load_dt:.3f}s | peak mem delta: {format_bytes(load_peak)}")

    # Verify
    assert len(loaded) == n_tensors, f"Tensor count mismatch: {len(loaded)} vs {n_tensors}"

    results["plain"] = {
        "file_size": plain_size,
        "save_time": save_dt,
        "load_time": load_dt,
        "save_peak_mem": save_peak,
        "load_peak_mem": load_peak,
    }
    print()

    # ── Zstd at various levels ─────────────────────────────────────────
    for level in ZSTD_LEVELS:
        print("─" * 72)
        print(f"ZSTD level={level} (bulk read/write)")
        print("─" * 72)

        zstd_path = CACHE_DIR / f"cache_zstd_l{level}.zst"

        # Save
        save_dt, save_peak = measure_peak_memory(save_zstd, tensors, zstd_path, metadata, level)
        zstd_size = zstd_path.stat().st_size
        ratio = plain_size / zstd_size
        saving_pct = (1 - zstd_size / plain_size) * 100
        print(f"  Save: {save_dt:.3f}s | file: {format_bytes(zstd_size)} | ratio: {ratio:.2f}x ({saving_pct:.1f}% smaller)")
        print(f"       peak mem delta: {format_bytes(save_peak)}")

        # Load
        (loaded_z, load_dt), load_peak = measure_peak_memory(load_zstd, zstd_path)
        print(f"  Load: {load_dt:.3f}s | peak mem delta: {format_bytes(load_peak)}")

        assert len(loaded_z) == n_tensors

        results[f"zstd_l{level}_bulk"] = {
            "file_size": zstd_size,
            "save_time": save_dt,
            "load_time": load_dt,
            "save_peak_mem": save_peak,
            "load_peak_mem": load_peak,
            "compression_ratio": ratio,
            "saving_pct": saving_pct,
        }
        print()

    # ── Zstd streaming at level 3 ─────────────────────────────────────
    print("─" * 72)
    print(f"ZSTD level=3 (streaming — lower peak memory)")
    print("─" * 72)

    zstd_stream_path = CACHE_DIR / "cache_zstd_l3_stream.zst"

    save_dt, save_peak = measure_peak_memory(save_zstd_streaming, tensors, zstd_stream_path, metadata, 3)
    stream_size = zstd_stream_path.stat().st_size
    ratio = plain_size / stream_size
    saving_pct = (1 - stream_size / plain_size) * 100
    print(f"  Save: {save_dt:.3f}s | file: {format_bytes(stream_size)} | ratio: {ratio:.2f}x ({saving_pct:.1f}% smaller)")
    print(f"       peak mem delta: {format_bytes(save_peak)}")

    (loaded_s, load_dt), load_peak = measure_peak_memory(load_zstd_streaming, zstd_stream_path)
    print(f"  Load: {load_dt:.3f}s | peak mem delta: {format_bytes(load_peak)}")

    assert len(loaded_s) == n_tensors

    results["zstd_l3_streaming"] = {
        "file_size": stream_size,
        "save_time": save_dt,
        "load_time": load_dt,
        "save_peak_mem": save_peak,
        "load_peak_mem": load_peak,
        "compression_ratio": ratio,
        "saving_pct": saving_pct,
    }
    print()

    # ── Summary ────────────────────────────────────────────────────────
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Method':<28} {'Size':>10} {'Save':>8} {'Load':>8} {'Ratio':>7}")
    print("-" * 72)
    for name, r in results.items():
        size_str = format_bytes(r["file_size"])
        save_str = f"{r['save_time']:.3f}s"
        load_str = f"{r['load_time']:.3f}s"
        ratio_str = f"{r.get('compression_ratio', 1.0):.2f}x"
        print(f"  {name:<26} {size_str:>10} {save_str:>8} {load_str:>8} {ratio_str:>7}")

    # Save results
    results_path = CACHE_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Cleanup
    for p in CACHE_DIR.glob("cache_*"):
        p.unlink(missing_ok=True)
    print("Temp files cleaned up.")


if __name__ == "__main__":
    main()
