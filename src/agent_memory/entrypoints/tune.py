# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Auto-tuning CLI for agent-memory server.

Runs a subset of profiling experiments on the user's hardware to find
optimal inference parameters, then writes a per-model TOML profile.

Usage:
    agent-memory tune
    agent-memory tune --quick
    agent-memory tune --output config/models/my-model.toml
"""

from __future__ import annotations

import asyncio
import logging
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import typer

logger = logging.getLogger(__name__)

# Ensure benchmarks directory is importable
_benchmarks_dir = str(Path(__file__).resolve().parents[3] / "benchmarks")
if _benchmarks_dir not in sys.path:
    sys.path.insert(0, _benchmarks_dir)


def _run_tune(
    quick: bool = False,
    output: str | None = None,
    port: int = 8399,
) -> None:
    """Run auto-tuning experiments and generate config profile."""
    from capability_benchmark import (
        MemoryProbe,
        PromptFactory,
        RequestClient,
        ServerManager,
    )

    base_url = f"http://127.0.0.1:{port}"
    server = ServerManager(port=port)
    prompt = PromptFactory(base_url)
    memory = MemoryProbe(base_url)
    runs = 2 if quick else 3

    base_env: dict[str, str] = {
        "SEMANTIC_MLX_CHUNKED_PREFILL_ENABLED": "true",
        "SEMANTIC_MLX_CHUNKED_PREFILL_THRESHOLD": "2048",
        "SEMANTIC_MLX_CHUNKED_PREFILL_MIN_CHUNK": "512",
        "SEMANTIC_MLX_CHUNKED_PREFILL_MAX_CHUNK": "4096",
        "SEMANTIC_MLX_DEFAULT_TEMPERATURE": "0.0",
        "SEMANTIC_MLX_KV_BITS": "4",
        "SEMANTIC_MLX_MAX_CONTEXT_LENGTH": "100000",
        "SEMANTIC_MLX_MAX_BATCH_SIZE": "1",
        "SEMANTIC_MLX_SCHEDULER_ENABLED": "false",
        "SEMANTIC_SERVER_LOG_LEVEL": "WARNING",
        "SEMANTIC_API_KEY": "",
    }

    async def measure(
        env: dict[str, str],
        target_tokens: int,
        max_tokens: int = 64,
        session_id: str = "tune",
        n_concurrent: int = 1,
    ) -> dict[str, float]:
        """Run one measurement and return metrics."""
        client = RequestClient(base_url)
        try:
            body = prompt.build_request(target_tokens, max_tokens)

            if n_concurrent > 1:
                t_start = time.perf_counter()
                coros = [
                    client.send_and_measure(body, session_id=f"{session_id}_{i}")
                    for i in range(n_concurrent)
                ]
                results_list = await asyncio.gather(*coros)
                wall_ms = (time.perf_counter() - t_start) * 1000
                total_out = sum(r.output_tokens for r in results_list)
                system_tps = total_out / (wall_ms / 1000) if wall_ms > 0 else 0
                # Cleanup
                for i in range(n_concurrent):
                    async with httpx.AsyncClient(timeout=5) as c:
                        try:
                            await c.delete(f"{base_url}/v1/agents/sess_{session_id}_{i}")
                        except Exception:
                            pass
                return {
                    "e2e_ms": wall_ms,
                    "system_tps": system_tps,
                    "output_tokens": total_out,
                }

            r = await client.send_and_measure(body, session_id=session_id)
            # Cleanup
            async with httpx.AsyncClient(timeout=5) as c:
                try:
                    await c.delete(f"{base_url}/v1/agents/sess_{session_id}")
                except Exception:
                    pass

            try:
                mem = await memory.snapshot()
            except Exception:
                mem = {}

            return {
                "e2e_ms": r.e2e_ms,
                "tps": r.decode_tps,
                "output_tokens": r.output_tokens,
                "peak_mb": mem.get("peak_memory_mb", 0),
                "error": r.error,
            }
        finally:
            await client.close()

    async def run_sweep(
        label: str,
        param_key: str,
        param_values: list[str],
        target_tokens: int,
        extra_env: dict[str, str] | None = None,
        n_concurrent: int = 1,
    ) -> dict[str, list[dict]]:
        """Sweep one parameter, restarting server for each value."""
        sweep_results: dict[str, list[dict]] = {}

        for val in param_values:
            env = {**base_env, **(extra_env or {}), param_key: val}
            server.start(env)

            try:
                # Warmup
                client = RequestClient(base_url)
                try:
                    body = prompt.build_request(500, 8)
                    await client.send_and_measure(body, session_id="warmup")
                    async with httpx.AsyncClient(timeout=5) as c:
                        try:
                            await c.delete(f"{base_url}/v1/agents/sess_warmup")
                        except Exception:
                            pass
                finally:
                    await client.close()
                await asyncio.sleep(1)

                run_data: list[dict] = []
                for i in range(runs):
                    m = await measure(
                        env,
                        target_tokens,
                        session_id=f"sweep_{label}_{val}_{i}",
                        n_concurrent=n_concurrent,
                    )
                    run_data.append(m)
                    tps = m.get("tps", m.get("system_tps", 0))
                    typer.echo(
                        f"    {label}={val} run {i + 1}/{runs}: "
                        f"E2E={m['e2e_ms']:.0f}ms "
                        f"TPS={tps:.1f} "
                        f"peak={m.get('peak_mb', 0):.0f}MB"
                    )

                sweep_results[val] = run_data
            finally:
                server.stop()

        return sweep_results

    async def tune_all() -> dict[str, Any]:
        """Run all tuning experiments."""
        optimal: dict[str, Any] = {}

        # --- Experiment 1: Prefill step size ---
        typer.echo("\n[1/3] Sweeping prefill_step_size...")
        input_sizes = [200, 2000, 8000] if quick else [200, 1000, 2000, 4000, 8000]
        step_values = ["256", "512", "1024"] if quick else ["128", "256", "512", "1024", "2048"]

        best_step = "512"
        best_combined_tps = 0.0
        for step in step_values:
            total_tps = 0.0
            count = 0
            env = {
                **base_env,
                "SEMANTIC_MLX_PREFILL_STEP_SIZE": step,
            }
            server.start(env)
            try:
                client = RequestClient(base_url)
                try:
                    await client.send_and_measure(
                        prompt.build_request(500, 8),
                        session_id="warmup",
                    )
                finally:
                    await client.close()
                await asyncio.sleep(1)

                for tok in input_sizes:
                    m = await measure(env, tok, session_id=f"step_{step}_{tok}")
                    tps = m.get("tps", 0)
                    total_tps += tps
                    count += 1
                    typer.echo(
                        f"    step={step} input={tok}: E2E={m['e2e_ms']:.0f}ms TPS={tps:.1f}"
                    )
            finally:
                server.stop()

            avg_tps = total_tps / count if count else 0
            if avg_tps > best_combined_tps:
                best_combined_tps = avg_tps
                best_step = step

        optimal["prefill_step_size"] = int(best_step)
        typer.echo(f"  → Best prefill_step_size: {best_step}")

        # --- Experiment 2: Batch size ---
        typer.echo("\n[2/3] Sweeping max_batch_size...")
        batch_values = ["1", "2", "3"] if quick else ["1", "2", "3", "4"]

        best_batch = "2"
        best_sys_tps = 0.0
        for bs in batch_values:
            n = int(bs)
            extra = {
                "SEMANTIC_MLX_SCHEDULER_ENABLED": "true" if n > 1 else "false",
                "SEMANTIC_MLX_PREFILL_STEP_SIZE": best_step,
            }
            sweep = await run_sweep(
                "batch",
                "SEMANTIC_MLX_MAX_BATCH_SIZE",
                [bs],
                target_tokens=2000,
                extra_env=extra,
                n_concurrent=n,
            )
            if bs in sweep:
                avg_tps = sum(r.get("system_tps", r.get("tps", 0)) for r in sweep[bs]) / max(
                    len(sweep[bs]), 1
                )
                if avg_tps > best_sys_tps:
                    best_sys_tps = avg_tps
                    best_batch = bs

        optimal["max_batch_size"] = int(best_batch)
        typer.echo(f"  → Best max_batch_size: {best_batch}")

        # --- Experiment 3: Output length ---
        typer.echo("\n[3/3] Measuring output length scaling...")
        out_lengths = [16, 64, 256] if quick else [16, 32, 64, 128, 256, 512]

        env = {
            **base_env,
            "SEMANTIC_MLX_PREFILL_STEP_SIZE": best_step,
        }
        server.start(env)
        try:
            client = RequestClient(base_url)
            try:
                await client.send_and_measure(prompt.build_request(500, 8), session_id="warmup")
            finally:
                await client.close()
            await asyncio.sleep(1)

            decode_tps_values = []
            for out_len in out_lengths:
                m = await measure(env, 1000, max_tokens=out_len, session_id=f"outlen_{out_len}")
                tps = m.get("tps", 0)
                decode_tps_values.append(tps)
                typer.echo(f"    output={out_len}: E2E={m['e2e_ms']:.0f}ms TPS={tps:.1f}")
        finally:
            server.stop()

        if decode_tps_values:
            optimal["avg_decode_tps"] = sum(decode_tps_values) / len(decode_tps_values)
            typer.echo(f"  → Average decode TPS: {optimal['avg_decode_tps']:.1f}")

        return optimal

    # Execute tuning
    typer.echo("=" * 60)
    typer.echo("  agent-memory Auto-Tuning")
    typer.echo("=" * 60)
    typer.echo(f"Mode: {'quick' if quick else 'full'}")
    typer.echo(f"Runs per scenario: {runs}")

    optimal = asyncio.run(tune_all())

    # Write TOML profile
    output_path = (
        Path(output)
        if output
        else (Path(__file__).resolve().parents[3] / "config" / "models" / "tuned.toml")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from agent_memory.adapters.config.settings import get_settings

    settings = get_settings()

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    # Read existing profile as base if available
    existing_profile = Path(__file__).resolve().parents[3] / "config" / "models"
    base_profile: dict[str, Any] = {}
    for f in existing_profile.glob("*.toml"):
        if "tuned" not in f.stem:
            with f.open("rb") as fh:
                base_profile = tomllib.load(fh)
            break

    # Build TOML content
    lines = [
        "# Auto-tuned configuration",
        f"# Generated: {datetime.now(UTC).isoformat()}",
        f"# Hardware: {platform.machine()} / {platform.system()} {platform.release()}",
        f"# Mode: {'quick' if quick else 'full'}",
        "",
        "[model]",
        f'model_id = "{settings.mlx.model_id}"',
        "",
        "[optimal]",
        f"max_batch_size = {optimal.get('max_batch_size', 2)}",
        f"prefill_step_size = {optimal.get('prefill_step_size', 512)}",
        f"kv_bits = {settings.mlx.kv_bits or 4}",
        f"kv_group_size = {settings.mlx.kv_group_size}",
        "chunked_prefill_enabled = true",
        f"chunked_prefill_threshold = {settings.mlx.chunked_prefill_threshold}",
        f"chunked_prefill_min_chunk = {settings.mlx.chunked_prefill_min_chunk}",
        f"chunked_prefill_max_chunk = {settings.mlx.chunked_prefill_max_chunk}",
        f"batch_window_ms = {settings.agent.batch_window_ms}",
        "scheduler_enabled = true"
        if optimal.get("max_batch_size", 1) > 1
        else "scheduler_enabled = false",
        f"max_agents_in_memory = {settings.agent.max_agents_in_memory}",
        "evict_to_disk = true",
        "",
        "[thresholds]",
        "long_context_threshold = {}".format(
            base_profile.get("thresholds", {}).get("long_context_threshold", 4000)
        ),
        "high_batch_threshold = {}".format(
            base_profile.get("thresholds", {}).get("high_batch_threshold", 3)
        ),
        "memory_pressure_mb = {}".format(
            base_profile.get("thresholds", {}).get("memory_pressure_mb", 12000)
        ),
        "min_cache_benefit_ratio = {}".format(
            base_profile.get("thresholds", {}).get("min_cache_benefit_ratio", 0.8)
        ),
        "",
        "[tuning_results]",
        f"avg_decode_tps = {optimal.get('avg_decode_tps', 0):.1f}",
        f"best_prefill_step = {optimal.get('prefill_step_size', 512)}",
        f"best_batch_size = {optimal.get('max_batch_size', 2)}",
    ]

    with output_path.open("w") as f:
        f.write("\n".join(lines) + "\n")

    typer.echo(f"\n{'=' * 60}")
    typer.echo("  Tuning complete!")
    typer.echo(f"  Profile written to: {output_path}")
    typer.echo(f"{'=' * 60}")
    typer.echo("\nOptimal settings:")
    typer.echo(f"  prefill_step_size = {optimal.get('prefill_step_size', 512)}")
    typer.echo(f"  max_batch_size = {optimal.get('max_batch_size', 2)}")
    typer.echo(f"  avg_decode_tps = {optimal.get('avg_decode_tps', 0):.1f}")
