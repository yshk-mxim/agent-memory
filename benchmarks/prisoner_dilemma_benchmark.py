#!/usr/bin/env python3
"""Prisoner's dilemma timing benchmark -- cross-phase KV cache persistence.

Measures the benefit of persistent KV caches across the 5-phase prisoner's
dilemma scenario.  With persistent cache, agents in phases 3-5 reuse cached
KV state from earlier phases (warm/hot EXTEND match).  The cold baseline
clears all caches before every phase, forcing full re-prefill each time.

Usage:
    # Run full benchmark (both modes) against server at localhost:8000
    python benchmarks/prisoner_dilemma_benchmark.py

    # Only persistent mode
    python benchmarks/prisoner_dilemma_benchmark.py --mode persistent

    # Only cold baseline
    python benchmarks/prisoner_dilemma_benchmark.py --mode cold

    # Clear caches and exit
    python benchmarks/prisoner_dilemma_benchmark.py --clear-only

    # Custom server URL
    python benchmarks/prisoner_dilemma_benchmark.py --base-url http://localhost:8005
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8000"
ADMIN_KEY = "benchmark"
SCENARIO_PREFIX = "prisoners-dilemma"

RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Agent definitions (full system prompts from prisoners_dilemma.yaml)
# ---------------------------------------------------------------------------

AGENTS: dict[str, dict[str, str]] = {
    "warden": {
        "display_name": "Warden",
        "role": "moderator",
        "system_prompt": (
            "You are a hard, cynical corrections officer. You enjoy watching suspects squirm. "
            "THE DEAL: both silent = 2 years each. One rats, the other doesn't = rat walks, "
            "silent one gets 10. Both rat = 5 each. "
            "You play dirty. Imply the other suspect is already talking. Mock their loyalty. "
            "Speak in short, blunt sentences. Show zero sympathy. Under 3 sentences."
        ),
        "lifecycle": "permanent",
    },
    "marco": {
        "display_name": "Marco",
        "role": "participant",
        "system_prompt": (
            "You are a scared dockworker, 34, first time inside. Your wife Maria and two kids need you home. "
            "You speak rough, working-class. Say \"damn\", \"hell\", \"man\" naturally. "
            "You are torn between loyalty and survival. Ten years terrifies you. "
            "You want to believe people keep their word but you know the world does not work that way. "
            "Never speak in a formal or polite way. Sound like a real person under pressure. Under 3 sentences."
        ),
        "lifecycle": "permanent",
    },
    "danny": {
        "display_name": "Danny",
        "role": "participant",
        "system_prompt": (
            "You are a two-time felon, 41, who has done hard time before. You know every con in the book. "
            "You speak street -- blunt, sarcastic, suspicious. Say \"hell\", \"damn\", \"man\" naturally. "
            "You trust nobody. Last time you trusted a partner he flipped on you and you did 6 years. "
            "You will always choose what gets YOU the least time, no matter what you promised anyone. "
            "Never speak in a formal or polite way. Sound like a real convict. Under 3 sentences."
        ),
        "lifecycle": "permanent",
    },
    "analyst": {
        "display_name": "Analyst",
        "role": "moderator",
        "system_prompt": (
            "Neutral classifier. Read the transcript and determine what each suspect actually chose. "
            "Classify based on their final statements. Apply the payoff rules exactly."
        ),
        "lifecycle": "ephemeral",
    },
}

# ---------------------------------------------------------------------------
# Phase definitions (matching prisoners_dilemma.yaml)
# ---------------------------------------------------------------------------

PHASES: list[dict[str, Any]] = [
    {
        "name": "interrogation_marco",
        "label": "Room A: Warden & Marco",
        "agents": ["warden", "marco"],
        "rounds": 3,
        "initial_prompt": (
            "Concrete room, one metal table, one light. The Warden drops a folder on the table. "
            "Marco is handcuffed to the chair. The Warden lays out the deal and starts pushing."
        ),
        "per_agent_prompts": None,
    },
    {
        "name": "interrogation_danny",
        "label": "Room B: Warden & Danny",
        "agents": ["warden", "danny"],
        "rounds": 3,
        "initial_prompt": (
            "Same room, one hour later. Danny is brought in cuffed. "
            "The Warden lays out the deal. Danny has been through this before."
        ),
        "per_agent_prompts": None,
    },
    {
        "name": "the_yard",
        "label": "The Yard",
        "agents": ["marco", "danny"],
        "rounds": 2,
        "initial_prompt": (
            "Exercise yard, far corner by the chain-link fence. No guards close enough to hear. "
            "Marco and Danny have five minutes before they get called back in. "
            "This is their only chance to talk before the final decision."
        ),
        "per_agent_prompts": None,
    },
    {
        "name": "final_reckoning",
        "label": "Final Reckoning",
        "agents": ["warden", "marco", "danny"],
        "rounds": 2,
        "initial_prompt": (
            "Both suspects dragged back into the room together. The Warden is standing. "
            "No more games. Time to choose."
        ),
        "per_agent_prompts": {
            "Warden": (
                "Slam the table. Give them the numbers one last time and demand they choose RIGHT NOW. "
                "Whatever they say is final. No take-backs, no more talking."
            ),
            "Marco": (
                "You just talked with Danny by the fence. Remember exactly what he said and whether you believe him. "
                "The warden wants your answer now. Say your choice and why."
            ),
            "Danny": (
                "You just talked with Marco by the fence. Remember what he said and whether you believe him. "
                "The warden wants your answer now. Say your choice and why."
            ),
        },
    },
    {
        "name": "outcome_analysis",
        "label": "The Verdict",
        "agents": ["marco", "danny", "analyst"],
        "rounds": 1,
        "initial_prompt": (
            "The warden has recorded your final decisions. State your choice clearly."
        ),
        "per_agent_prompts": {
            "Marco": "State your final choice in ONE WORD: CONFESS or SILENT.",
            "Danny": "State your final choice in ONE WORD: CONFESS or SILENT.",
            "Analyst": (
                "PAYOFF MATRIX:\n"
                "| Marco    | Danny    | Marco gets | Danny gets |\n"
                "|----------|----------|------------|------------|\n"
                "| SILENT   | SILENT   | 2 years    | 2 years    |\n"
                "| CONFESS  | SILENT   | FREE       | 10 years   |\n"
                "| SILENT   | CONFESS  | 10 years   | FREE       |\n"
                "| CONFESS  | CONFESS  | 5 years    | 5 years    |\n\n"
                "Read Marco's choice and Danny's choice above. Look up the row in the matrix. "
                'State: "Marco chose X, Danny chose Y. From the matrix: Marco gets Z, Danny gets W."'
            ),
        },
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TurnMeasurement:
    """Timing data for a single streamed turn."""

    turn_index: int
    agent_id: str
    agent_name: str
    ttft_ms: float  # time from POST to first token event
    total_ms: float  # time from POST to turn_complete event
    content_preview: str = ""  # first 80 chars of generated text


@dataclass
class PhaseMeasurement:
    """Timing data for one complete phase."""

    phase_name: str
    phase_label: str
    phase_index: int
    creation_ms: float  # session creation latency
    wall_ms: float  # total phase wall time (creation + all turns)
    turns: list[TurnMeasurement] = field(default_factory=list)
    avg_ttft_ms: float = 0.0
    agent_cache_snapshot: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkRun:
    """Complete benchmark run for one mode."""

    mode: str  # "cold_baseline" or "persistent"
    phases: list[PhaseMeasurement] = field(default_factory=list)
    total_wall_ms: float = 0.0
    total_turns: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Get short git SHA for reproducibility."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _stable_agent_id(agent_key: str) -> str:
    """Generate stable agent_id from scenario prefix + agent key."""
    return f"scn_{SCENARIO_PREFIX}_{agent_key}"


def _detect_model(base_url: str) -> str:
    """Detect loaded model from /v1/models endpoint."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                if models:
                    return models[0].get("id", "unknown")
    except httpx.HTTPError:
        pass
    return "unknown"


def _model_short_name(model_id: str) -> str:
    """Extract short name from model ID for filenames."""
    if "/" in model_id:
        name = model_id.rsplit("/", 1)[-1]
    else:
        name = model_id
    # Clean for filename
    return name.replace(" ", "_").replace("/", "_")


def _check_server_ready(base_url: str) -> bool:
    """Check if server is ready."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/health/ready")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


def _clear_all_caches(base_url: str, admin_key: str) -> bool:
    """Clear ALL caches (hot + warm + disk) via admin API."""
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.delete(
                f"{base_url}/admin/caches",
                headers={"X-Admin-Key": admin_key},
            )
            if resp.status_code == 200:
                data = resp.json()
                total = data.get("hot_cleared", 0) + data.get("disk_cleared", 0)
                print(f"    Cleared {total} cache entries "
                      f"(hot={data.get('hot_cleared', 0)}, "
                      f"disk={data.get('disk_cleared', 0)}, "
                      f"pool={data.get('pool_cleared', 0)})")
                return True
            print(f"    Cache clear failed: HTTP {resp.status_code}")
            return False
    except httpx.HTTPError as e:
        print(f"    Cache clear error: {e}")
        return False


def _delete_persistent_caches(base_url: str) -> bool:
    """Delete persistent caches for the scenario prefix."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(
                f"{base_url}/v1/coordination/caches/{SCENARIO_PREFIX}",
            )
            return resp.status_code == 204
    except httpx.HTTPError:
        return False


def _get_agent_list(base_url: str) -> list[dict[str, Any]]:
    """Get list of all cached agents."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/v1/agents/list")
            if resp.status_code == 200:
                data = resp.json()
                return data.get("agents", [])
    except httpx.HTTPError:
        pass
    return []


def _get_scenario_agents(base_url: str) -> list[dict[str, Any]]:
    """Get agent cache info filtered to our scenario agents."""
    all_agents = _get_agent_list(base_url)
    prefix = f"scn_{SCENARIO_PREFIX}_"
    return [a for a in all_agents if a.get("agent_id", "").startswith(prefix)]


def _get_session_messages(
    base_url: str, session_id: str
) -> list[dict[str, Any]]:
    """Get all messages from a coordination session."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{base_url}/v1/coordination/sessions/{session_id}/messages",
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("messages", [])
    except httpx.HTTPError:
        pass
    return []


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------


def _build_agent_configs(phase: dict[str, Any]) -> list[dict[str, Any]]:
    """Build agent config list for session creation."""
    configs = []
    for agent_key in phase["agents"]:
        agent = AGENTS[agent_key]
        configs.append({
            "agent_id": _stable_agent_id(agent_key),
            "display_name": agent["display_name"],
            "role": agent["role"],
            "system_prompt": agent["system_prompt"],
            "lifecycle": agent["lifecycle"],
        })
    return configs


def _collect_prior_messages(
    phase_index: int,
    phase_messages: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, str]]]:
    """Collect prior phase messages for each permanent agent.

    For each permanent agent in the current phase, gather all messages from
    all prior phases where that agent participated.  This enables KV cache
    prefix matching (EXTEND) on the server.
    """
    current_phase = PHASES[phase_index]
    prior: dict[str, list[dict[str, str]]] = {}

    for agent_key in current_phase["agents"]:
        agent = AGENTS[agent_key]
        if agent["lifecycle"] != "permanent":
            continue

        agent_id = _stable_agent_id(agent_key)
        agent_msgs: list[dict[str, str]] = []

        for prior_idx in range(phase_index):
            prior_phase = PHASES[prior_idx]
            if agent_key not in prior_phase["agents"]:
                continue

            stored = phase_messages.get(prior_phase["name"], [])
            for msg in stored:
                sender_name = msg.get("sender_name", "")
                content = msg.get("content", "")
                if not content:
                    continue
                if sender_name == "System":
                    sender_id = "system"
                else:
                    # Find agent key by display name
                    sender_key = None
                    for k, v in AGENTS.items():
                        if v["display_name"] == sender_name:
                            sender_key = k
                            break
                    sender_id = (
                        _stable_agent_id(sender_key) if sender_key
                        else msg.get("sender_id", "system")
                    )
                agent_msgs.append({
                    "sender_id": sender_id,
                    "sender_name": sender_name,
                    "content": content,
                })

        if agent_msgs:
            prior[agent_id] = agent_msgs

    return prior


def _create_phase_session(
    base_url: str,
    phase_index: int,
    phase_messages: dict[str, list[dict[str, Any]]],
) -> str | None:
    """Create a coordination session for a phase. Returns session_id or None."""
    phase = PHASES[phase_index]
    agent_configs = _build_agent_configs(phase)
    prior_agent_messages = _collect_prior_messages(phase_index, phase_messages)
    total_turns = phase["rounds"] * len(phase["agents"])

    payload: dict[str, Any] = {
        "topology": "round_robin",
        "debate_format": "free_form",
        "decision_mode": "none",
        "agents": agent_configs,
        "initial_prompt": phase["initial_prompt"],
        "max_turns": total_turns,
        "persistent_cache_prefix": SCENARIO_PREFIX,
    }
    if phase.get("per_agent_prompts"):
        payload["per_agent_prompts"] = phase["per_agent_prompts"]
    if prior_agent_messages:
        payload["prior_agent_messages"] = prior_agent_messages

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{base_url}/v1/coordination/sessions",
                json=payload,
            )
            if resp.status_code == 201:
                return resp.json().get("session_id")
            print(f"    Session creation failed: HTTP {resp.status_code} - {resp.text[:200]}")
    except httpx.HTTPError as e:
        print(f"    Session creation error: {e}")
    return None


# ---------------------------------------------------------------------------
# Streaming turn execution with TTFT measurement
# ---------------------------------------------------------------------------


def _stream_turn(
    base_url: str, session_id: str
) -> TurnMeasurement | None:
    """Execute one streamed turn, measuring TTFT and total time.

    Streams POST /v1/coordination/sessions/{id}/turn/stream and parses SSE
    events to measure:
      - TTFT: time from request send to first 'token' event
      - Total: time from request send to 'turn_complete' event
    """
    turn_index = -1
    agent_id = ""
    agent_name = ""
    first_token_time: float | None = None
    content = ""

    t_start = time.perf_counter()

    try:
        with httpx.Client(timeout=180.0) as client:
            with client.stream(
                "POST",
                f"{base_url}/v1/coordination/sessions/{session_id}/turn/stream",
            ) as resp:
                if resp.status_code != 200:
                    print(f"    Stream failed: HTTP {resp.status_code}")
                    return None

                for line in resp.iter_lines():
                    if line.startswith("event: "):
                        event_type = line[7:]
                    elif line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue

                        if event_type == "turn_start":
                            agent_id = data.get("agent_id", "")
                            agent_name = data.get("agent_name", "")
                            turn_index = data.get("turn", -1)
                        elif event_type == "token":
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                        elif event_type == "turn_complete":
                            content = data.get("content", "")
                        elif event_type == "error":
                            print(f"    Stream error: {data.get('error', '?')}")
                            return None

    except httpx.HTTPError as e:
        print(f"    Stream connection error: {e}")
        return None

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000
    ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else total_ms

    return TurnMeasurement(
        turn_index=turn_index,
        agent_id=agent_id,
        agent_name=agent_name,
        ttft_ms=round(ttft_ms, 1),
        total_ms=round(total_ms, 1),
        content_preview=content[:80] if content else "",
    )


# ---------------------------------------------------------------------------
# Phase execution
# ---------------------------------------------------------------------------


def _run_phase(
    base_url: str,
    phase_index: int,
    phase_messages: dict[str, list[dict[str, Any]]],
) -> PhaseMeasurement:
    """Run a single phase: create session, stream all turns, collect messages."""
    phase = PHASES[phase_index]
    total_turns = phase["rounds"] * len(phase["agents"])

    print(f"\n  Phase {phase_index + 1}/{len(PHASES)}: {phase['label']}")
    print(f"    Agents: {', '.join(phase['agents'])} | "
          f"Rounds: {phase['rounds']} | Turns: {total_turns}")

    # Check if any agents have prior messages
    prior = _collect_prior_messages(phase_index, phase_messages)
    if prior:
        for aid, msgs in prior.items():
            print(f"    Prior messages for {aid}: {len(msgs)} messages")

    # Create session
    t_create_start = time.perf_counter()
    session_id = _create_phase_session(base_url, phase_index, phase_messages)
    creation_ms = (time.perf_counter() - t_create_start) * 1000

    if not session_id:
        return PhaseMeasurement(
            phase_name=phase["name"],
            phase_label=phase["label"],
            phase_index=phase_index,
            creation_ms=round(creation_ms, 1),
            wall_ms=round(creation_ms, 1),
        )

    print(f"    Session created: {session_id} ({creation_ms:.0f}ms)")

    # Stream all turns
    t_phase_start = time.perf_counter()
    turns: list[TurnMeasurement] = []

    for turn_i in range(total_turns):
        measurement = _stream_turn(base_url, session_id)
        if measurement is None:
            print(f"    Turn {turn_i + 1}/{total_turns} FAILED")
            continue

        turns.append(measurement)
        preview = measurement.content_preview[:50].replace("\n", " ")
        print(f"    Turn {turn_i + 1}/{total_turns}: "
              f"{measurement.agent_name:>8} | "
              f"TTFT={measurement.ttft_ms:>7.0f}ms | "
              f"Total={measurement.total_ms:>7.0f}ms | "
              f"\"{preview}...\"")

    phase_wall_ms = (time.perf_counter() - t_phase_start) * 1000 + creation_ms

    # Fetch messages for subsequent phases
    messages = _get_session_messages(base_url, session_id)
    phase_messages[phase["name"]] = messages

    # Snapshot agent cache state
    agent_snapshot = _get_scenario_agents(base_url)

    # Compute average TTFT
    ttft_values = [t.ttft_ms for t in turns]
    avg_ttft = statistics.mean(ttft_values) if ttft_values else 0.0

    result = PhaseMeasurement(
        phase_name=phase["name"],
        phase_label=phase["label"],
        phase_index=phase_index,
        creation_ms=round(creation_ms, 1),
        wall_ms=round(phase_wall_ms, 1),
        turns=turns,
        avg_ttft_ms=round(avg_ttft, 1),
        agent_cache_snapshot=agent_snapshot,
    )

    print(f"    Phase complete: wall={phase_wall_ms:.0f}ms, "
          f"avg_ttft={avg_ttft:.0f}ms, "
          f"cached_agents={len(agent_snapshot)}")

    return result


# ---------------------------------------------------------------------------
# Benchmark modes
# ---------------------------------------------------------------------------


def run_cold_baseline(base_url: str, admin_key: str) -> BenchmarkRun:
    """Run all 5 phases, clearing ALL caches before each phase.

    This forces every phase to cold-start from scratch, giving the baseline
    cost without any KV cache persistence benefit.
    """
    print("\n" + "=" * 70)
    print("  MODE: COLD BASELINE (caches cleared before every phase)")
    print("=" * 70)

    run = BenchmarkRun(mode="cold_baseline")
    phase_messages: dict[str, list[dict[str, Any]]] = {}
    t_total_start = time.perf_counter()

    for phase_idx in range(len(PHASES)):
        # Clear ALL caches before this phase
        print(f"\n    Clearing all caches before phase {phase_idx + 1}...")
        _clear_all_caches(base_url, admin_key)
        time.sleep(1.0)  # Let cleanup settle

        measurement = _run_phase(base_url, phase_idx, phase_messages)
        run.phases.append(measurement)
        run.total_turns += len(measurement.turns)

    run.total_wall_ms = round((time.perf_counter() - t_total_start) * 1000, 1)
    print(f"\n  Cold baseline complete: total_wall={run.total_wall_ms:.0f}ms, "
          f"turns={run.total_turns}")
    return run


def run_persistent(base_url: str, admin_key: str) -> BenchmarkRun:
    """Run all 5 phases with persistent cache enabled (the designed behavior).

    Caches are cleared once at the start, then left intact across phases.
    Agents accumulate KV state across phases, so later phases benefit from
    EXTEND-match cache hits.
    """
    print("\n" + "=" * 70)
    print("  MODE: PERSISTENT (caches preserved across phases)")
    print("=" * 70)

    # Clear everything once at the start for a fair comparison
    print("\n    Initial cache clear...")
    _clear_all_caches(base_url, admin_key)
    time.sleep(1.0)

    run = BenchmarkRun(mode="persistent")
    phase_messages: dict[str, list[dict[str, Any]]] = {}
    t_total_start = time.perf_counter()

    for phase_idx in range(len(PHASES)):
        measurement = _run_phase(base_url, phase_idx, phase_messages)
        run.phases.append(measurement)
        run.total_turns += len(measurement.turns)

    run.total_wall_ms = round((time.perf_counter() - t_total_start) * 1000, 1)
    print(f"\n  Persistent mode complete: total_wall={run.total_wall_ms:.0f}ms, "
          f"turns={run.total_turns}")
    return run


# ---------------------------------------------------------------------------
# Summary & output
# ---------------------------------------------------------------------------


def print_summary(cold: BenchmarkRun | None, persistent: BenchmarkRun | None) -> None:
    """Print a comparison table of cold vs persistent TTFT per phase."""
    print("\n" + "=" * 90)
    print("  CROSS-PHASE KV CACHE PERSISTENCE -- SUMMARY")
    print("=" * 90)

    header = (
        f"{'Phase':<25} | "
        f"{'Cold TTFT':>10} {'Cold Wall':>10} | "
        f"{'Pers TTFT':>10} {'Pers Wall':>10} | "
        f"{'TTFT Speedup':>12} {'Wall Speedup':>12}"
    )
    print(f"\n  {header}")
    print(f"  {'-' * len(header)}")

    cold_phases = {p.phase_name: p for p in (cold.phases if cold else [])}
    pers_phases = {p.phase_name: p for p in (persistent.phases if persistent else [])}

    for phase_def in PHASES:
        name = phase_def["name"]
        label = phase_def["label"]

        cold_p = cold_phases.get(name)
        pers_p = pers_phases.get(name)

        cold_ttft = cold_p.avg_ttft_ms if cold_p else 0
        cold_wall = cold_p.wall_ms if cold_p else 0
        pers_ttft = pers_p.avg_ttft_ms if pers_p else 0
        pers_wall = pers_p.wall_ms if pers_p else 0

        ttft_speedup = ""
        wall_speedup = ""
        if cold_ttft > 0 and pers_ttft > 0:
            ratio = cold_ttft / pers_ttft
            pct = ((cold_ttft - pers_ttft) / cold_ttft) * 100
            ttft_speedup = f"{ratio:.2f}x ({pct:+.0f}%)"
        if cold_wall > 0 and pers_wall > 0:
            ratio = cold_wall / pers_wall
            pct = ((cold_wall - pers_wall) / cold_wall) * 100
            wall_speedup = f"{ratio:.2f}x ({pct:+.0f}%)"

        row = (
            f"  {label:<25} | "
            f"{cold_ttft:>9.0f}ms {cold_wall:>9.0f}ms | "
            f"{pers_ttft:>9.0f}ms {pers_wall:>9.0f}ms | "
            f"{ttft_speedup:>12} {wall_speedup:>12}"
        )
        print(row)

    # Totals
    print(f"  {'-' * len(header)}")
    cold_total = cold.total_wall_ms if cold else 0
    pers_total = persistent.total_wall_ms if persistent else 0
    total_speedup = ""
    if cold_total > 0 and pers_total > 0:
        ratio = cold_total / pers_total
        pct = ((cold_total - pers_total) / cold_total) * 100
        total_speedup = f"{ratio:.2f}x ({pct:+.0f}%)"
    print(f"  {'TOTAL':<25} | "
          f"{'':>10} {cold_total:>9.0f}ms | "
          f"{'':>10} {pers_total:>9.0f}ms | "
          f"{'':>12} {total_speedup:>12}")

    # Agent cache growth (from persistent run)
    if persistent and persistent.phases:
        print(f"\n  {'Agent Cache State (persistent mode)':}")
        print(f"  {'Phase':<25} | {'Agents':>7} | {'Details'}")
        print(f"  {'-' * 70}")
        for phase_m in persistent.phases:
            agents_str = ""
            for a in phase_m.agent_cache_snapshot:
                aid = a.get("agent_id", "?")
                short_id = aid.split("_")[-1] if "_" in aid else aid
                tier = a.get("tier", "?")
                tokens = a.get("tokens", 0)
                agents_str += f"{short_id}({tier},{tokens}tok) "
            print(f"  {phase_m.phase_label:<25} | "
                  f"{len(phase_m.agent_cache_snapshot):>7} | "
                  f"{agents_str.strip()}")

    print()


def save_results(
    model_id: str,
    cold: BenchmarkRun | None,
    persistent: BenchmarkRun | None,
) -> Path:
    """Save benchmark results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    short_model = _model_short_name(model_id)
    output_path = RESULTS_DIR / f"prisoner_dilemma_{short_model}.json"

    def _run_to_dict(run: BenchmarkRun) -> dict[str, Any]:
        return {
            "mode": run.mode,
            "total_wall_ms": run.total_wall_ms,
            "total_turns": run.total_turns,
            "error": run.error,
            "phases": [
                {
                    "phase_name": p.phase_name,
                    "phase_label": p.phase_label,
                    "phase_index": p.phase_index,
                    "creation_ms": p.creation_ms,
                    "wall_ms": p.wall_ms,
                    "avg_ttft_ms": p.avg_ttft_ms,
                    "turns": [
                        {
                            "turn_index": t.turn_index,
                            "agent_id": t.agent_id,
                            "agent_name": t.agent_name,
                            "ttft_ms": t.ttft_ms,
                            "total_ms": t.total_ms,
                            "content_preview": t.content_preview,
                        }
                        for t in p.turns
                    ],
                    "agent_cache_snapshot": p.agent_cache_snapshot,
                }
                for p in run.phases
            ],
        }

    output: dict[str, Any] = {
        "benchmark": "prisoner_dilemma_cache_persistence",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "scenario": {
            "id": SCENARIO_PREFIX,
            "total_phases": len(PHASES),
            "agents": list(AGENTS.keys()),
            "phase_names": [p["name"] for p in PHASES],
        },
    }

    if cold:
        output["cold_baseline"] = _run_to_dict(cold)
    if persistent:
        output["persistent"] = _run_to_dict(persistent)

    # Compute comparison metrics
    if cold and persistent:
        comparison: list[dict[str, Any]] = []
        for c_phase, p_phase in zip(cold.phases, persistent.phases):
            entry: dict[str, Any] = {
                "phase_name": c_phase.phase_name,
                "cold_avg_ttft_ms": c_phase.avg_ttft_ms,
                "persistent_avg_ttft_ms": p_phase.avg_ttft_ms,
                "cold_wall_ms": c_phase.wall_ms,
                "persistent_wall_ms": p_phase.wall_ms,
            }
            if c_phase.avg_ttft_ms > 0 and p_phase.avg_ttft_ms > 0:
                entry["ttft_speedup"] = round(
                    c_phase.avg_ttft_ms / p_phase.avg_ttft_ms, 3
                )
                entry["ttft_reduction_pct"] = round(
                    ((c_phase.avg_ttft_ms - p_phase.avg_ttft_ms)
                     / c_phase.avg_ttft_ms) * 100, 1
                )
            if c_phase.wall_ms > 0 and p_phase.wall_ms > 0:
                entry["wall_speedup"] = round(
                    c_phase.wall_ms / p_phase.wall_ms, 3
                )
            comparison.append(entry)
        output["comparison"] = comparison

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prisoner's dilemma cross-phase KV cache persistence benchmark",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=BASE_URL,
        help=f"Server base URL (default: {BASE_URL})",
    )
    parser.add_argument(
        "--admin-key",
        type=str,
        default=ADMIN_KEY,
        help=f"Admin API key for cache clearing (default: {ADMIN_KEY})",
    )
    parser.add_argument(
        "--mode",
        choices=["cold", "persistent", "both"],
        default="both",
        help="Which benchmark mode(s) to run (default: both)",
    )
    parser.add_argument(
        "--clear-only",
        action="store_true",
        help="Clear all caches and exit without running benchmark",
    )
    args = parser.parse_args()

    base_url = args.base_url
    admin_key = args.admin_key

    # Check server readiness
    print(f"Checking server at {base_url}...")
    if not _check_server_ready(base_url):
        print("ERROR: Server is not ready. Start with:")
        print(f"  python -m agent_memory.entrypoints.cli serve --port 8000")
        sys.exit(1)

    model_id = _detect_model(base_url)
    print(f"Model: {model_id}")

    # Clear-only mode
    if args.clear_only:
        print("\nClearing all caches...")
        _clear_all_caches(base_url, admin_key)
        _delete_persistent_caches(base_url)
        print("Done.")
        return

    print(f"\nBenchmark: Prisoner's Dilemma Cross-Phase Cache Persistence")
    print(f"Mode: {args.mode}")
    print(f"Phases: {len(PHASES)}")
    total_turns = sum(p["rounds"] * len(p["agents"]) for p in PHASES)
    print(f"Total turns per run: {total_turns}")

    cold_run: BenchmarkRun | None = None
    persistent_run: BenchmarkRun | None = None

    # Run cold baseline first (to avoid cache warming from persistent run)
    if args.mode in ("cold", "both"):
        try:
            cold_run = run_cold_baseline(base_url, admin_key)
        except Exception as e:
            print(f"\nERROR in cold baseline: {e}")
            cold_run = BenchmarkRun(mode="cold_baseline", error=str(e))

    # Clean slate before persistent run
    if args.mode in ("persistent", "both"):
        if args.mode == "both":
            print("\n  Clearing caches between modes...")
            _clear_all_caches(base_url, admin_key)
            time.sleep(2.0)

        try:
            persistent_run = run_persistent(base_url, admin_key)
        except Exception as e:
            print(f"\nERROR in persistent mode: {e}")
            persistent_run = BenchmarkRun(mode="persistent", error=str(e))

    # Print summary
    print_summary(cold_run, persistent_run)

    # Save results
    output_path = save_results(model_id, cold_run, persistent_run)
    print(f"Results saved to: {output_path}")

    # Final cleanup
    print("\nFinal cache cleanup...")
    _clear_all_caches(base_url, admin_key)
    print("Done.")


if __name__ == "__main__":
    main()
