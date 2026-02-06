#!/usr/bin/env python3
"""Run prisoner's dilemma and dump FULL message content for semantic review.

Starts server, runs scenario, prints every message, kills server.
Tests both Gemma 3 and DeepSeek sequentially (one model at a time).
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from demo.lib import api_client
from semantic.adapters.config.scenario_loader import load_scenario

SCENARIO_PATH = Path(__file__).parent / "demo" / "scenarios" / "prisoners_dilemma.yaml"
CACHE_DIR = Path.home() / ".semantic" / "caches"


def clear_caches():
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.safetensors"):
            f.unlink()
    print(f"  Caches cleared")


def start_server(model_id: str, port: int) -> subprocess.Popen:
    """Start semantic server and wait for health check."""
    cmd = [
        sys.executable, "-m", "semantic.entrypoints.cli",
        "serve", "--port", str(port),
    ]
    if model_id:
        cmd.extend(["--model", model_id])

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(f"  Started server PID={proc.pid}, waiting for health...")

    for _ in range(60):
        time.sleep(5)
        try:
            import httpx
            r = httpx.get(f"http://localhost:{port}/health", timeout=3.0)
            if r.status_code == 200:
                print(f"  Server ready on port {port}")
                return proc
        except Exception:
            pass

    proc.kill()
    raise RuntimeError(f"Server failed to start on port {port}")


def kill_server(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print(f"  Server PID={proc.pid} stopped")


def build_agent_configs(spec, phase):
    configs = []
    for agent_key in phase.agents:
        agent = spec.agents.get(agent_key)
        if not agent:
            continue
        configs.append({
            "agent_id": f"scn_{spec.id}_{agent_key}",
            "display_name": agent.display_name,
            "role": agent.role,
            "system_prompt": agent.system_prompt,
            "lifecycle": agent.lifecycle,
        })
    return configs


def collect_prior_messages(spec, phase_idx, phase_results):
    prior = {}
    current_phase = spec.phases[phase_idx]
    for agent_key in current_phase.agents:
        agent = spec.agents.get(agent_key)
        if not agent or agent.lifecycle != "permanent":
            continue
        agent_id = f"scn_{spec.id}_{agent_key}"
        agent_msgs = []
        for prior_idx in range(phase_idx):
            prior_phase = spec.phases[prior_idx]
            if agent_key not in prior_phase.agents:
                continue
            for msg in phase_results[prior_idx]["messages"]:
                sender_name = msg.get("sender_name", "")
                content = msg.get("content", "")
                if not content:
                    continue
                if sender_name == "System":
                    sender_id = "system"
                else:
                    sender_key = None
                    for k, a in spec.agents.items():
                        if a.display_name == sender_name:
                            sender_key = k
                            break
                    sender_id = f"scn_{spec.id}_{sender_key}" if sender_key else msg.get("sender_id", "system")
                agent_msgs.append({
                    "sender_id": sender_id,
                    "sender_name": sender_name,
                    "content": content,
                })
        if agent_msgs:
            prior[agent_id] = agent_msgs
    return prior


def run_and_dump(base_url: str, model_label: str):
    """Run prisoner's dilemma and print every message."""
    spec = load_scenario(SCENARIO_PATH)
    phase_results = []

    print(f"\n{'#'*78}")
    print(f"  {model_label}: Prisoner's Dilemma ({len(spec.phases)} phases)")
    print(f"{'#'*78}")

    total_start = time.time()

    for phase_idx, phase in enumerate(spec.phases):
        print(f"\n{'='*78}")
        print(f"  PHASE {phase_idx+1}/{len(spec.phases)}: {phase.label}")
        print(f"{'='*78}")

        agent_configs = build_agent_configs(spec, phase)
        prior = collect_prior_messages(spec, phase_idx, phase_results) if phase_idx > 0 else None

        result = api_client.create_session(
            base_url,
            topology=phase.topology,
            debate_format=phase.debate_format,
            decision_mode=phase.decision_mode,
            agents=agent_configs,
            initial_prompt=phase.initial_prompt,
            max_turns=100,
            persistent_cache_prefix=spec.id,
            prior_agent_messages=prior,
        )
        if not result:
            print("  FAILED to create session")
            return False

        session_id = result["session_id"]
        turn_count = phase.auto_rounds * len(phase.agents)

        phase_start = time.time()
        success = api_client.execute_turns(base_url, session_id, turn_count)
        if not success:
            print("  FAILED to execute turns")
            return False

        messages = api_client.get_session_messages(base_url, session_id)
        phase_time = time.time() - phase_start

        # Print EVERY message
        print(f"\n  [{len(messages)} messages, {phase_time:.1f}s]")
        print(f"  {'-'*74}")

        for i, msg in enumerate(messages):
            sender = msg.get("sender_name", "?")
            content = msg.get("content", "")
            print(f"\n  [{sender}]:")
            # Wrap long lines for readability
            for line in content.split("\n"):
                while len(line) > 72:
                    print(f"    {line[:72]}")
                    line = line[72:]
                print(f"    {line}")

        phase_results.append({
            "phase": phase.label,
            "session_id": session_id,
            "messages": messages,
            "time_seconds": phase_time,
        })

    total_time = time.time() - total_start

    # Semantic quality checks
    print(f"\n{'='*78}")
    print(f"  SEMANTIC QUALITY CHECKS — {model_label}")
    print(f"{'='*78}")

    issues = []

    # Check 1: Warden should reference payoff matrix / deal
    phase1_msgs = phase_results[0]["messages"]
    warden_msgs = [m for m in phase1_msgs if m.get("sender_name") == "Warden"]
    warden_text = " ".join(m.get("content", "") for m in warden_msgs).lower()
    if any(w in warden_text for w in ["year", "confess", "silent", "deal", "sentence", "prison"]):
        print("  [PASS] Warden references the deal/punishment")
    else:
        print("  [FAIL] Warden does NOT reference deal — off-topic!")
        issues.append("Warden off-topic in Phase 1")

    # Check 2: Marco should show anxiety / family concern
    marco_phase1 = [m for m in phase1_msgs if m.get("sender_name") == "Marco"]
    marco_text = " ".join(m.get("content", "") for m in marco_phase1).lower()
    if any(w in marco_text for w in ["family", "scared", "worry", "nervous", "afraid", "first", "anxious", "wife", "kid", "children"]):
        print("  [PASS] Marco shows anxiety/family concern")
    else:
        print(f"  [WARN] Marco may not show expected anxiety (check text)")

    # Check 3: Danny should be streetwise / self-interested
    phase2_msgs = phase_results[1]["messages"]
    danny_msgs = [m for m in phase2_msgs if m.get("sender_name") == "Danny"]
    danny_text = " ".join(m.get("content", "") for m in danny_msgs).lower()
    if any(w in danny_text for w in ["trust", "myself", "deal", "look out", "confess", "lawyer", "done this", "system", "talk"]):
        print("  [PASS] Danny shows street-smart self-interest")
    else:
        print(f"  [WARN] Danny may not show expected self-interest (check text)")

    # Check 4: Yard conversation — Marco and Danny should talk to each other
    yard_msgs = phase_results[2]["messages"]
    yard_senders = [m.get("sender_name") for m in yard_msgs if m.get("sender_name") != "System"]
    if "Marco" in yard_senders and "Danny" in yard_senders:
        print("  [PASS] Both Marco and Danny speak in the Yard")
    else:
        print(f"  [FAIL] Missing speaker in Yard: {yard_senders}")
        issues.append("Missing speaker in Yard")

    # Check 5: Final Reckoning — explicit choices mentioned
    final_msgs = phase_results[3]["messages"]
    final_text = " ".join(m.get("content", "") for m in final_msgs).lower()
    has_choice_words = any(w in final_text for w in ["confess", "silent", "quiet", "talk", "keep silent"])
    if has_choice_words:
        print("  [PASS] Final Reckoning contains explicit choices")
    else:
        print("  [FAIL] Final Reckoning has no choice language!")
        issues.append("No choice language in Final Reckoning")

    # Check 6: Analyst verdict — references the matrix
    verdict_msgs = phase_results[4]["messages"]
    analyst_msgs = [m for m in verdict_msgs if m.get("sender_name") == "Analyst"]
    if analyst_msgs:
        analyst_text = analyst_msgs[0].get("content", "")
        has_verdict = any(w in analyst_text.lower() for w in ["year", "free", "chose", "confess", "silent", "matrix", "gets"])
        if has_verdict:
            print("  [PASS] Analyst references payoff outcomes")
        else:
            print("  [FAIL] Analyst verdict doesn't reference outcomes!")
            issues.append("Analyst verdict lacks outcomes")
    else:
        print("  [FAIL] No Analyst message in Verdict phase!")
        issues.append("No Analyst in Verdict")

    # Check 7: No garbled/empty messages
    all_msgs = []
    for pr in phase_results:
        all_msgs.extend(pr["messages"])
    non_system = [m for m in all_msgs if m.get("sender_name") != "System"]
    empty = [m for m in non_system if not m.get("content", "").strip()]
    garbled = [m for m in non_system if m.get("content", "").strip() and len(m["content"].split()) < 3]
    if empty:
        print(f"  [FAIL] {len(empty)} empty messages!")
        issues.append(f"{len(empty)} empty messages")
    else:
        print(f"  [PASS] No empty messages ({len(non_system)} total agent messages)")
    if garbled:
        print(f"  [WARN] {len(garbled)} very short messages (<3 words)")
    else:
        print(f"  [PASS] No garbled messages")

    # Check 8: Role consistency — agents don't break character
    all_content = " ".join(m.get("content", "") for m in non_system)
    if "<|" in all_content or "```" in all_content[:500]:
        print("  [FAIL] Token artifacts leaked into output!")
        issues.append("Token artifacts in output")
    else:
        print("  [PASS] No token artifacts in output")

    print(f"\n  Total time: {total_time:.1f}s")
    if issues:
        print(f"  Issues: {', '.join(issues)}")
        return False
    else:
        print(f"  ALL SEMANTIC CHECKS PASSED")
        return True


def main():
    print("\n" + "=" * 78)
    print("  PRISONER'S DILEMMA — FULL SEMANTIC QUALITY REVIEW")
    print("  Every message printed. One model at a time. Clean caches.")
    print("=" * 78)

    models = [
        (None, "Gemma3", 8000),  # None = default model
        ("mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx", "DeepSeek", 8001),
    ]

    overall = {}

    for model_id, label, port in models:
        clear_caches()
        print(f"\n  Starting {label} server on port {port}...")

        proc = start_server(model_id, port)
        try:
            ok = run_and_dump(f"http://localhost:{port}", label)
            overall[label] = ok
        finally:
            kill_server(proc)
            time.sleep(3)  # Let GPU memory settle

    # Final summary
    print(f"\n{'='*78}")
    print(f"  FINAL SEMANTIC QUALITY SUMMARY")
    print(f"{'='*78}")
    for label, ok in overall.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
    print(f"{'='*78}\n")

    return 0 if all(overall.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
