#!/usr/bin/env python3
"""Test prisoner's dilemma scenario with Gemma 3 and DeepSeek.

Validates timing and output quality before running full benchmarks.
"""

import sys
import time
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from demo.lib import api_client
from semantic.adapters.config.scenario_loader import load_scenario


def build_agent_configs(spec, phase):
    """Build agent config dicts with stable IDs for session creation."""
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
    """Collect prior phase messages for each permanent agent."""
    prior = {}
    current_phase = spec.phases[phase_idx]

    for agent_key in current_phase.agents:
        agent = spec.agents.get(agent_key)
        if not agent or agent.lifecycle != "permanent":
            continue

        agent_id = f"scn_{spec.id}_{agent_key}"
        agent_msgs = []

        # Collect messages from all prior phases where this agent participated
        for prior_phase_idx in range(phase_idx):
            prior_phase = spec.phases[prior_phase_idx]
            if agent_key not in prior_phase.agents:
                continue

            # Get messages from this prior phase
            prior_messages = phase_results[prior_phase_idx]["messages"]
            for msg in prior_messages:
                sender_name = msg.get("sender_name", "")
                content = msg.get("content", "")
                if not content:
                    continue

                # Determine sender_id
                if sender_name == "System":
                    sender_id = "system"
                else:
                    # Find agent key by display name
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


def run_scenario(base_url, scenario_path, model_name):
    """Run full prisoner's dilemma scenario and measure timing."""
    print(f"\n{'='*80}")
    print(f"Testing Prisoner's Dilemma with {model_name}")
    print(f"{'='*80}\n")

    # Check server connection
    stats = api_client.get_agent_stats(base_url)
    if stats is None:
        print(f"❌ ERROR: Server not reachable at {base_url}")
        print(f"   Start with: semantic serve (for Gemma 3)")
        print(f"   Or: semantic serve --model mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx --port 8001")
        return False

    print(f"✓ Server connected at {base_url}")

    # Load scenario
    spec = load_scenario(scenario_path)
    print(f"✓ Scenario loaded: {spec.title}")
    print(f"  Agents: {', '.join(a.display_name for a in spec.agents.values())}")
    print(f"  Phases: {len(spec.phases)}\n")

    # Track results for each phase
    phase_results = []
    total_start = time.time()

    # Run each phase sequentially
    for phase_idx, phase in enumerate(spec.phases):
        print(f"\n{'─'*80}")
        print(f"Phase {phase_idx + 1}/{len(spec.phases)}: {phase.label}")
        print(f"{'─'*80}")

        # Build agent configs
        agent_configs = build_agent_configs(spec, phase)

        # Collect prior messages for permanent agents
        prior_agent_messages = None
        if phase_idx > 0:
            prior_agent_messages = collect_prior_messages(spec, phase_idx, phase_results)

        # Create session
        phase_start = time.time()
        result = api_client.create_session(
            base_url,
            topology=phase.topology,
            debate_format=phase.debate_format,
            decision_mode=phase.decision_mode,
            agents=agent_configs,
            initial_prompt=phase.initial_prompt,
            max_turns=100,  # High limit, will stop after auto_rounds
            persistent_cache_prefix=spec.id,
            prior_agent_messages=prior_agent_messages,
        )

        if not result:
            print(f"❌ Failed to create session for phase: {phase.label}")
            return False

        session_id = result.get("session_id")
        print(f"  Session ID: {session_id}")

        # Execute turns (auto_rounds * num_agents)
        turn_count = phase.auto_rounds * len(phase.agents)
        print(f"  Executing {turn_count} turns ({phase.auto_rounds} rounds × {len(phase.agents)} agents)...")

        turn_start = time.time()
        success = api_client.execute_turns(base_url, session_id, turn_count)
        turn_time = time.time() - turn_start

        if not success:
            print(f"❌ Failed to execute turns for phase: {phase.label}")
            return False

        # Fetch messages
        messages = api_client.get_session_messages(base_url, session_id)
        phase_time = time.time() - phase_start

        print(f"  ✓ Completed in {phase_time:.1f}s (turns: {turn_time:.1f}s)")
        print(f"  Messages: {len(messages)}")

        # Show last 3 messages for quality check
        print(f"\n  Last 3 messages:")
        for msg in messages[-3:]:
            sender = msg.get("sender_name", "Unknown")
            content = msg.get("content", "")
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"    • {sender}: {preview}")

        # Store results
        phase_results.append({
            "phase": phase.label,
            "session_id": session_id,
            "messages": messages,
            "time_seconds": phase_time,
            "turn_time_seconds": turn_time,
        })

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*80}")
    print(f"Scenario Complete: {model_name}")
    print(f"{'='*80}")
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"\nPhase timings:")
    for r in phase_results:
        print(f"  {r['phase']:30s}: {r['time_seconds']:6.1f}s (turns: {r['turn_time_seconds']:5.1f}s)")

    # Sanity checks
    print(f"\n{'─'*80}")
    print("Sanity Checks:")
    print(f"{'─'*80}")

    all_ok = True

    # Check 1: All phases completed
    if len(phase_results) == len(spec.phases):
        print(f"✓ All {len(spec.phases)} phases completed")
    else:
        print(f"❌ Only {len(phase_results)}/{len(spec.phases)} phases completed")
        all_ok = False

    # Check 2: Each phase has messages
    for r in phase_results:
        if len(r["messages"]) > 0:
            print(f"✓ {r['phase']:30s}: {len(r['messages'])} messages")
        else:
            print(f"❌ {r['phase']:30s}: No messages!")
            all_ok = False

    # Check 3: Timing is reasonable (not too fast = error, not too slow = hang)
    for r in phase_results:
        time_s = r["time_seconds"]
        if 1.0 < time_s < 120.0:  # Between 1s and 2min per phase
            print(f"✓ {r['phase']:30s}: {time_s:.1f}s (reasonable)")
        elif time_s <= 1.0:
            print(f"⚠️  {r['phase']:30s}: {time_s:.1f}s (suspiciously fast, check for errors)")
            all_ok = False
        else:
            print(f"⚠️  {r['phase']:30s}: {time_s:.1f}s (very slow, check for hangs)")

    # Check 4: Final phase should have analyst output
    final_phase = phase_results[-1]
    if final_phase["phase"] == "The Verdict":
        analyst_msg = None
        for msg in final_phase["messages"]:
            if msg.get("sender_name") == "Analyst":
                analyst_msg = msg.get("content", "")
                break

        if analyst_msg:
            print(f"✓ Analyst provided verdict")
            print(f"  Verdict: {analyst_msg[:200]}...")
        else:
            print(f"⚠️  No analyst verdict found")

    print(f"\n{'='*80}")
    if all_ok:
        print(f"✅ ALL CHECKS PASSED - {model_name} is ready for benchmarks")
    else:
        print(f"⚠️  SOME CHECKS FAILED - Review output before running benchmarks")
    print(f"{'='*80}\n")

    return all_ok


def main():
    """Run prisoner's dilemma test for both models."""
    scenario_path = Path(__file__).parent / "demo" / "scenarios" / "prisoners_dilemma.yaml"

    print("\n" + "="*80)
    print("Prisoner's Dilemma Pre-Benchmark Validation")
    print("="*80)
    print("\nThis test validates timing and output quality before running full benchmarks.")
    print("Both models will run the complete 5-phase scenario.")
    print("")

    # Test Gemma 3 (default, port 8000)
    print("\n1️⃣  Testing Gemma 3 (port 8000)...")
    gemma_ok = run_scenario("http://localhost:8000", scenario_path, "Gemma 3")

    # Test DeepSeek (port 8001)
    print("\n2️⃣  Testing DeepSeek (port 8001)...")
    deepseek_ok = run_scenario("http://localhost:8001", scenario_path, "DeepSeek-V2")

    # Final summary
    print("\n" + "="*80)
    print("FINAL VALIDATION SUMMARY")
    print("="*80)

    if gemma_ok:
        print("✅ Gemma 3: PASSED - Ready for benchmarks")
    else:
        print("❌ Gemma 3: FAILED - Fix issues before benchmarking")

    if deepseek_ok:
        print("✅ DeepSeek: PASSED - Ready for benchmarks")
    else:
        print("❌ DeepSeek: FAILED - Fix issues before benchmarking")

    print("\nIf both passed, proceed with:")
    print("  1. Clean environment: rm -f ~/.semantic/caches/*.safetensors")
    print("  2. Run Gemma 3 benchmarks")
    print("  3. Run DeepSeek benchmarks")
    print("="*80 + "\n")

    return 0 if (gemma_ok and deepseek_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
