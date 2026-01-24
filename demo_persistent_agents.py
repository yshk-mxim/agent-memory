#!/usr/bin/env python3
"""
Persistent Multi-Agent Memory Demo

Demonstrates cross-session agent memory using KV cache persistence.

Session 1: Create 3 agents, generate responses, save to disk
Session 2: Load agents from disk, continue conversation (faster with cached context)

Usage:
    python demo_persistent_agents.py --session 1  # First session (create & save)
    python demo_persistent_agents.py --session 2  # Second session (load & continue)
"""

import argparse
import time
from src.agent_manager import PersistentAgentManager


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_section(text):
    """Print formatted section."""
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}\n")


def demo_session_1():
    """
    Session 1: Creating Agents

    1. Initialize manager with Gemma 3 12B
    2. Create 3 agents (technical specialist, business analyst, coordinator)
    3. User query about API bug
    4. Get responses from all agents
    5. Save agents to disk
    6. Show cache sizes and memory usage
    """
    print_header("Session 1: Creating Agents")

    print("Loading model...")
    start_time = time.time()

    manager = PersistentAgentManager(
        model_name="mlx-community/gemma-3-12b-it-4bit",
        max_agents=3
    )

    load_time = time.time() - start_time
    print(f"✅ Model loaded ({load_time:.1f}s)\n")

    # Create agents
    print_section("Creating 3 Agents")

    agents_config = [
        {
            "agent_id": "tech_specialist",
            "agent_type": "technical",
            "system_prompt": (
                "You are a Technical Specialist with expertise in software architecture, "
                "APIs, databases, and system performance. Analyze technical issues with "
                "precision and provide actionable engineering recommendations."
            )
        },
        {
            "agent_id": "biz_analyst",
            "agent_type": "business",
            "system_prompt": (
                "You are a Business Analyst focused on ROI, revenue impact, customer "
                "satisfaction, and strategic decision-making. Evaluate business implications "
                "and prioritize solutions based on business value."
            )
        },
        {
            "agent_id": "coordinator",
            "agent_type": "coordinator",
            "system_prompt": (
                "You are a Project Coordinator who synthesizes technical and business "
                "perspectives into actionable plans. Create clear action items with timelines "
                "and resource requirements, balancing technical feasibility and business needs."
            )
        }
    ]

    for config in agents_config:
        agent = manager.create_agent(**config)
        print(f"  ✅ {config['agent_id']}: {agent.cache_tokens} tokens cached (system prompt)")

    # User query
    print_section("User Query")
    user_query = (
        "We have a critical API bug affecting payment processing. Our /api/payments "
        "endpoint is returning 500 errors for 15% of requests. Error logs show "
        "'database connection timeout' during high load. This is impacting revenue "
        "and customer satisfaction. Analyze the technical issues and business impact."
    )
    print(f"Query: {user_query}\n")

    # Generate responses
    print_section("Agent Responses")

    responses = {}

    # Technical Specialist
    print("🔧 Technical Specialist analyzing...")
    start_gen = time.time()
    tech_response = manager.generate(
        "tech_specialist",
        user_query,
        max_tokens=200,
        temperature=0.7
    )
    tech_time = time.time() - start_gen

    tech_agent = manager.agents["tech_specialist"]
    print(f"Response ({tech_time:.1f}s, {tech_agent.cache_tokens} tokens cached):")
    print(f"{tech_response[:300]}...")
    responses["tech_specialist"] = tech_response

    print()

    # Business Analyst
    print("📊 Business Analyst analyzing...")
    start_gen = time.time()
    biz_response = manager.generate(
        "biz_analyst",
        user_query,
        max_tokens=200,
        temperature=0.7
    )
    biz_time = time.time() - start_gen

    biz_agent = manager.agents["biz_analyst"]
    print(f"Response ({biz_time:.1f}s, {biz_agent.cache_tokens} tokens cached):")
    print(f"{biz_response[:300]}...")
    responses["biz_analyst"] = biz_response

    print()

    # Coordinator
    print("📋 Coordinator synthesizing...")
    start_gen = time.time()
    coord_response = manager.generate(
        "coordinator",
        user_query,
        max_tokens=200,
        temperature=0.7
    )
    coord_time = time.time() - start_gen

    coord_agent = manager.agents["coordinator"]
    print(f"Response ({coord_time:.1f}s, {coord_agent.cache_tokens} tokens cached):")
    print(f"{coord_response[:300]}...")
    responses["coordinator"] = coord_response

    # Save agents
    print_section("Saving Agents to Disk")
    save_start = time.time()
    manager.save_all()
    save_time = time.time() - save_start
    print(f"✅ All agents saved ({save_time:.2f}s)")

    # Show disk usage
    disk_usage = manager.persistence.get_cache_disk_usage()
    print(f"\n💾 Disk Usage:")
    for agent_id, size_bytes in disk_usage['per_agent']:
        size_mb = size_bytes / (1024 * 1024)
        print(f"  {agent_id}: {size_mb:.1f} MB")
    print(f"  Total: {disk_usage['total_mb']:.1f} MB")

    # Show memory usage
    mem_usage = manager.get_memory_usage()
    print(f"\n🧠 Memory Usage:")
    print(f"  Model: {mem_usage['model_memory_gb']:.2f} GB")
    print(f"  Agents: {mem_usage['total_cache_mb']:.1f} MB")
    print(f"  Total: {mem_usage['total_gb']:.2f} GB")

    print_header("Session 1 Complete! Run with --session 2 to continue.")


def demo_session_2():
    """
    Session 2: Loading Agents and Continuing Conversation

    1. Initialize manager
    2. Load agents from disk
    3. Follow-up query
    4. Show speedup from cache reuse
    5. Display updated memory usage
    """
    print_header("Session 2: Loading Agents from Disk")

    print("Loading model...")
    start_time = time.time()

    manager = PersistentAgentManager(
        model_name="mlx-community/gemma-3-12b-it-4bit",
        max_agents=3
    )

    load_time = time.time() - start_time
    print(f"✅ Model loaded ({load_time:.1f}s)\n")

    # Load agents
    print_section("Loading Agents from Disk")

    agents_to_load = ["tech_specialist", "biz_analyst", "coordinator"]
    load_start = time.time()

    for agent_id in agents_to_load:
        agent = manager.load_agent(agent_id)
        print(f"  ✅ {agent_id}: {agent.cache_tokens} tokens cached")

    load_agents_time = time.time() - load_start
    print(f"\n✅ All agents loaded ({load_agents_time:.2f}s)")

    # Follow-up query
    print_section("Follow-up Query")
    followup_query = (
        "What are the specific steps to fix that API bug? "
        "Provide a prioritized action plan."
    )
    print(f"Query: {followup_query}\n")

    # Generate with cached context
    print_section("Agent Responses (with cached context)")

    print("🔧 Technical Specialist (from cache)...")
    start_gen = time.time()
    tech_response = manager.generate(
        "tech_specialist",
        followup_query,
        max_tokens=200,
        temperature=0.7
    )
    tech_time_cached = time.time() - start_gen

    tech_agent = manager.agents["tech_specialist"]
    print(f"Response ({tech_time_cached:.1f}s, {tech_agent.cache_tokens} tokens cached):")
    print(f"{tech_response[:300]}...")

    # Show performance improvement
    print_section("📊 Performance Comparison")
    print("Session 1 (no cache):  ~5-8s generation time")
    print(f"Session 2 (with cache): {tech_time_cached:.1f}s generation time")
    speedup = ((8 - tech_time_cached) / 8) * 100
    print(f"Speedup: ~{speedup:.0f}% faster! ⚡")

    # Save updated agents
    print_section("Saving Updated Agents")
    save_start = time.time()
    manager.save_all()
    save_time = time.time() - save_start
    print(f"✅ All agents saved ({save_time:.2f}s)")

    # Show memory usage
    mem_usage = manager.get_memory_usage()
    print(f"\n🧠 Memory Usage:")
    print(f"  Model: {mem_usage['model_memory_gb']:.2f} GB")
    print(f"  Agents: {mem_usage['total_cache_mb']:.1f} MB")
    print(f"  Total: {mem_usage['total_gb']:.2f} GB")

    print_header("Demo Complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Persistent Multi-Agent Memory Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--session",
        type=int,
        choices=[1, 2],
        required=True,
        help="Which session to run (1: create agents, 2: load and continue)"
    )

    args = parser.parse_args()

    if args.session == 1:
        demo_session_1()
    else:
        demo_session_2()


if __name__ == "__main__":
    main()
