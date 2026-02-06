# Config-Driven Multi-Agent Coordination with Persistent KV Cache on Edge

**Research Contribution**: Declarative YAML-driven multi-agent scenario specification with cross-phase context injection, integrated with on-device persistent KV cache.

**Date**: February 2, 2026

---

## Problem

Multi-agent demos on local LLM infrastructure require extensive hardcoded UI and prompt logic. Adding a new scenario (e.g., debate, collaborative writing, role-play) means writing custom frontend components, prompt assembly code, and agent lifecycle management from scratch. There is no standard way to specify:

- **Agent topologies**: which agents participate in which phases of a conversation.
- **Cross-session context injection**: how knowledge from one phase flows into the next without server-side long-term memory.
- **Phase-based conversations**: ordered stages where the set of active agents, prompt context, and interaction rules change between phases.

The result is that every new multi-agent scenario is a full-stack development effort rather than a configuration change.

---

## Solution

A YAML-driven scenario specification that cleanly separates four concerns:

1. **Agent definitions** -- identity, role, system prompt, and memory lifecycle for each agent, declared once and reused across phases.
2. **Interaction topology** -- per-phase agent selection and ordered phase sequencing. A phase declares which agents are active and what context they receive.
3. **Prompt construction** -- a cross-phase template system using `${phase.messages[agent]}` patterns. Templates compose prior conversation text into new prompts without server-side state.
4. **UI layout** -- columns, tabs, and controls derived from the scenario definition. The renderer reads the YAML and builds the interface; no custom UI code per scenario.

Adding a new multi-agent scenario reduces to writing a YAML file.

---

## Novel Aspects

### 1. Cross-Phase Context Injection via Template System

Agents carry knowledge between phases without server-side long-term memory. The client composes prompts using `${phase.messages[agent]}` references that inline prior conversation text into the system prompt for the next phase. This is a pure client-side composition pattern: no vector database, no retrieval step, no server-side memory store. The template system handles context flow declaratively.

### 2. Scenario-Driven Prompt Construction

Agent identity is maintained across topology changes. The same agent can participate in Phase 1 (e.g., brainstorming with Agent B) and Phase 3 (e.g., critique with Agent C) with different context windows but a consistent persona. The scenario YAML controls which slices of conversation history each agent sees per phase, preventing context pollution while preserving continuity.

### 3. Integration with Persistent KV Cache

Each phase creates session-scoped caches keyed by agent and phase. When cross-phase templates inject prior conversation text into a new prompt, the resulting prompt shares a long common prefix with the previous phase's prompt (system prompt + accumulated history). The cache EXTEND path (character-level prefix matching) detects this shared prefix and reuses cached KV entries, so only the newly injected text triggers compute. This turns cross-phase context injection from an expensive re-prefill into an incremental cache extension.

---

## Architecture

The implementation follows hexagonal architecture:

- **Domain**: Pure Python dataclasses defining `ScenarioSpec`, `AgentSpec`, `PhaseSpec`, and `TemplateRef`. No framework dependencies.
- **Adapters**: YAML loading with Pydantic validation converts raw YAML into domain objects. Validation enforces that template references point to valid phases/agents and that phase ordering is acyclic.
- **Demo layer**: `ScenarioRenderer` reads the validated domain objects and builds the Gradio UI. The renderer is generic; it does not contain scenario-specific logic.

Dependencies point inward: Demo -> Adapters -> Domain. The domain layer has zero imports from Pydantic, Gradio, or YAML libraries.

---

## Comparison to Existing Frameworks

| Framework | Agent Config | Topology | Prompt Templates | YAML-Driven | KV Cache Integration |
|-----------|-------------|----------|-----------------|-------------|---------------------|
| AutoGen | Code-defined | Code-defined | Manual | No | No |
| CrewAI | Code-defined | Code-defined | Manual | No | No |
| LangGraph | Code-defined | Graph DSL | Manual | No | No |
| **This work** | **YAML** | **YAML phases** | **Cross-phase templates** | **Yes** | **Yes** |

Key differentiators:

- **AutoGen / CrewAI**: Agent definitions and interaction patterns are specified in Python code. Changing topology requires code changes and redeployment.
- **LangGraph**: Offers a graph DSL for agent coordination but does not provide YAML-driven topology specification or a template system for cross-phase prompt construction.
- **None** of the above integrate with on-device KV cache persistence. Cross-phase context injection in existing frameworks relies on server-side memory or vector stores, not cache-aware prompt composition.

---

## Relation to Existing Novelty

This work builds on two previously documented techniques:

- **Character-level prefix matching** (`character_level_prefix_matching.md`): The EXTEND path compares raw text rather than tokens, so cross-phase prompts that share a long conversational prefix get efficient cache reuse despite BPE boundary shifts.
- **Continuous batching** (`continuous_batching.md`): Multiple agents within a phase can be served concurrently. Each agent's session-scoped cache is managed independently by the batch engine.

The coordination layer generates prompts that are structurally favorable for cache reuse: long shared prefixes (system prompt + prior phases) with short unique suffixes (new phase instructions). This alignment between prompt construction and cache architecture is intentional and represents a co-design between the scenario specification and the inference engine.
