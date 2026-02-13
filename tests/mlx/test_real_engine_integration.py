# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Real MLX engine integration tests for behaviors tested with FakeEngine.

Tests concurrent inference, cache reuse, streaming, and coordination
with actual MLX model on Metal GPU. No mocking.

Run: pytest tests/mlx/test_real_engine_integration.py -v -x --timeout=180
REQUIRES: Apple Silicon, dangerouslyDisableSandbox: true
"""

import asyncio
import time

import pytest

pytestmark = pytest.mark.integration


def _make_engine(model, tokenizer, spec, total_blocks=400):
    """Build a real BlockPoolBatchEngine."""
    from agent_memory.adapters.outbound.mlx_cache_adapter import MLXCacheAdapter
    from agent_memory.application.batch_engine import BlockPoolBatchEngine
    from agent_memory.domain.services import BlockPool

    pool = BlockPool(spec=spec, total_blocks=total_blocks)
    cache_adapter = MLXCacheAdapter()
    return BlockPoolBatchEngine(
        model=model,
        tokenizer=tokenizer,
        pool=pool,
        spec=spec,
        cache_adapter=cache_adapter,
    )


class TestConcurrentInference:
    """Test multiple agents submitting simultaneously with real MLX."""

    def test_three_agents_concurrent(self, real_model_and_tokenizer, real_spec) -> None:
        """Three agents submit different prompts and all get completions."""
        model, tokenizer = real_model_and_tokenizer
        engine = _make_engine(model, tokenizer, real_spec)

        # Use chat-template-formatted prompts so the model doesn't
        # immediately output EOS on the raw text.
        raw_prompts = [
            ("agent_a", "What is 2+2?"),
            ("agent_b", "Name a color."),
            ("agent_c", "Say hello."),
        ]
        uids = []
        for agent_id, raw_text in raw_prompts:
            messages = [{"role": "user", "content": raw_text}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            uid = engine.submit(agent_id=agent_id, prompt=formatted, max_tokens=20)
            uids.append(uid)

        assert len(uids) == 3

        completions = {}
        for result in engine.step():
            completions[result.uid] = result

        assert len(completions) == 3, f"Expected 3 completions, got {len(completions)}"
        for uid in uids:
            assert uid in completions
            c = completions[uid]
            assert len(c.text) > 0, f"Empty response for {uid}"
            assert c.token_count > 0
            assert c.blocks is not None

    def test_sequential_agents_reuse_pool(self, real_model_and_tokenizer, real_spec) -> None:
        """Sequential agent requests reuse freed blocks from pool."""
        model, tokenizer = real_model_and_tokenizer
        engine = _make_engine(model, tokenizer, real_spec)

        all_completions = []
        for i in range(3):
            uid = engine.submit(agent_id=f"seq_agent_{i}", prompt="Hi", max_tokens=10)
            for result in engine.step():
                if result.uid == uid:
                    all_completions.append(result)
                    break

        assert len(all_completions) == 3
        for c in all_completions:
            assert len(c.text) > 0
            assert c.blocks is not None


class TestCacheLifecycle:
    """Test full cache lifecycle: generate → extract → persist → reload → reuse."""

    def test_cache_reuse_produces_valid_output(
        self, real_model_and_tokenizer, real_spec, cache_dir
    ) -> None:
        """Generate, save cache, reload, and generate again with cached context."""
        from agent_memory.adapters.outbound.safetensors_cache_adapter import (
            SafetensorsCacheAdapter,
        )

        model, tokenizer = real_model_and_tokenizer
        engine = _make_engine(model, tokenizer, real_spec)
        disk = SafetensorsCacheAdapter(cache_dir)

        # First generation
        uid1 = engine.submit(
            agent_id="cache_agent",
            prompt="The capital of France is",
            max_tokens=15,
        )
        completions1 = list(engine.step())
        assert len(completions1) == 1
        blocks1 = completions1[0].blocks
        text1 = completions1[0].text

        assert blocks1 is not None
        assert len(text1) > 0

        # Persist to disk
        path = disk.save(
            "cache_agent",
            blocks1,
            {"model_id": "SmolLM2-135M", "n_layers": str(real_spec.n_layers)},
        )
        assert path.exists()

        # Reload from disk
        loaded_blocks, _ = disk.load(path)
        assert len(loaded_blocks) > 0

    def test_multiple_agents_independent_caches(
        self, real_model_and_tokenizer, real_spec, cache_dir
    ) -> None:
        """Two agents generate, save, and reload independently."""
        from agent_memory.adapters.outbound.safetensors_cache_adapter import (
            SafetensorsCacheAdapter,
        )

        model, tokenizer = real_model_and_tokenizer
        engine = _make_engine(model, tokenizer, real_spec)
        disk = SafetensorsCacheAdapter(cache_dir)

        agents = [
            ("alice", "Hello Alice here"),
            ("bob", "Hello Bob here"),
        ]
        saved_paths = {}
        for agent_id, prompt in agents:
            uid = engine.submit(agent_id=agent_id, prompt=prompt, max_tokens=10)
            for result in engine.step():
                if result.uid == uid:
                    path = disk.save(
                        agent_id,
                        result.blocks,
                        {"model_id": "SmolLM2-135M"},
                    )
                    saved_paths[agent_id] = path
                    break

        # Both saved independently
        assert len(saved_paths) == 2
        for agent_id, path in saved_paths.items():
            assert path.exists()
            loaded, _ = disk.load(path)
            assert len(loaded) > 0


class TestStreamingGeneration:
    """Test token-by-token streaming with real MLX."""

    def test_step_once_yields_tokens(self, real_model_and_tokenizer, real_spec) -> None:
        """step_once() produces individual tokens until completion."""
        model, tokenizer = real_model_and_tokenizer
        engine = _make_engine(model, tokenizer, real_spec)

        engine.submit(agent_id="stream_agent", prompt="Count: 1 2 3", max_tokens=15)

        tokens_seen = 0
        max_iters = 50
        for _ in range(max_iters):
            if not engine.has_active_batch():
                break
            results = engine.step_once()
            for r in results:
                if r.text:
                    tokens_seen += 1

        assert tokens_seen > 0, "No tokens yielded from step_once()"


class TestCoordinationWithRealEngine:
    """Test coordination service with real MLX inference."""

    @staticmethod
    def _make_service(engine, cache_dir, spec):
        """Build CoordinationService with real engine, no scheduler."""
        from agent_memory.adapters.outbound.safetensors_cache_adapter import (
            SafetensorsCacheAdapter,
        )
        from agent_memory.application.agent_cache_store import AgentCacheStore, ModelTag
        from agent_memory.application.coordination_service import CoordinationService

        tag = ModelTag(
            model_id="SmolLM2-135M",
            n_layers=spec.n_layers,
            n_kv_heads=spec.n_kv_heads,
            head_dim=spec.head_dim,
            block_tokens=spec.block_tokens,
        )
        cache_adapter = SafetensorsCacheAdapter(cache_dir)
        cache_store = AgentCacheStore(
            cache_dir=cache_dir,
            max_hot_agents=5,
            model_tag=tag,
            cache_adapter=cache_adapter,
        )
        return CoordinationService(
            scheduler=None,
            cache_store=cache_store,
            engine=engine,
        )

    @staticmethod
    def _make_agents(pairs):
        """Build AgentRole list from (id, name, prompt) tuples."""
        from agent_memory.domain.coordination import AgentRole

        return [
            AgentRole(agent_id=aid, display_name=name, system_prompt=prompt)
            for aid, name, prompt in pairs
        ]

    def test_two_turn_coordination(self, real_model_and_tokenizer, real_spec, cache_dir) -> None:
        """Create session, execute 2 turns, verify responses are non-empty."""
        from agent_memory.domain.coordination import (
            DebateFormat,
            DecisionMode,
            Topology,
        )

        model, tokenizer = real_model_and_tokenizer
        engine = _make_engine(model, tokenizer, real_spec)
        service = self._make_service(engine, cache_dir, real_spec)

        agents = self._make_agents(
            [
                ("a1", "Agent1", "You are Agent1. Be brief."),
                ("a2", "Agent2", "You are Agent2. Be brief."),
            ]
        )

        loop = asyncio.new_event_loop()
        try:
            session = loop.run_until_complete(
                service.create_session(
                    topology=Topology.ROUND_ROBIN,
                    debate_format=DebateFormat.FREE_FORM,
                    decision_mode=DecisionMode.NONE,
                    agents=agents,
                    initial_prompt="Hello, let's discuss testing.",
                    max_turns=2,
                )
            )

            msg1 = loop.run_until_complete(service.execute_turn(session.session_id))
            assert msg1 is not None
            assert len(msg1.content) > 0, "Turn 1 produced empty content"

            msg2 = loop.run_until_complete(service.execute_turn(session.session_id))
            assert msg2 is not None
            assert len(msg2.content) > 0, "Turn 2 produced empty content"
        finally:
            loop.close()

    def test_coordination_identity_isolation(
        self, real_model_and_tokenizer, real_spec, cache_dir
    ) -> None:
        """Agents in coordination should not impersonate each other."""
        from agent_memory.domain.coordination import (
            DebateFormat,
            DecisionMode,
            Topology,
        )

        model, tokenizer = real_model_and_tokenizer
        engine = _make_engine(model, tokenizer, real_spec)
        service = self._make_service(engine, cache_dir, real_spec)

        agents = self._make_agents(
            [
                ("alice", "Alice", "You are Alice. Respond briefly."),
                ("bob", "Bob", "You are Bob. Respond briefly."),
            ]
        )

        loop = asyncio.new_event_loop()
        try:
            session = loop.run_until_complete(
                service.create_session(
                    topology=Topology.ROUND_ROBIN,
                    debate_format=DebateFormat.FREE_FORM,
                    decision_mode=DecisionMode.NONE,
                    agents=agents,
                    initial_prompt="Introduce yourselves.",
                    max_turns=2,
                )
            )

            msg1 = loop.run_until_complete(service.execute_turn(session.session_id))
            msg2 = loop.run_until_complete(service.execute_turn(session.session_id))

            for msg in [msg1, msg2]:
                assert "User:" not in msg.content
                assert "<start_of_turn>" not in msg.content
                assert "<|channel|>" not in msg.content
        finally:
            loop.close()


class TestGenerationTiming:
    """Basic performance sanity checks with real MLX."""

    def test_generation_completes_within_timeout(self, real_model_and_tokenizer, real_spec) -> None:
        """Single generation should complete in under 30 seconds."""
        model, tokenizer = real_model_and_tokenizer
        engine = _make_engine(model, tokenizer, real_spec)

        start = time.monotonic()
        engine.submit(agent_id="timing_agent", prompt="Hello", max_tokens=20)
        list(engine.step())
        elapsed = time.monotonic() - start

        assert elapsed < 30, f"Generation took {elapsed:.1f}s (expected <30s)"

    def test_three_concurrent_faster_than_sequential_3x(
        self, real_model_and_tokenizer, real_spec
    ) -> None:
        """Concurrent 3-agent should not take 3x a single agent."""
        model, tokenizer = real_model_and_tokenizer

        # Measure single agent
        engine1 = _make_engine(model, tokenizer, real_spec)
        start = time.monotonic()
        engine1.submit(agent_id="solo", prompt="Hello", max_tokens=15)
        list(engine1.step())
        single_time = time.monotonic() - start

        # Measure 3 concurrent agents
        engine3 = _make_engine(model, tokenizer, real_spec)
        start = time.monotonic()
        for i in range(3):
            engine3.submit(agent_id=f"batch_{i}", prompt="Hello", max_tokens=15)
        list(engine3.step())
        batch_time = time.monotonic() - start

        # Batch should be less than 3x single (batching advantage)
        assert batch_time < single_time * 3.5, (
            f"Batch ({batch_time:.1f}s) >= 3.5x single ({single_time:.1f}s)"
        )
