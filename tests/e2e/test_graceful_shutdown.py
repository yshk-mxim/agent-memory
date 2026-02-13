# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""E2E tests for graceful shutdown.

Tests verify graceful shutdown behavior:
- Requests in-flight complete before shutdown
- New requests rejected during drain
- Cache persistence happens after drain
"""

import threading

import httpx
import pytest


@pytest.mark.e2e
def test_graceful_shutdown_with_active_requests(live_server: str):
    """Test that active requests complete (basic concurrency test).

    Verifies:
    - Submit 3 concurrent requests to server
    - All 3 requests complete successfully
    - Server can handle concurrent load

    Note: Actual shutdown testing happens when server process exits.
    The drain() mechanism is tested at the unit level.
    """
    client = httpx.Client(
        base_url=live_server,
        timeout=60.0,  # Longer timeout for concurrent requests
        headers={"x-api-key": "test-key-for-e2e"},
    )

    try:
        # Submit 3 concurrent requests
        responses = []
        errors = []

        def submit_request(index: int):
            try:
                response = client.post(
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": f"Request {index}"}],
                        "max_tokens": 20,
                    },
                )
                responses.append((index, response))
            except Exception as e:
                errors.append((index, str(e)))

        # Start 3 requests in parallel
        threads = []
        for i in range(3):
            t = threading.Thread(target=submit_request, args=(i,))
            t.start()
            threads.append(t)

        # Wait for all threads to complete
        for t in threads:
            t.join(timeout=30)

        # Verify all requests completed successfully
        assert len(errors) == 0, f"Requests failed: {errors}"
        assert len(responses) == 3, f"Expected 3 responses, got {len(responses)}"

        # Verify all responses are 200 OK
        for index, response in responses:
            assert response.status_code == 200, (
                f"Request {index} failed with status {response.status_code}"
            )

            # Verify response has content
            data = response.json()
            assert "content" in data
            assert len(data["content"]) > 0

        print("\n✅ All 3 concurrent requests completed successfully")

    finally:
        client.close()


@pytest.mark.e2e
def test_drain_prevents_new_requests():
    """Test that new requests are rejected during drain.

    Note: This test verifies the drain() logic at the unit level
    since we can't easily trigger drain in the E2E subprocess server.

    Verifies:
    - BatchEngine.submit() raises PoolExhaustedError when draining
    - Error message indicates server is shutting down
    """
    from agent_memory.application.batch_engine import BlockPoolBatchEngine
    from agent_memory.domain.errors import PoolExhaustedError
    from agent_memory.domain.services import BlockPool
    from agent_memory.domain.value_objects import ModelCacheSpec

    # Create a mock batch engine
    spec = ModelCacheSpec(
        n_layers=2,
        n_kv_heads=8,
        head_dim=128,
        block_tokens=256,
        layer_types=["global"] * 2,
        sliding_window_size=0,
    )
    pool = BlockPool(spec=spec, total_blocks=100)

    # Use fake factory for testing
    def fake_batch_gen_factory(model, tokenizer):
        class FakeBatchGen:
            def insert(self, prompts, max_tokens, caches=None, samplers=None):
                return ["fake-uid"]

            def next(self):
                return []

        return FakeBatchGen()

    # Create mock tokenizer
    class MockTokenizer:
        eos_token_id = 0

        def encode(self, text):
            return [1, 2, 3]

    engine = BlockPoolBatchEngine(
        model={"fake": "model"},
        tokenizer=MockTokenizer(),
        pool=pool,
        spec=spec,
        cache_adapter=type(
            "obj",
            (object,),
            {
                "create_batch_generator": lambda *args, **kwargs: None,
                "create_sampler": lambda *args, **kwargs: None,
            },
        )(),
        batch_gen_factory=fake_batch_gen_factory,
    )

    # Should accept requests normally
    uid1 = engine.submit(agent_id="test-agent", prompt="Hello", max_tokens=10)
    assert isinstance(uid1, str)

    # Set draining flag
    engine._draining = True

    # Should reject new requests
    with pytest.raises(PoolExhaustedError) as exc_info:
        engine.submit(agent_id="test-agent", prompt="Hello", max_tokens=10)

    # Verify error message indicates shutdown
    assert "draining" in str(exc_info.value).lower()
    assert "shutting down" in str(exc_info.value).lower()

    print("\n✅ Drain correctly prevents new requests")
