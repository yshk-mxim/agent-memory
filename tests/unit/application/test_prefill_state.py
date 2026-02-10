"""Unit tests for PrefillState."""


from agent_memory.application.prefill_state import PrefillState


class TestPrefillStateBasic:
    def test_initial_state(self) -> None:
        state = PrefillState(agent_id="a1", tokens=[1, 2, 3, 4, 5])
        assert state.pos == 0
        assert state.total_tokens == 5
        # remaining_tokens = prefill_end - pos = (len-1) - 0 = 4
        # prefill_end reserves 1 token for BatchGenerator initial logits
        assert state.remaining_tokens == 4
        assert state.chunk_count == 0
        assert not state.is_done

    def test_empty_tokens(self) -> None:
        state = PrefillState(agent_id="a1", tokens=[])
        assert state.is_done
        assert state.total_tokens == 0
        assert state.remaining_tokens == 0

    def test_advance(self) -> None:
        state = PrefillState(agent_id="a1", tokens=[1, 2, 3, 4, 5])
        state.advance(3)
        assert state.pos == 3
        # remaining = prefill_end - pos = 4 - 3 = 1
        assert state.remaining_tokens == 1
        assert state.chunk_count == 1
        assert not state.is_done

    def test_advance_to_completion(self) -> None:
        state = PrefillState(agent_id="a1", tokens=[1, 2, 3])
        state.advance(3)
        assert state.is_done
        assert state.remaining_tokens == 0
        assert state.chunk_count == 1

    def test_advance_beyond_end(self) -> None:
        state = PrefillState(agent_id="a1", tokens=[1, 2])
        state.advance(10)
        assert state.is_done
        assert state.remaining_tokens == 0


class TestPrefillStateChunkRange:
    def test_first_chunk(self) -> None:
        state = PrefillState(agent_id="a1", tokens=list(range(100)))
        start, end = state.next_chunk_range(chunk_size=30)
        assert start == 0
        assert end == 30

    def test_chunk_does_not_advance(self) -> None:
        state = PrefillState(agent_id="a1", tokens=list(range(100)))
        state.next_chunk_range(chunk_size=30)
        assert state.pos == 0

    def test_chunk_clamps_to_end(self) -> None:
        state = PrefillState(agent_id="a1", tokens=[1, 2, 3])
        start, end = state.next_chunk_range(chunk_size=100)
        assert start == 0
        # Clamps to prefill_end (len-1=2), not len(tokens)
        assert end == 2

    def test_chunk_after_partial_advance(self) -> None:
        state = PrefillState(agent_id="a1", tokens=list(range(100)))
        state.advance(40)
        start, end = state.next_chunk_range(chunk_size=30)
        assert start == 40
        assert end == 70

    def test_last_chunk_shorter(self) -> None:
        state = PrefillState(agent_id="a1", tokens=list(range(100)))
        state.advance(90)
        start, end = state.next_chunk_range(chunk_size=30)
        assert start == 90
        # Clamps to prefill_end (100-1=99), not len(tokens)
        assert end == 99


class TestPrefillStateMultiChunk:
    def test_full_walkthrough(self) -> None:
        tokens = list(range(1000))
        state = PrefillState(agent_id="a1", tokens=tokens, max_tokens=128)
        chunk_size = 256
        chunks_processed = 0

        while not state.is_done:
            start, end = state.next_chunk_range(chunk_size)
            n = end - start
            assert n > 0
            assert n <= chunk_size
            state.advance(n)
            chunks_processed += 1

        assert state.is_done
        # prefill_end = len(tokens)-1 = 999, so pos stops at 999
        assert state.pos == 999
        assert state.chunk_count == chunks_processed
        # 256+256+256+231=999 tokens processed in 4 chunks
        assert chunks_processed == 4

    def test_kv_caches_opaque(self) -> None:
        state = PrefillState(agent_id="a1", tokens=[1, 2, 3])
        assert state.kv_caches is None
        state.kv_caches = {"fake": "cache"}
        assert state.kv_caches == {"fake": "cache"}

    def test_request_ref_not_in_repr(self) -> None:
        state = PrefillState(agent_id="a1", tokens=[1])
        state._request_ref = object()
        r = repr(state)
        assert "_request_ref" not in r
