"""Stateful chunked prefill tracker for interleaved scheduling.

PrefillState tracks the progress of a single sequence's chunked prefill,
allowing the scheduler to process one chunk at a time and interleave
prefill chunks with decode steps for other sequences.

This is a pure application object — no MLX or infrastructure imports.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PrefillState:
    """Tracks progress of a single sequence's chunked prefill.

    The scheduler creates a PrefillState for each incoming request with a
    long prompt. Each call to advance() marks one chunk as processed.
    When is_done is True, the sequence is promoted to decode in BatchGenerator.

    Attributes:
        agent_id: Unique identifier for the agent/session.
        tokens: Full prompt token sequence to prefill.
        pos: Current position in tokens (next chunk starts here).
        max_tokens: Maximum tokens to generate after prefill.
        kv_caches: Opaque KV cache state built up during prefill.
            Set by the PrefillChunkPort adapter. Type depends on backend.
        chunk_count: Number of chunks processed so far.
    """

    agent_id: str
    tokens: list[int]
    pos: int = 0
    max_tokens: int = 256
    kv_caches: Any = None
    chunk_count: int = 0
    _request_ref: Any = field(default=None, repr=False)

    @property
    def prefill_end(self) -> int:
        """Position up to which chunked prefill should process.

        We stop one token before the end because BatchGenerator
        needs at least one unprocessed token to produce initial
        logits for generation. Processing all tokens leaves
        BatchGenerator with an empty prompt, which corrupts the
        MoE routing indices.
        """
        return max(0, len(self.tokens) - 1)

    @property
    def is_done(self) -> bool:
        return self.pos >= self.prefill_end

    @property
    def total_tokens(self) -> int:
        return len(self.tokens)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.prefill_end - self.pos)

    def next_chunk_range(self, chunk_size: int) -> tuple[int, int]:
        """Return (start, end) for the next chunk without advancing pos.

        Args:
            chunk_size: Maximum tokens to include in this chunk.

        Returns:
            Tuple of (start_index, end_index) into self.tokens.
        """
        end = min(self.pos + chunk_size, self.prefill_end)
        return (self.pos, end)

    def advance(self, n_tokens: int) -> None:
        """Mark n_tokens as processed, advancing pos.

        Args:
            n_tokens: Number of tokens just processed in the chunk.
        """
        self.pos += n_tokens
        self.chunk_count += 1
