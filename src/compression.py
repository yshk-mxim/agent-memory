"""
Context compression methods for simulating KV cache eviction.

Implements various compression strategies to test instruction degradation.
"""

import random
from typing import List, Tuple


def compress_context_random(
    full_context: str,
    compression_ratio: float = 0.5,
    keep_first_last: bool = True
) -> str:
    """
    Simulate random token eviction (similar to H2O compression).

    Args:
        full_context: Full conversation context
        compression_ratio: Fraction of content to keep (0-1)
        keep_first_last: If True, always keep first and last sentences

    Returns:
        Compressed context string
    """
    sentences = full_context.split('. ')

    if len(sentences) <= 2:
        return full_context

    keep_count = max(1, int(len(sentences) * compression_ratio))

    if keep_count >= len(sentences):
        return full_context

    if keep_first_last and keep_count >= 2:
        # Keep first and last (like StreamingLLM)
        kept = [sentences[0]]
        middle = sentences[1:-1]
        random.shuffle(middle)
        kept.extend(middle[:max(0, keep_count-2)])
        kept.append(sentences[-1])
    else:
        # Random selection
        kept = random.sample(sentences, keep_count)
        # Sort by original position
        kept = sorted(kept, key=lambda s: sentences.index(s))

    return '. '.join(kept)


def compress_context_sequential(
    full_context: str,
    compression_ratio: float = 0.5,
    keep_recent: bool = True
) -> str:
    """
    Keep only first N% or last N% of context.

    Args:
        full_context: Full conversation context
        compression_ratio: Fraction of content to keep (0-1)
        keep_recent: If True, keep most recent; else keep oldest

    Returns:
        Compressed context string
    """
    sentences = full_context.split('. ')

    keep_count = max(1, int(len(sentences) * compression_ratio))

    if keep_count >= len(sentences):
        return full_context

    if keep_recent:
        # Keep most recent sentences
        kept = sentences[-keep_count:]
    else:
        # Keep oldest sentences
        kept = sentences[:keep_count]

    return '. '.join(kept)


def compress_turns(
    turns: List[dict],
    compression_ratio: float = 0.5,
    method: str = 'random'
) -> List[dict]:
    """
    Compress multi-turn conversation.

    Args:
        turns: List of turn dicts with 'instruction' and 'content' keys
        compression_ratio: Fraction to keep
        method: 'random', 'sequential_recent', or 'sequential_old'

    Returns:
        List of compressed turns
    """
    # Combine all turns into single context
    context_parts = []
    for turn in turns:
        if 'instruction' in turn:
            context_parts.append(f"Instruction: {turn['instruction']}")
        if 'content' in turn:
            context_parts.append(f"Content: {turn['content']}")

    full_context = '. '.join(context_parts)

    # Compress
    if method == 'random':
        compressed = compress_context_random(full_context, compression_ratio)
    elif method == 'sequential_recent':
        compressed = compress_context_sequential(full_context, compression_ratio, keep_recent=True)
    elif method == 'sequential_old':
        compressed = compress_context_sequential(full_context, compression_ratio, keep_recent=False)
    else:
        raise ValueError(f"Unknown compression method: {method}")

    # Return as single "compressed" turn
    return [{'role': 'user', 'content': compressed}]


def estimate_compression_rate(original: str, compressed: str) -> float:
    """
    Estimate actual compression ratio achieved.

    Returns:
        Ratio of compressed/original length
    """
    if not original:
        return 0.0
    return len(compressed) / len(original)


if __name__ == '__main__':
    # Test compression methods
    test_context = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence. Sixth sentence."

    print("Original:", test_context)
    print()

    # Test random compression
    compressed_50 = compress_context_random(test_context, 0.5)
    print(f"Random 50%: {compressed_50}")
    print(f"Actual ratio: {estimate_compression_rate(test_context, compressed_50):.2f}")
    print()

    # Test sequential (recent)
    compressed_recent = compress_context_sequential(test_context, 0.5, keep_recent=True)
    print(f"Sequential (recent) 50%: {compressed_recent}")
    print(f"Actual ratio: {estimate_compression_rate(test_context, compressed_recent):.2f}")
    print()

    # Test sequential (old)
    compressed_old = compress_context_sequential(test_context, 0.5, keep_recent=False)
    print(f"Sequential (old) 50%: {compressed_old}")
    print(f"Actual ratio: {estimate_compression_rate(test_context, compressed_old):.2f}")
