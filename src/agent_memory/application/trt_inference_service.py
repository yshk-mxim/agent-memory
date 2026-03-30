# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT inference service — implements InferencePort for TRT backend.

Wraps TRTSubprocessAdapter (ModelBackendPort) with cache persistence,
providing the same generate(agent_id, prompt) interface as the MLX path.
"""

import logging
from typing import Any

import numpy as np

from agent_memory.application.generation_request import GenerationRequest
from agent_memory.domain.entities import AgentBlocks, KVBlock
from agent_memory.domain.value_objects import GenerationResult
from agent_memory.ports.outbound import ModelBackendPort

logger = logging.getLogger(__name__)


class TRTInferenceService:
    """Application service bridging TRT backend with cache persistence.

    Caller provides agent_id + messages; this service handles
    cache load → inject → generate → extract → save transparently.
    """

    def __init__(
        self,
        backend: ModelBackendPort,
        tokenizer: Any,
        cache_adapter: Any | None = None,
        quantizer: Any | None = None,
    ) -> None:
        """Initialize TRT inference service.

        Args:
            backend: TRT subprocess adapter (implements ModelBackendPort).
            tokenizer: HuggingFace tokenizer for prompt processing.
            cache_adapter: Cache persistence adapter (TRTSafetensorsCacheAdapter).
            quantizer: CacheQuantizationPort for Q4→FP16 dequantization on load.
        """
        self._backend = backend
        self._tokenizer = tokenizer
        self._cache_adapter = cache_adapter
        self._quantizer = quantizer

    @property
    def tokenizer(self) -> Any:
        """Tokenizer for prompt processing."""
        return self._tokenizer

    def generate(
        self,
        agent_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        messages: list[dict[str, str]] | None = None,
        top_p: float = 0.95,
        top_k: int = 40,
        stop_sequences: list[str] | None = None,
    ) -> GenerationResult:
        """Generate text with automatic KV cache persistence.

        Args:
            agent_id: Agent identifier for cache lookup/save.
            prompt: Text prompt (used for tokenization/fallback).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            messages: Chat messages for the engine's tokenizer.
            top_p: Top-p (nucleus) sampling parameter.
            top_k: Top-k sampling parameter.
            stop_sequences: Optional stop strings to truncate output at.

        Returns:
            GenerationResult with text, tokens, and updated cache.
        """
        # Load cached KV state for this agent
        cached_kv = self._load_agent_cache(agent_id)

        # Tokenize prompt for token count
        tokens = self._tokenizer.encode(prompt)

        # Generate via backend (all sampling params forwarded)
        result = self._backend.generate(  # type: ignore[call-arg]
            prompt_tokens=tokens,
            cache=cached_kv,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
        )

        # Save updated cache to disk
        if result.cache:
            self._save_agent_cache(agent_id, result.cache, len(tokens) + len(result.tokens))

        # Strip special tokens from output (model-agnostic via tokenizer)
        cleaned_text = self._strip_special_tokens(result.text)

        # Apply stop sequences (post-generation truncation).
        # Edge-LLM runtime does not support stop sequences natively,
        # so we truncate after the fact. This means the model may
        # generate past the stop point, wasting some compute.
        if stop_sequences:
            earliest_stop = len(cleaned_text)
            for seq in stop_sequences:
                idx = cleaned_text.find(seq)
                if idx != -1 and idx < earliest_stop:
                    earliest_stop = idx
            if earliest_stop < len(cleaned_text):
                cleaned_text = cleaned_text[:earliest_stop]

        # Strip thinking tags — extract final answer after </think>
        cleaned_text = self._strip_thinking_tags(cleaned_text)

        return GenerationResult(
            text=cleaned_text,
            tokens=result.tokens,
            cache=result.cache,
        )

    def generate_from_request(self, req: GenerationRequest) -> GenerationResult:
        """Generate from a unified GenerationRequest (preferred entry point).

        Both Anthropic and OpenAI adapters build a GenerationRequest,
        then call this method. Handles FIM mode, penalties, stop sequences.
        """
        messages = req.messages

        # FIM (fill-in-the-middle) mode: construct infill prompt
        if req.fim_mode and req.fim_prefix is not None:
            fim_prompt = self._build_fim_prompt(req.fim_prefix, req.fim_suffix)
            messages = [{"role": "user", "content": fim_prompt}]

        return self.generate(
            agent_id=req.agent_id,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            messages=messages,
            top_p=req.top_p,
            top_k=req.top_k,
            stop_sequences=req.stop_sequences or None,
        )

    def _build_fim_prompt(self, prefix: str, suffix: str | None = None) -> str:
        """Build a fill-in-the-middle prompt using the tokenizer's FIM tokens.

        Falls back to generic FIM format if tokenizer lacks FIM-specific tokens.

        Args:
            prefix: Code before the cursor.
            suffix: Code after the cursor (None = complete from prefix only).

        Returns:
            FIM-formatted prompt string.
        """
        # Check tokenizer for model-specific FIM tokens
        fim_prefix_token = getattr(self._tokenizer, "fim_prefix_token", "<|fim_prefix|>")
        fim_suffix_token = getattr(self._tokenizer, "fim_suffix_token", "<|fim_suffix|>")
        fim_middle_token = getattr(self._tokenizer, "fim_middle_token", "<|fim_middle|>")

        # Also check for common FIM token patterns
        if hasattr(self._tokenizer, "special_tokens_map"):
            stm = self._tokenizer.special_tokens_map
            if "fim_prefix" in stm:
                fim_prefix_token = stm["fim_prefix"]
            if "fim_suffix" in stm:
                fim_suffix_token = stm["fim_suffix"]
            if "fim_middle" in stm:
                fim_middle_token = stm["fim_middle"]

        if suffix:
            return f"{fim_prefix_token}{prefix}{fim_suffix_token}{suffix}{fim_middle_token}"
        return f"{fim_prefix_token}{prefix}{fim_middle_token}"

    def _strip_special_tokens(self, text: str) -> str:
        """Remove special tokens from generated text using the tokenizer.

        Model-agnostic: uses tokenizer.all_special_tokens to strip markers
        like <|im_end|>, <|im_start|>, </s>, <s>, etc.
        Works for ChatML (SmolLM2), Llama, Qwen, and other formats.
        """
        if not text:
            return text

        # Get special tokens from tokenizer
        special_tokens = set()
        if hasattr(self._tokenizer, "all_special_tokens"):
            special_tokens = set(self._tokenizer.all_special_tokens)
        if hasattr(self._tokenizer, "additional_special_tokens"):
            special_tokens.update(self._tokenizer.additional_special_tokens)

        # Also strip common role markers that may leak through
        role_markers = {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}
        special_tokens.update(role_markers)

        # Strip all special tokens from text
        cleaned = text
        for token in special_tokens:
            cleaned = cleaned.replace(token, "")

        # Strip role prefixes like "system\n", "user\n", "assistant\n"
        for role in ("system", "user", "assistant"):
            if cleaned.startswith(f"{role}\n"):
                cleaned = cleaned[len(role) + 1 :]

        return cleaned.strip()

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """Strip <think>...</think> reasoning tags, return final answer only.

        Qwen3.5 and other reasoning models wrap chain-of-thought in think tags.
        The actual response comes after the closing </think> tag.
        If no think tags present, returns text unchanged.
        """
        if "</think>" in text:
            # Return everything after the last </think>
            parts = text.rsplit("</think>", 1)
            return parts[-1].strip()
        # Strip opening <think> without closing (incomplete thinking)
        if text.startswith("<think>"):
            return text.replace("<think>", "").strip()
        # Also handle "Thinking Process:" prefix (Qwen3.5 non-tag format)
        if text.startswith("Thinking Process:"):
            # Find first line that doesn't start with a number or whitespace
            lines = text.split("\n")
            for i, line in enumerate(lines):
                stripped = line.strip()
                is_reasoning = stripped[0].isdigit() or stripped.startswith(("*", "-", "Thinking"))
                if stripped and not is_reasoning and i > 0:
                    return "\n".join(lines[i:]).strip()
        return text

    def _load_agent_cache(self, agent_id: str) -> list[Any] | None:
        """Load KV cache for agent from disk."""
        if self._cache_adapter is None:
            return None
        try:
            if not self._cache_adapter.exists(agent_id):
                return None
            agent_blocks, _metadata = self._cache_adapter.load(agent_id)
            return self._blocks_to_kv_cache(agent_blocks)
        except Exception:
            logger.warning(f"Failed to load cache for {agent_id}", exc_info=True)
            return None

    def _save_agent_cache(
        self,
        agent_id: str,
        cache: list[Any],
        total_tokens: int,
    ) -> None:
        """Save KV cache to disk as AgentBlocks."""
        if self._cache_adapter is None:
            return
        try:
            blocks_dict: dict[int, list[KVBlock]] = {}
            for layer_idx, kv_pair in enumerate(cache):
                block = KVBlock(
                    block_id=layer_idx * 1_000_000,
                    layer_id=layer_idx,
                    token_count=total_tokens,
                    layer_data=kv_pair,
                )
                blocks_dict[layer_idx] = [block]

            agent_blocks = AgentBlocks(
                agent_id=agent_id,
                blocks=blocks_dict,
                total_tokens=total_tokens,
            )

            metadata = {
                "agent_id": agent_id,
                "total_tokens": str(total_tokens),
                "n_layers": str(len(cache)),
            }

            self._cache_adapter.save(agent_id, agent_blocks, metadata)
            logger.debug(
                f"Saved TRT cache for {agent_id}: {len(cache)} layers, {total_tokens} tokens"
            )
        except Exception:
            logger.warning(f"Failed to save cache for {agent_id}", exc_info=True)

    def _blocks_to_kv_cache(self, blocks: AgentBlocks) -> list[Any]:
        """Convert AgentBlocks to per-layer FP16 KV tuples for backend injection.

        If blocks contain Q4 quantized data (tuples of weights/scales/biases),
        dequantizes to FP16 before returning. The TRT subprocess expects FP16.
        """
        kv_cache = []
        for layer_id in sorted(blocks.blocks.keys()):
            layer_blocks = blocks.blocks[layer_id]
            if not layer_blocks or layer_blocks[0].layer_data is None:
                continue

            data = layer_blocks[0].layer_data

            # Check if data is quantized: ((kw,ks,kb), (vw,vs,vb))
            quantized_tuple_len = 3
            if (
                isinstance(data, tuple)
                and len(data) == 2  # noqa: PLR2004
                and isinstance(data[0], tuple)
                and len(data[0]) == quantized_tuple_len
                and self._quantizer is not None
            ):
                (kw, ks, kb), (vw, vs, vb) = data
                # Dequantize returns flat 1D array — reshape to [n_kv_heads, seq_len, head_dim]
                # The shape info must be inferred from the quantized data size
                k_flat = self._quantizer.dequantize(kw, ks, kb, bits=4, group_size=64)
                v_flat = self._quantizer.dequantize(vw, vs, vb, bits=4, group_size=64)

                # Get model spec for reshape dimensions
                spec = self._backend.extract_model_spec()
                n_kv_heads = spec.n_kv_heads
                head_dim = spec.head_dim
                k_elements = k_flat.shape[0]
                seq_len = k_elements // (n_kv_heads * head_dim)

                k_fp16 = (
                    k_flat[: n_kv_heads * seq_len * head_dim]
                    .reshape(n_kv_heads, seq_len, head_dim)
                    .astype(np.float16)
                )
                v_fp16 = (
                    v_flat[: n_kv_heads * seq_len * head_dim]
                    .reshape(n_kv_heads, seq_len, head_dim)
                    .astype(np.float16)
                )
                kv_cache.append((k_fp16, v_fp16))
            else:
                kv_cache.append(data)

        return kv_cache if kv_cache else []
