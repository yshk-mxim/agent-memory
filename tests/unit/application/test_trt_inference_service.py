# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for TRT inference service — generate, FIM, stop sequences, strip tokens."""

from unittest.mock import Mock

import pytest

from agent_memory.application.generation_request import GenerationRequest
from agent_memory.application.trt_inference_service import TRTInferenceService
from agent_memory.domain.value_objects import GenerationResult

pytestmark = pytest.mark.unit


def _mock_tokenizer(
    all_special_tokens: list[str] | None = None,
    additional_special_tokens: list[str] | None = None,
    special_tokens_map: dict[str, str] | None = None,
) -> Mock:
    """Create a mock tokenizer with configurable special token lists."""
    tok = Mock()
    tok.encode.return_value = [1, 2, 3]

    tok.all_special_tokens = (
        all_special_tokens
        if all_special_tokens is not None
        else ["<s>", "</s>", "<|im_start|>", "<|im_end|>"]
    )
    tok.additional_special_tokens = (
        additional_special_tokens if additional_special_tokens is not None else []
    )

    if special_tokens_map is not None:
        tok.special_tokens_map = special_tokens_map
    else:
        # Remove the attribute so hasattr returns False
        del tok.special_tokens_map

    return tok


def _mock_backend(text: str = "Hello", tokens: list[int] | None = None) -> Mock:
    """Create a mock backend that returns a fixed GenerationResult."""
    backend = Mock()
    result = GenerationResult(
        text=text,
        tokens=tokens or [101, 102],
        cache=[(Mock(), Mock())],
    )
    backend.generate.return_value = result
    return backend


@pytest.fixture
def service() -> TRTInferenceService:
    return TRTInferenceService(
        backend=_mock_backend(),
        tokenizer=_mock_tokenizer(),
        cache_adapter=None,
        quantizer=None,
    )


class TestGenerateFromRequest:
    def test_delegates_to_generate(self) -> None:
        backend = _mock_backend("world")
        tokenizer = _mock_tokenizer()
        service = TRTInferenceService(backend=backend, tokenizer=tokenizer)

        req = GenerationRequest(
            agent_id="agent-1",
            messages=[{"role": "user", "content": "hi"}],
            prompt="hi",
            max_tokens=100,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
        )

        result = service.generate_from_request(req)

        assert result.text == "world"
        backend.generate.assert_called_once()
        call_kwargs = backend.generate.call_args
        assert call_kwargs.kwargs["max_tokens"] == 100
        assert call_kwargs.kwargs["temperature"] == 0.5
        assert call_kwargs.kwargs["top_p"] == 0.9
        assert call_kwargs.kwargs["top_k"] == 50

    def test_fim_mode_constructs_fim_prompt(self) -> None:
        backend = _mock_backend("completed code")
        tokenizer = _mock_tokenizer()
        service = TRTInferenceService(backend=backend, tokenizer=tokenizer)

        req = GenerationRequest(
            agent_id="agent-1",
            messages=[{"role": "user", "content": "original"}],
            prompt="test",
            fim_mode=True,
            fim_prefix="def hello():\n    ",
            fim_suffix="\n    return result",
        )

        result = service.generate_from_request(req)

        # FIM mode should override messages
        call_kwargs = backend.generate.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        # The content should contain FIM tokens
        content = messages[0]["content"]
        assert "def hello():\n    " in content
        assert "return result" in content

    def test_fim_mode_disabled_passes_original_messages(self) -> None:
        backend = _mock_backend()
        tokenizer = _mock_tokenizer()
        service = TRTInferenceService(backend=backend, tokenizer=tokenizer)

        original_msgs = [{"role": "user", "content": "just chat"}]
        req = GenerationRequest(
            agent_id="agent-1",
            messages=original_msgs,
            prompt="test",
            fim_mode=False,
            fim_prefix="some prefix",
        )

        service.generate_from_request(req)

        call_kwargs = backend.generate.call_args
        assert call_kwargs.kwargs["messages"] is original_msgs

    def test_empty_stop_sequences_passed_as_none(self) -> None:
        backend = _mock_backend()
        tokenizer = _mock_tokenizer()
        service = TRTInferenceService(backend=backend, tokenizer=tokenizer)

        req = GenerationRequest(
            agent_id="agent-1",
            messages=[],
            prompt="test",
            stop_sequences=[],
        )

        service.generate_from_request(req)

        call_kwargs = backend.generate.call_args
        assert call_kwargs.kwargs["stop_sequences"] is None


class TestStripSpecialTokens:
    def test_strips_im_start_end(self, service: TRTInferenceService) -> None:
        result = service._strip_special_tokens("<|im_start|>Hello<|im_end|>")
        assert result == "Hello"

    def test_strips_bos_eos(self, service: TRTInferenceService) -> None:
        result = service._strip_special_tokens("<s>Hello world</s>")
        assert result == "Hello world"

    def test_strips_endoftext(self, service: TRTInferenceService) -> None:
        result = service._strip_special_tokens("output<|endoftext|>")
        assert result == "output"

    def test_strips_role_prefix_assistant(self, service: TRTInferenceService) -> None:
        result = service._strip_special_tokens("assistant\nHere is the answer")
        assert result == "Here is the answer"

    def test_strips_role_prefix_user(self, service: TRTInferenceService) -> None:
        result = service._strip_special_tokens("user\nWhat is 2+2?")
        assert result == "What is 2+2?"

    def test_strips_role_prefix_system(self, service: TRTInferenceService) -> None:
        result = service._strip_special_tokens("system\nYou are a helper")
        assert result == "You are a helper"

    def test_empty_string_unchanged(self, service: TRTInferenceService) -> None:
        assert service._strip_special_tokens("") == ""

    def test_no_special_tokens_preserved(self, service: TRTInferenceService) -> None:
        result = service._strip_special_tokens("Just normal text.")
        assert result == "Just normal text."

    def test_strips_whitespace(self, service: TRTInferenceService) -> None:
        result = service._strip_special_tokens("  Hello  ")
        assert result == "Hello"

    def test_tokenizer_without_special_tokens_attr(self) -> None:
        tok = Mock(spec=["encode"])  # Minimal spec, no special token attrs
        tok.encode.return_value = [1, 2, 3]
        service = TRTInferenceService(backend=Mock(), tokenizer=tok)
        # Should still strip common role markers (hardcoded in the method)
        result = service._strip_special_tokens("<|im_end|>text<|im_start|>")
        assert result == "text"


class TestBuildFimPrompt:
    def test_with_prefix_and_suffix(self) -> None:
        tokenizer = Mock(spec=[])  # No FIM-specific attrs
        service = TRTInferenceService(backend=Mock(), tokenizer=tokenizer)

        result = service._build_fim_prompt("def foo():", "\nreturn 42")
        assert result == "<|fim_prefix|>def foo():<|fim_suffix|>\nreturn 42<|fim_middle|>"

    def test_prefix_only(self) -> None:
        tokenizer = Mock(spec=[])
        service = TRTInferenceService(backend=Mock(), tokenizer=tokenizer)

        result = service._build_fim_prompt("def foo():", None)
        assert result == "<|fim_prefix|>def foo():<|fim_middle|>"

    def test_empty_suffix(self) -> None:
        tokenizer = Mock(spec=[])
        service = TRTInferenceService(backend=Mock(), tokenizer=tokenizer)

        # Empty string is falsy, so treated as no suffix
        result = service._build_fim_prompt("code", "")
        assert result == "<|fim_prefix|>code<|fim_middle|>"

    def test_uses_tokenizer_fim_tokens(self) -> None:
        tokenizer = Mock()
        tokenizer.fim_prefix_token = "<PRE>"  # noqa: S105
        tokenizer.fim_suffix_token = "<SUF>"  # noqa: S105
        tokenizer.fim_middle_token = "<MID>"  # noqa: S105
        # No special_tokens_map
        del tokenizer.special_tokens_map
        service = TRTInferenceService(backend=Mock(), tokenizer=tokenizer)

        result = service._build_fim_prompt("before", "after")
        assert result == "<PRE>before<SUF>after<MID>"

    def test_uses_special_tokens_map(self) -> None:
        tokenizer = Mock(spec=[])
        tokenizer.special_tokens_map = {
            "fim_prefix": "[PREFIX]",
            "fim_suffix": "[SUFFIX]",
            "fim_middle": "[MIDDLE]",
        }
        service = TRTInferenceService(backend=Mock(), tokenizer=tokenizer)

        result = service._build_fim_prompt("A", "B")
        assert result == "[PREFIX]A[SUFFIX]B[MIDDLE]"


class TestStopSequences:
    def test_single_stop_sequence_truncates(self) -> None:
        backend = _mock_backend("Hello\n---\nMore text after stop")
        tokenizer = _mock_tokenizer()
        service = TRTInferenceService(backend=backend, tokenizer=tokenizer)

        result = service.generate(
            agent_id="a1",
            prompt="hi",
            stop_sequences=["---"],
        )

        assert result.text == "Hello\n"

    def test_multiple_stop_sequences_earliest_wins(self) -> None:
        backend = _mock_backend("AB<stop1>CD<stop2>EF")
        tokenizer = _mock_tokenizer()
        service = TRTInferenceService(backend=backend, tokenizer=tokenizer)

        result = service.generate(
            agent_id="a1",
            prompt="hi",
            stop_sequences=["<stop2>", "<stop1>"],
        )

        assert result.text == "AB"

    def test_no_stop_sequence_match_preserves_text(self) -> None:
        backend = _mock_backend("No stop here")
        tokenizer = _mock_tokenizer()
        service = TRTInferenceService(backend=backend, tokenizer=tokenizer)

        result = service.generate(
            agent_id="a1",
            prompt="hi",
            stop_sequences=["<|STOP|>"],
        )

        assert result.text == "No stop here"

    def test_stop_at_start_returns_empty(self) -> None:
        backend = _mock_backend("<end>everything after")
        tokenizer = _mock_tokenizer()
        service = TRTInferenceService(backend=backend, tokenizer=tokenizer)

        result = service.generate(
            agent_id="a1",
            prompt="hi",
            stop_sequences=["<end>"],
        )

        assert result.text == ""

    def test_no_stop_sequences_returns_full_text(self) -> None:
        backend = _mock_backend("Full output here")
        tokenizer = _mock_tokenizer()
        service = TRTInferenceService(backend=backend, tokenizer=tokenizer)

        result = service.generate(
            agent_id="a1",
            prompt="hi",
        )

        assert result.text == "Full output here"


class TestCachePersistence:
    def test_saves_cache_after_generation(self) -> None:
        cache_adapter = Mock()
        cache_adapter.exists.return_value = False

        backend = _mock_backend("output")
        tokenizer = _mock_tokenizer()
        service = TRTInferenceService(
            backend=backend, tokenizer=tokenizer, cache_adapter=cache_adapter
        )

        service.generate(agent_id="a1", prompt="hi")

        cache_adapter.save.assert_called_once()
        call_args = cache_adapter.save.call_args
        assert call_args.args[0] == "a1"

    def test_loads_cache_when_exists(self) -> None:
        import numpy as np

        from agent_memory.domain.entities import AgentBlocks, KVBlock

        k = np.zeros((4, 32, 128), dtype=np.float16)
        v = np.zeros((4, 32, 128), dtype=np.float16)
        blocks = AgentBlocks(
            agent_id="a1",
            blocks={0: [KVBlock(block_id=0, layer_id=0, token_count=32, layer_data=(k, v))]},
            total_tokens=32,
        )

        cache_adapter = Mock()
        cache_adapter.exists.return_value = True
        cache_adapter.load.return_value = (blocks, {"total_tokens": "32"})

        backend = _mock_backend("output")
        tokenizer = _mock_tokenizer()
        service = TRTInferenceService(
            backend=backend, tokenizer=tokenizer, cache_adapter=cache_adapter
        )

        service.generate(agent_id="a1", prompt="hi")

        cache_adapter.load.assert_called_once_with("a1")

    def test_no_cache_adapter_skips_persistence(self) -> None:
        backend = _mock_backend("output")
        tokenizer = _mock_tokenizer()
        service = TRTInferenceService(backend=backend, tokenizer=tokenizer, cache_adapter=None)

        result = service.generate(agent_id="a1", prompt="hi")
        assert result.text == "output"


class TestTokenizerProperty:
    def test_tokenizer_property_returns_tokenizer(self) -> None:
        tokenizer = _mock_tokenizer()
        service = TRTInferenceService(backend=Mock(), tokenizer=tokenizer)
        assert service.tokenizer is tokenizer
