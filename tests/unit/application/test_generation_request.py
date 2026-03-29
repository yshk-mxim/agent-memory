# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for GenerationRequest dataclass — defaults, FIM fields, penalties."""

import pytest

from agent_memory.application.generation_request import GenerationRequest

pytestmark = pytest.mark.unit


class TestDefaults:
    def test_required_fields_only(self) -> None:
        req = GenerationRequest(
            agent_id="agent-1",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.agent_id == "agent-1"
        assert req.messages == [{"role": "user", "content": "hi"}]

    def test_default_prompt(self) -> None:
        req = GenerationRequest(agent_id="a", messages=[])
        assert req.prompt == ""

    def test_default_max_tokens(self) -> None:
        req = GenerationRequest(agent_id="a", messages=[])
        assert req.max_tokens == 256

    def test_default_temperature(self) -> None:
        req = GenerationRequest(agent_id="a", messages=[])
        assert req.temperature == 0.7

    def test_default_top_p(self) -> None:
        req = GenerationRequest(agent_id="a", messages=[])
        assert req.top_p == 0.95

    def test_default_top_k(self) -> None:
        req = GenerationRequest(agent_id="a", messages=[])
        assert req.top_k == 40

    def test_default_stop_sequences_is_empty_list(self) -> None:
        req = GenerationRequest(agent_id="a", messages=[])
        assert req.stop_sequences == []
        # Verify independent instances (not shared mutable default)
        req2 = GenerationRequest(agent_id="b", messages=[])
        req.stop_sequences.append("test")
        assert req2.stop_sequences == []

    def test_default_stream_false(self) -> None:
        req = GenerationRequest(agent_id="a", messages=[])
        assert req.stream is False

    def test_default_model_empty(self) -> None:
        req = GenerationRequest(agent_id="a", messages=[])
        assert req.model == ""


class TestFIMFields:
    def test_fim_defaults_disabled(self) -> None:
        req = GenerationRequest(agent_id="a", messages=[])
        assert req.fim_prefix is None
        assert req.fim_suffix is None
        assert req.fim_mode is False

    def test_fim_mode_enabled(self) -> None:
        req = GenerationRequest(
            agent_id="a",
            messages=[],
            fim_mode=True,
            fim_prefix="def hello():",
            fim_suffix="\nreturn 42",
        )
        assert req.fim_mode is True
        assert req.fim_prefix == "def hello():"
        assert req.fim_suffix == "\nreturn 42"

    def test_fim_prefix_without_suffix(self) -> None:
        req = GenerationRequest(
            agent_id="a",
            messages=[],
            fim_mode=True,
            fim_prefix="code before cursor",
        )
        assert req.fim_prefix == "code before cursor"
        assert req.fim_suffix is None

    def test_fim_suffix_without_mode(self) -> None:
        """FIM fields can be set without enabling FIM mode."""
        req = GenerationRequest(
            agent_id="a",
            messages=[],
            fim_prefix="prefix",
            fim_suffix="suffix",
            fim_mode=False,
        )
        assert req.fim_mode is False
        assert req.fim_prefix == "prefix"


class TestRepetitionControl:
    def test_default_repetition_penalty(self) -> None:
        req = GenerationRequest(agent_id="a", messages=[])
        assert req.repetition_penalty == 1.0

    def test_default_frequency_penalty(self) -> None:
        req = GenerationRequest(agent_id="a", messages=[])
        assert req.frequency_penalty == 0.0

    def test_default_presence_penalty(self) -> None:
        req = GenerationRequest(agent_id="a", messages=[])
        assert req.presence_penalty == 0.0

    def test_custom_penalties(self) -> None:
        req = GenerationRequest(
            agent_id="a",
            messages=[],
            repetition_penalty=1.2,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )
        assert req.repetition_penalty == 1.2
        assert req.frequency_penalty == 0.5
        assert req.presence_penalty == 0.3


class TestSystemPromptPin:
    def test_default_pin_false(self) -> None:
        req = GenerationRequest(agent_id="a", messages=[])
        assert req.pin_system_cache is False

    def test_pin_enabled(self) -> None:
        req = GenerationRequest(agent_id="a", messages=[], pin_system_cache=True)
        assert req.pin_system_cache is True


class TestCustomValues:
    def test_all_fields_set(self) -> None:
        req = GenerationRequest(
            agent_id="my-agent",
            messages=[{"role": "system", "content": "Be helpful"}],
            prompt="Hello world",
            max_tokens=512,
            temperature=0.9,
            top_p=0.8,
            top_k=20,
            stop_sequences=["<|end|>", "\n\n"],
            stream=True,
            model="smollm2-135m",
            fim_prefix="pre",
            fim_suffix="suf",
            fim_mode=True,
            repetition_penalty=1.1,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            pin_system_cache=True,
        )
        assert req.agent_id == "my-agent"
        assert req.prompt == "Hello world"
        assert req.max_tokens == 512
        assert req.temperature == 0.9
        assert req.top_p == 0.8
        assert req.top_k == 20
        assert req.stop_sequences == ["<|end|>", "\n\n"]
        assert req.stream is True
        assert req.model == "smollm2-135m"
        assert req.fim_prefix == "pre"
        assert req.fim_suffix == "suf"
        assert req.fim_mode is True
        assert req.repetition_penalty == 1.1
        assert req.frequency_penalty == 0.2
        assert req.presence_penalty == 0.1
        assert req.pin_system_cache is True
