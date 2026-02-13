# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for adapter_helpers shared functions."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from agent_memory.adapters.inbound.adapter_helpers import (
    STEP_TIMEOUT_SECONDS,
    get_semantic_state,
    run_step_for_uid,
    try_parse_json_at,
)


class TestGetSemanticState:
    def test_returns_state_when_present(self) -> None:
        mock_state = SimpleNamespace(batch_engine=MagicMock())
        mock_request = MagicMock()
        mock_request.app.state.agent_memory = mock_state
        result = get_semantic_state(mock_request)
        assert result is mock_state

    def test_raises_503_when_not_initialized(self) -> None:
        mock_request = MagicMock()
        mock_request.app.state = SimpleNamespace()  # no 'semantic' attr
        with pytest.raises(HTTPException) as exc_info:
            get_semantic_state(mock_request)
        assert exc_info.value.status_code == 503

    def test_raises_503_when_none(self) -> None:
        mock_request = MagicMock()
        mock_request.app.state.agent_memory = None
        with pytest.raises(HTTPException) as exc_info:
            get_semantic_state(mock_request)
        assert exc_info.value.status_code == 503


class TestRunStepForUid:
    def test_returns_matching_result(self) -> None:
        result_obj = SimpleNamespace(uid="target_uid", text="hello")
        engine = MagicMock()
        engine.step.return_value = iter([result_obj])
        result = run_step_for_uid(engine, "target_uid")
        assert result.uid == "target_uid"
        assert result.text == "hello"

    def test_returns_none_when_no_results(self) -> None:
        engine = MagicMock()
        engine.step.return_value = iter([])
        result = run_step_for_uid(engine, "missing_uid")
        assert result is None

    def test_step_timeout_default(self) -> None:
        assert STEP_TIMEOUT_SECONDS == 300


class TestTryParseJsonAt:
    def test_simple_object(self) -> None:
        text = '{"key": "value"}'
        parsed, end = try_parse_json_at(text, 0)
        assert parsed == {"key": "value"}
        assert end == len(text)

    def test_nested_object(self) -> None:
        text = '{"a": {"b": 1}}'
        parsed, end = try_parse_json_at(text, 0)
        assert parsed == {"a": {"b": 1}}
        assert end == len(text)

    def test_object_with_prefix(self) -> None:
        text = 'data: {"id": "123"} extra'
        parsed, end = try_parse_json_at(text, 6)
        assert parsed == {"id": "123"}
        assert end == 19

    def test_escaped_quotes(self) -> None:
        text = '{"msg": "say \\"hello\\""}'
        parsed, end = try_parse_json_at(text, 0)
        assert parsed == {"msg": 'say "hello"'}

    def test_braces_in_string_ignored(self) -> None:
        text = '{"code": "if (a) { b }"}'
        parsed, end = try_parse_json_at(text, 0)
        assert parsed == {"code": "if (a) { b }"}

    def test_empty_input_returns_none(self) -> None:
        parsed, end = try_parse_json_at("", 0)
        assert parsed is None
        assert end == 0

    def test_start_not_brace_returns_none(self) -> None:
        parsed, end = try_parse_json_at("hello", 0)
        assert parsed is None
        assert end == 0

    def test_start_beyond_length_returns_none(self) -> None:
        parsed, end = try_parse_json_at("{}", 10)
        assert parsed is None
        assert end == 10

    def test_incomplete_json_returns_none(self) -> None:
        parsed, end = try_parse_json_at('{"key": "val', 0)
        assert parsed is None
        assert end == 0

    def test_invalid_json_returns_none(self) -> None:
        text = "{not valid json}"
        parsed, end = try_parse_json_at(text, 0)
        assert parsed is None
        assert end == 0

    def test_multiple_objects_parses_first(self) -> None:
        text = '{"a": 1}{"b": 2}'
        parsed, end = try_parse_json_at(text, 0)
        assert parsed == {"a": 1}
        assert end == 8

    def test_numeric_values(self) -> None:
        text = '{"count": 42, "rate": 3.14}'
        parsed, end = try_parse_json_at(text, 0)
        assert parsed == {"count": 42, "rate": 3.14}

    def test_boolean_and_null_values(self) -> None:
        text = '{"active": true, "deleted": false, "note": null}'
        parsed, end = try_parse_json_at(text, 0)
        assert parsed == {"active": True, "deleted": False, "note": None}
