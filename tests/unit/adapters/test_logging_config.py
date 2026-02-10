# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for structured logging configuration."""


import pytest
import structlog

from agent_memory.adapters.config.logging import configure_logging, get_logger

pytestmark = pytest.mark.unit


class TestConfigureLogging:
    def test_configures_structlog_with_info(self) -> None:
        configure_logging("INFO", json_output=True)
        config = structlog.get_config()
        assert config["cache_logger_on_first_use"] is True

    def test_no_crash_with_debug_level(self) -> None:
        configure_logging("DEBUG", json_output=True)

    def test_no_crash_with_warning_level(self) -> None:
        configure_logging("WARNING", json_output=True)

    def test_no_crash_with_error_level(self) -> None:
        configure_logging("ERROR", json_output=True)

    def test_no_crash_with_critical_level(self) -> None:
        configure_logging("CRITICAL", json_output=True)

    def test_case_insensitive_level(self) -> None:
        # Should not raise for lowercase
        configure_logging("info", json_output=True)
        configure_logging("debug", json_output=False)

    def test_invalid_level_defaults_to_info_no_crash(self) -> None:
        # Should not crash; falls back to INFO internally
        configure_logging("INVALID", json_output=True)
        configure_logging("GARBAGE", json_output=False)
        configure_logging("", json_output=True)


class TestConfigureLoggingJsonMode:
    def test_json_output_uses_json_renderer(self) -> None:
        configure_logging("INFO", json_output=True)
        config = structlog.get_config()
        processor_types = [type(p).__name__ for p in config["processors"]]
        assert "JSONRenderer" in processor_types

    def test_console_output_uses_console_renderer(self) -> None:
        configure_logging("INFO", json_output=False)
        config = structlog.get_config()
        processor_types = [type(p).__name__ for p in config["processors"]]
        assert "ConsoleRenderer" in processor_types

    def test_json_mode_includes_timestamper(self) -> None:
        configure_logging("INFO", json_output=True)
        config = structlog.get_config()
        processor_types = [type(p).__name__ for p in config["processors"]]
        assert "TimeStamper" in processor_types

    def test_console_mode_includes_timestamper(self) -> None:
        configure_logging("INFO", json_output=False)
        config = structlog.get_config()
        processor_types = [type(p).__name__ for p in config["processors"]]
        assert "TimeStamper" in processor_types

    def test_json_mode_includes_log_level_processor(self) -> None:
        configure_logging("INFO", json_output=True)
        config = structlog.get_config()
        processor_names = [getattr(p, "__name__", type(p).__name__) for p in config["processors"]]
        assert any("log_level" in str(name).lower() for name in processor_names)

    def test_json_mode_has_more_processors_than_console(self) -> None:
        configure_logging("INFO", json_output=True)
        json_count = len(structlog.get_config()["processors"])

        configure_logging("INFO", json_output=False)
        console_count = len(structlog.get_config()["processors"])

        # JSON mode has StackInfoRenderer + format_exc_info that console doesn't
        assert json_count > console_count


class TestGetLogger:
    def test_returns_logger_with_name(self) -> None:
        configure_logging("INFO", json_output=True)
        logger = get_logger("test_module")
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "debug")

    def test_returns_logger_without_name(self) -> None:
        configure_logging("INFO", json_output=True)
        logger = get_logger()
        assert hasattr(logger, "info")

    def test_returns_logger_with_none_name(self) -> None:
        configure_logging("INFO", json_output=True)
        logger = get_logger(None)
        assert hasattr(logger, "info")

    def test_logger_is_callable(self) -> None:
        configure_logging("INFO", json_output=True)
        logger = get_logger("test")
        # Should not crash when logging
        logger.info("test message", key="value")
