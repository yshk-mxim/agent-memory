# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""Unit tests for TRT model loader — tokenizer load, subprocess launch."""

import json
from io import BytesIO
from unittest.mock import MagicMock, Mock, patch

import pytest

pytestmark = pytest.mark.unit

# TRTModelLoader imports chain through ports → domain which may trigger
# MLX import errors when integration conftest corrupts sys.modules.
# Guard the import and skip if it fails.
try:
    from agent_memory.adapters.outbound.trt_model_loader import TRTModelLoader
    from agent_memory.domain.errors import TRTEngineError

    _IMPORT_OK = True
except (ValueError, ImportError):
    _IMPORT_OK = False

skip_if_import_broken = pytest.mark.skipif(
    not _IMPORT_OK, reason="MLX mock conflict during test collection"
)


@skip_if_import_broken
class TestLoadTokenizerFailure:
    """Test tokenizer load error handling."""

    def test_bad_model_id_raises_trt_engine_error(self) -> None:
        with patch("agent_memory.adapters.outbound.trt_model_loader.AutoTokenizer") as mock_tok:
            mock_tok.from_pretrained.side_effect = Exception("Model not found")
            loader = TRTModelLoader()
            with pytest.raises(TRTEngineError, match="Failed to load tokenizer"):
                loader.load(
                    model_id="nonexistent/model",
                    engine_path="/fake",
                    llm_inference_bin="/fake/bin",
                )


@skip_if_import_broken
class TestSuccessfulLoad:
    """Test successful tokenizer + subprocess load."""

    def test_returns_backend_and_tokenizer(self) -> None:
        with (
            patch("agent_memory.adapters.outbound.trt_model_loader.AutoTokenizer") as mock_tok,
            patch(
                "agent_memory.adapters.outbound.trt_subprocess_adapter.subprocess.Popen"
            ) as mock_popen,
        ):
            mock_tok.from_pretrained.return_value = Mock()

            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_proc.stdout = BytesIO(json.dumps({"status": "ready"}).encode() + b"\n")
            mock_proc.stdin = BytesIO()
            mock_proc.stderr = BytesIO()
            mock_popen.return_value = mock_proc

            loader = TRTModelLoader()
            backend, tokenizer = loader.load(
                model_id="test/model",
                engine_path="/fake",
                llm_inference_bin="/fake/bin",
            )

            assert tokenizer is not None
            assert backend is not None
