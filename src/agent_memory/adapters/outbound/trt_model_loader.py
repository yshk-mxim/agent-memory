# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yakov Shkolnikov and contributors
"""TRT model loader.

Launches the ``llm_inference`` subprocess and loads the HuggingFace
tokenizer (same ``transformers`` library as the MLX path).
"""

from typing import Any

import structlog
from transformers import AutoTokenizer

from agent_memory.domain.errors import TRTEngineError
from agent_memory.ports.outbound import ModelBackendPort

logger = structlog.get_logger(__name__)


class TRTModelLoader:
    """Load TRT engine and tokenizer."""

    def load(
        self,
        model_id: str,
        engine_path: str,
        llm_inference_bin: str,
        timeout_s: float = 30.0,
        shm_dir: str = "/dev/shm",  # noqa: S108
    ) -> tuple[ModelBackendPort, Any]:
        """Launch subprocess and load tokenizer.

        Args:
            model_id: HuggingFace model ID (for tokenizer).
            engine_path: Path to TRT engine directory.
            llm_inference_bin: Path to llm_inference binary.
            timeout_s: Subprocess command timeout.
            shm_dir: Shared memory directory.

        Returns:
            Tuple of (ModelBackendPort, tokenizer).
        """
        # Load tokenizer from HuggingFace (same as MLX path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)  # type: ignore[no-untyped-call]
        except Exception as e:
            raise TRTEngineError(f"Failed to load tokenizer for {model_id}: {e}") from e

        logger.info("trt_tokenizer_loaded", model_id=model_id)

        # Factory creates the concrete adapter (runtime import avoids cross-adapter dep)
        from agent_memory.adapters.outbound.trt_subprocess_adapter import (  # noqa: PLC0415
            TRTSubprocessAdapter,
        )

        adapter = TRTSubprocessAdapter(
            llm_inference_bin=llm_inference_bin,
            engine_path=engine_path,
            timeout_s=timeout_s,
            shm_dir=shm_dir,
        )
        adapter.start()

        logger.info("trt_model_loaded", engine_path=engine_path)
        return adapter, tokenizer
