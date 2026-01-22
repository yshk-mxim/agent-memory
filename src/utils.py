"""
Utility functions and API client wrappers for RDIC project.

Provides unified interfaces for:
- Claude (Anthropic) API
- DeepSeek R1 API (OpenAI-compatible)
- Local Llama inference via llama-cpp-python
"""

import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import get_config


class APIClients:
    """Centralized API client management"""

    def __init__(self, config=None):
        """
        Initialize API clients.

        Args:
            config: Optional Config object. If None, uses global config.
        """
        if config is None:
            config = get_config()

        self.config = config
        self._claude = None
        self._deepseek = None
        self._llama = None

    @property
    def claude(self):
        """Get Claude client (lazy initialization)"""
        if self._claude is None:
            try:
                import anthropic
                self._claude = anthropic.Anthropic(
                    api_key=self.config.claude_api_key
                )
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Run: pip install anthropic"
                )
        return self._claude

    @property
    def deepseek(self):
        """Get DeepSeek client (lazy initialization)"""
        if self._deepseek is None:
            try:
                from openai import OpenAI
                self._deepseek = OpenAI(
                    api_key=self.config.deepseek_api_key,
                    base_url="https://api.deepseek.com"
                )
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Run: pip install openai"
                )
        return self._deepseek

    def get_llama(self, model_path: str = None, n_ctx: int = 4096):
        """
        Get Llama model instance (lazy initialization).

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size

        Returns:
            Llama instance
        """
        if self._llama is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python not installed. "
                    "Run: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python"
                )

            if model_path is None:
                # Default model path
                model_path = str(Path(__file__).parent.parent / "models" /
                                "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    "Download from: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
                )

            print(f"Loading Llama model from: {model_path}")
            start_time = time.time()

            self._llama = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=-1,  # Use Metal on Mac
                verbose=False
            )

            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f}s")

        return self._llama

    def call_claude(
        self,
        prompt: str,
        model: str = "claude-haiku-3-5-20250110",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        system: str = None
    ) -> str:
        """
        Call Claude API.

        Args:
            prompt: User prompt
            model: Model name (haiku, sonnet, opus)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system: Optional system prompt

        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }

        if system:
            kwargs["system"] = system

        response = self.claude.messages.create(**kwargs)
        return response.content[0].text

    def call_deepseek_r1(
        self,
        prompt: str,
        model: str = "deepseek-reasoner",
        max_tokens: int = 4000,
        return_reasoning: bool = False
    ) -> str:
        """
        Call DeepSeek R1 API.

        Args:
            prompt: User prompt
            model: Model name (deepseek-reasoner or deepseek-chat)
            max_tokens: Maximum tokens to generate
            return_reasoning: If True, return dict with reasoning and content

        Returns:
            Generated text (or dict if return_reasoning=True)
        """
        response = self.deepseek.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )

        message = response.choices[0].message

        if return_reasoning and model == "deepseek-reasoner":
            return {
                "reasoning": getattr(message, "reasoning_content", ""),
                "content": message.content
            }
        else:
            return message.content

    def call_llama(
        self,
        prompt: str,
        model_path: str = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        system: str = "You are a helpful assistant."
    ) -> str:
        """
        Call local Llama model.

        Args:
            prompt: User prompt
            model_path: Optional path to model file
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system: System prompt

        Returns:
            Generated text
        """
        llama = self.get_llama(model_path)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        response = llama.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response['choices'][0]['message']['content']


# Helper functions for common operations

def save_json(data: Any, filepath: str):
    """Save data as JSON"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved to {filepath}")


def load_json(filepath: str) -> Any:
    """Load JSON data"""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_section(title: str, width: int = 60):
    """Pretty print section headers"""
    print(f"\n{'='*width}")
    print(f" {title}")
    print(f"{'='*width}\n")


if __name__ == "__main__":
    # Test API clients
    print_section("Testing API Client Initialization")

    clients = APIClients()

    print("✓ APIClients initialized")
    print(f"✓ Config loaded from: {clients.config.config_path}")

    # Test config access
    print(f"✓ Claude API key present: {bool(clients.config.claude_api_key)}")
    print(f"✓ DeepSeek API key present: {bool(clients.config.deepseek_api_key)}")

    print("\nNote: Actual API calls require network connectivity.")
    print("Run individual test scripts once network is available.")
