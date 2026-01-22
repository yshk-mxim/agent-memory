"""
Configuration loader for RDIC project.

Loads API keys and settings from env.json.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for RDIC project"""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to env.json file. If None, searches in parent directories.
        """
        if config_path is None:
            # Search for env.json in current dir and parent dirs
            config_path = self._find_config_file()

        self.config_path = config_path
        self._config = self._load_config()

    def _find_config_file(self) -> str:
        """Find env.json in current or parent directories"""
        current = Path.cwd()

        # Check current directory and up to 3 parent levels
        for _ in range(4):
            env_file = current / "env.json"
            if env_file.exists():
                return str(env_file)
            current = current.parent

        # If not found, default to ../env.json
        return str(Path(__file__).parent.parent / "env.json")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                "Please create env.json with API keys."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.config_path}: {e}")

    @property
    def claude_api_key(self) -> str:
        """Get Claude/Anthropic API key"""
        key = self._config.get("claude_api_key")
        if not key:
            raise ValueError("claude_api_key not found in env.json")
        return key

    @property
    def deepseek_api_key(self) -> str:
        """Get DeepSeek API key"""
        key = self._config.get("deepseek_api_key")
        if not key:
            raise ValueError("deepseek_api_key not found in env.json")
        return key

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self._config.get(key, default)

    def __repr__(self) -> str:
        return f"Config(path={self.config_path}, keys={list(self._config.keys())})"


# Global config instance
_config = None


def get_config() -> Config:
    """Get global configuration instance (singleton pattern)"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config():
    """Reset global configuration (useful for testing)"""
    global _config
    _config = None


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print(f"Configuration loaded from: {config.config_path}")
    print(f"Claude API key present: {bool(config.claude_api_key)}")
    print(f"DeepSeek API key present: {bool(config.deepseek_api_key)}")
