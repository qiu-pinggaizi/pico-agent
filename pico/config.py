"""YAML configuration management for Pico Agent.

Loads config from ~/.pico-agent/config.yaml, with environment variable overrides.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".pico-agent"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
MEMORY_FILE = CONFIG_DIR / "memory.md"
DB_FILE = CONFIG_DIR / "sessions.db"

DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
    },
    "agent": {
        "max_turns": 50,
        "max_tokens": 128000,
        "compression_threshold": 0.80,
    },
    "servers": {},
    "default_server": "",
}

# Environment variable mapping
# PICO_* vars always take precedence (applied last if both exist)
ENV_OVERRIDES: dict[str, str] = {
    "OPENAI_API_KEY": "model.api_key",
    "OPENAI_BASE_URL": "model.base_url",
    "ANTHROPIC_API_KEY": "model.api_key",
    "ANTHROPIC_BASE_URL": "model.base_url",
    "PICO_API_KEY": "model.api_key",
    "PICO_MODEL": "model.model",
    "PICO_BASE_URL": "model.base_url",
}


def _resolve_env_vars(value: str) -> str:
    """Resolve ${VAR_NAME} patterns in string values."""
    def replacer(match: re.Match) -> str:
        env_name = match.group(1)
        return os.environ.get(env_name, match.group(0))

    return re.sub(r"\$\{(\w+)\}", replacer, value)


def _walk_and_resolve(obj: Any) -> Any:
    """Recursively resolve environment variables in config values."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: _walk_and_resolve(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_walk_and_resolve(item) for item in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, returning new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested(d: dict, path: str, value: str) -> None:
    """Set a nested dict value by dot-separated path."""
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


@dataclass
class Config:
    """Pico Agent configuration.

    Attributes:
        data: Raw configuration dictionary.
        provider: LLM provider name (openai | anthropic).
        model: Model identifier string.
        base_url: API base URL.
        api_key: API authentication key.
        max_turns: Maximum tool-call turns per interaction.
        max_tokens: Token budget for context window.
        compression_threshold: Fraction of max_tokens that triggers compression.
        servers: Remote server configurations.
        default_server: Default server name for remote operations.
    """

    _VALID_PROVIDERS = ("openai", "anthropic")

    data: dict[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values and warn on suspicious settings."""
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")
        if self.max_turns <= 0:
            raise ValueError(f"max_turns must be > 0, got {self.max_turns}")
        if not (0 <= self.temperature <= 2):
            raise ValueError(f"temperature must be between 0 and 2, got {self.temperature}")
        provider = self.provider
        if provider not in self._VALID_PROVIDERS:
            raise ValueError(
                f"provider must be one of {self._VALID_PROVIDERS}, got '{provider}'"
            )

    @property
    def provider(self) -> str:
        return self.data.get("model", {}).get("provider", "openai")

    @property
    def model(self) -> str:
        return self.data.get("model", {}).get("model", "gpt-4o-mini")

    @property
    def base_url(self) -> str:
        return self.data.get("model", {}).get("base_url", "https://api.openai.com/v1")

    @property
    def api_key(self) -> str:
        return self.data.get("model", {}).get("api_key", "")

    @property
    def max_turns(self) -> int:
        return int(self.data.get("agent", {}).get("max_turns", 50))

    @property
    def max_tokens(self) -> int:
        return int(self.data.get("agent", {}).get("max_tokens", 128000))

    @property
    def compression_threshold(self) -> float:
        return float(self.data.get("agent", {}).get("compression_threshold", 0.80))

    @property
    def temperature(self) -> float:
        return float(self.data.get("model", {}).get("temperature", 1.0))

    @property
    def servers(self) -> dict[str, Any]:
        return self.data.get("servers", {})

    @property
    def default_server(self) -> str:
        return self.data.get("default_server", "")

    @property
    def base_dir(self) -> str:
        """Base directory for pico-agent data (datasets, repos, etc.)."""
        return str(CONFIG_DIR)

    @property
    def work_dir(self) -> str:
        """Working directory for commands (default: cwd where pico-agent was launched)."""
        return self.data.get("work_dir", "")


def load_config(config_path: str | Path | None = None, *, strict: bool = False) -> Config:
    """Load configuration from YAML file, with env var overrides.

    Args:
        config_path: Path to config YAML file. Uses default if None.
        strict: If True, raise on invalid YAML instead of silently falling back.
    """
    if config_path is None:
        config_path = CONFIG_FILE
    config_path = Path(config_path)

    # Start with defaults
    data = _deep_merge({}, DEFAULT_CONFIG)

    # Load from file if it exists
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_data = yaml.safe_load(f) or {}
            data = _deep_merge(data, file_data)
            logger.info("Loaded config from %s", config_path)
        except Exception as e:
            if strict:
                raise ValueError(f"Invalid config file {config_path}: {e}") from e
            logger.warning("Failed to load config from %s: %s", config_path, e)
    else:
        logger.info("Config file %s not found, using defaults", config_path)

    # Apply environment variable overrides
    for env_var, config_path_str in ENV_OVERRIDES.items():
        env_value = os.environ.get(env_var)
        if env_value:
            _set_nested(data, config_path_str, env_value)
            logger.debug("Applied env override: %s -> %s", env_var, config_path_str)

    # Resolve any remaining ${VAR} references
    data = _walk_and_resolve(data)

    # Ensure config directory exists
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    return Config(data=data)


# ---------------------------------------------------------------------------
# Module-level config cache
# ---------------------------------------------------------------------------

_current_config: Config | None = None


def get_config(config_path: Path | str | None = None) -> Config:
    """Return the cached Config instance, loading it on first call.

    Subsequent calls return the same instance unless *config_path* is provided
    (which forces a reload).
    """
    global _current_config
    if _current_config is None or config_path is not None:
        _current_config = load_config(config_path)
    return _current_config
