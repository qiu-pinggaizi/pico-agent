"""Tests for Config loading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from pico.config import Config, get_config, load_config


class TestDefaultConfig:
    """test_default_config — load_config() with no file returns defaults."""

    def test_default_config(self, tmp_path: Path) -> None:
        """Pass a non-existent path so no file is loaded."""
        cfg = load_config(config_path=tmp_path / "does_not_exist.yaml")
        assert cfg.model == "gpt-4o-mini"
        assert cfg.provider == "openai"
        assert cfg.max_turns == 50
        assert cfg.max_tokens == 128000
        assert isinstance(cfg.data, dict)


class TestCustomConfig:
    """test_custom_config — write a YAML to tmp_path, load it, verify values."""

    def test_custom_config(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "custom.yaml"
        custom_data = {
            "model": {
                "provider": "anthropic",
                "model": "claude-3-opus",
                "api_key": "sk-test-123",
                "base_url": "https://custom.api.test/v1",
            },
            "agent": {
                "max_turns": 10,
                "max_tokens": 64000,
                "compression_threshold": 0.7,
            },
        }
        cfg_path.write_text(yaml.dump(custom_data), encoding="utf-8")

        cfg = load_config(config_path=cfg_path)
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-3-opus"
        assert cfg.api_key == "sk-test-123"
        assert cfg.max_turns == 10
        assert cfg.max_tokens == 64000
        assert cfg.compression_threshold == pytest.approx(0.7)


class TestGetConfigCaching:
    """test_get_config_caching — second call returns same instance when no path given."""

    def test_get_config_caching(self) -> None:
        """Clear the module-level cache, call get_config() twice with no path."""
        import pico.config as cfg_module

        original = cfg_module._current_config
        cfg_module._current_config = None
        try:
            c1 = get_config()
            c2 = get_config()
            assert c1 is c2
        finally:
            cfg_module._current_config = original
