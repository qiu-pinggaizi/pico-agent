"""Tests for --version flag, --config validation, piped stdin, and edge cases.

Round 2 testing — verifying bug fixes from developer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pico.cli import main


# ---------------------------------------------------------------------------
# --version / -V
# ---------------------------------------------------------------------------

class TestVersionFlag:
    """Test --version and -V flags."""

    def test_version_long_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--version should print 'pico-agent 0.1.0' and exit 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out.strip()
        assert out == "pico-agent 0.1.0"

    def test_version_short_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """-V should print 'pico-agent 0.1.0' and exit 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["-V"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out.strip()
        assert out == "pico-agent 0.1.0"

    def test_version_with_verbose(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--version --verbose should still print version and exit."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version", "--verbose"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out.strip()
        assert out == "pico-agent 0.1.0"

    def test_version_with_session(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--version --session test should still print version and exit."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version", "--session", "test"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out.strip()
        assert out == "pico-agent 0.1.0"

    def test_version_with_config(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--version --config X should still print version (version checked first)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version", "--config", "/nonexistent.yaml"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out.strip()
        assert out == "pico-agent 0.1.0"

    def test_version_in_help_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Help text should mention version (v0.1.0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "v0.1.0" in out


# ---------------------------------------------------------------------------
# --config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    """Test --config flag error handling."""

    def test_config_nonexistent_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--config /nonexistent.yaml should error and exit 1."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--config", "/nonexistent.yaml"])
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "not found" in err.lower()

    def test_config_valid_file(self, tmp_path: Path) -> None:
        """--config with a valid YAML file should succeed (single-shot mode)."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "model:\n  provider: openai\n  model: test-model\n  api_key: test-key\n"
        )
        # We can't fully run without API, but config loading should succeed
        with patch("pico.agent.AIAgent") as mock_agent_cls:
            with patch("pico.config.load_config") as mock_cfg:
                with patch("pico.session.SessionStore"):
                    with patch("pico.memory.Memory"):
                        with patch("pico.tools.registry.ToolRegistry"):
                            with patch("pico.tools.discover_and_register"):
                                mock_cfg.return_value = MagicMock()
                                mock_agent = MagicMock()
                                mock_agent.run.return_value = "ok"
                                mock_agent_cls.return_value = mock_agent
                                try:
                                    main(["--config", str(config_file), "hello"])
                                except SystemExit:
                                    pass
        # If we get here without "not found" error, config was accepted
        mock_cfg.assert_called_once_with(str(config_file), strict=True)

    def test_config_equals_syntax(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--config=/nonexistent.yaml should error the same way."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--config=/nonexistent.yaml"])
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "not found" in err.lower()

    def test_config_invalid_yaml(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """--config with invalid YAML should warn but NOT exit with error (currently)."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{invalid yaml: [unclosed")
        # Current behavior: warns and continues with defaults
        # This is a potential issue (should it error out?)
        with patch("pico.agent.AIAgent") as mock_agent_cls:
            with patch("pico.session.SessionStore"):
                with patch("pico.memory.Memory"):
                    with patch("pico.tools.registry.ToolRegistry"):
                        with patch("pico.tools.discover_and_register"):
                            mock_agent = MagicMock()
                            mock_agent.run.return_value = "ok"
                            mock_agent_cls.return_value = mock_agent
                            try:
                                main(["--config", str(config_file), "hello"])
                            except SystemExit:
                                pass


# ---------------------------------------------------------------------------
# Piped stdin (single-shot)
# ---------------------------------------------------------------------------

class TestPipedStdin:
    """Test that piped stdin triggers single-shot mode, not REPL."""

    def test_piped_stdin_enters_single_shot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When stdin is piped (not a tty), should enter single-shot mode."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr(sys.stdin, "read", lambda: "hello world")

        with patch("pico.agent.AIAgent") as mock_agent_cls:
            with patch("pico.config.load_config") as mock_cfg:
                with patch("pico.session.SessionStore") as mock_ss:
                    with patch("pico.memory.Memory"):
                        with patch("pico.tools.registry.ToolRegistry"):
                            with patch("pico.tools.discover_and_register"):
                                mock_cfg.return_value = MagicMock()
                                mock_agent = MagicMock()
                                mock_agent.run.return_value = "test response"
                                mock_agent_cls.return_value = mock_agent
                                with pytest.raises(SystemExit) as exc_info:
                                    main([])
                                assert exc_info.value.code == 0
                                mock_agent.run.assert_called_once_with("hello world")

    def test_piped_empty_stdin_exits_cleanly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty piped stdin should still exit cleanly."""
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        monkeypatch.setattr(sys.stdin, "read", lambda: "")

        with patch("pico.agent.AIAgent") as mock_agent_cls:
            with patch("pico.config.load_config") as mock_cfg:
                with patch("pico.session.SessionStore") as mock_ss:
                    with patch("pico.memory.Memory"):
                        with patch("pico.tools.registry.ToolRegistry"):
                            with patch("pico.tools.discover_and_register"):
                                mock_cfg.return_value = MagicMock()
                                mock_ss_instance = MagicMock()
                                mock_ss.return_value = mock_ss_instance
                                mock_agent = MagicMock()
                                mock_agent_cls.return_value = mock_agent
                                with pytest.raises(SystemExit) as exc_info:
                                    main([])
                                assert exc_info.value.code == 0
                                # Empty piped stdin should NOT call agent.run
                                mock_agent.run.assert_not_called()


# ---------------------------------------------------------------------------
# Empty string message
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_string_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        """pico-agent "" should reject empty message with helpful error."""
        with pytest.raises(SystemExit) as exc_info:
            main([""])
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "empty" in err.lower()

    def test_version_before_config_check(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--version should be processed before --config validation."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version", "--config", "/nonexistent.yaml"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out.strip()
        assert "pico-agent 0.1.0" in out
