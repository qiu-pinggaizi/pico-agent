"""Tests for pico.cli — CLI entry point, argument parsing, shortcuts."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pico.cli import main


class TestHelpFlags:
    """Test that --help prints help and exits 0."""

    def test_main_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main(['--help']) prints help text and calls sys.exit(0)."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "Pico Agent" in out
        assert "USAGE" in out

    def test_main_help_short_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main(['-h']) prints help and exits 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["-h"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "USAGE" in out


class TestSubcommandHelp:
    """Test --help for each subcommand."""

    def test_train_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["train", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "data_dir" in out

    def test_eval_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["eval", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "model_path" in out

    def test_search_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["search", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "query" in out

    def test_download_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["download", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "query_or_url" in out

    def test_clone_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["clone", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "url" in out

    def test_infer_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["infer", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "model_path" in out


class TestErrorCases:
    """Test error paths that should exit non-zero."""

    def test_train_nonexistent_dir(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main(['train', '/nonexistent']) should exit 1 with error message."""
        with pytest.raises(SystemExit) as exc_info:
            main(["train", "/nonexistent"])
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "not found" in err.lower() or "error" in err.lower()

    def test_eval_nonexistent_file(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main(['eval', '/nonexistent.pt']) should exit 1 with error."""
        with pytest.raises(SystemExit) as exc_info:
            main(["eval", "/nonexistent.pt"])
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "not found" in err.lower() or "error" in err.lower()

    def test_train_missing_args(self) -> None:
        """main(['train']) with no args should error."""
        with pytest.raises((SystemExit, SystemError)):
            main(["train"])


class TestVerboseFlag:
    """Test --verbose flag handling."""

    def test_verbose_sets_debug_logging(self) -> None:
        """--verbose should set logging to DEBUG."""
        # AIAgent, load_config, etc. are lazy imports inside main()
        # so we must patch at the source module level.
        with patch("pico.cli._setup_logging") as mock_setup:
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
                                        main(["--verbose", "hello"])
                                    except SystemExit:
                                        pass
            mock_setup.assert_called_once_with(True)


class TestSessionFlag:
    """Test --session flag handling."""

    def test_session_flag_accepted(self) -> None:
        """--session flag should be accepted and parsed."""
        with patch("pico.cli._setup_logging") as mock_setup:
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
                                        main(["--session", "test", "hello"])
                                    except SystemExit:
                                        pass
            mock_setup.assert_called_once_with(False)
