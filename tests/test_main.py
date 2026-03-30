"""Tests for the main module (per-game file logging)."""

from __future__ import annotations

import logging
import os
import tempfile
import unittest

from main import _DEFAULT_LOG_FILENAME, _configure_file_logging


class TestConfigureFileLogging(unittest.TestCase):
    """Tests for _configure_file_logging."""

    def setUp(self) -> None:
        self.log_path = os.path.join(tempfile.gettempdir(), "test_sts_game.log")
        self._original_handlers = logging.getLogger().handlers[:]
        self._original_level = logging.getLogger().level
        # Ensure root logger propagates INFO messages to handlers.
        logging.getLogger().setLevel(logging.INFO)

    def tearDown(self) -> None:
        # Remove any file handlers added during the test.
        root = logging.getLogger()
        for handler in list(root.handlers):
            if isinstance(handler, logging.FileHandler):
                root.removeHandler(handler)
                handler.close()
        # Restore original handlers and level.
        root.handlers = self._original_handlers
        logging.getLogger().setLevel(self._original_level)
        # Clean up temporary log file.
        if os.path.exists(self.log_path):
            os.unlink(self.log_path)

    def test_creates_log_file(self) -> None:
        """Calling _configure_file_logging should create the log file on disk."""
        _configure_file_logging(self.log_path)
        self.assertTrue(os.path.exists(self.log_path))

    def test_log_messages_written_to_file(self) -> None:
        """Log records emitted after setup should appear in the file."""
        _configure_file_logging(self.log_path)
        test_logger = logging.getLogger("test_sts_main")
        test_logger.info("sentinel message")
        # Flush all handlers so the write is flushed to disk.
        for handler in logging.getLogger().handlers:
            handler.flush()
        with open(self.log_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("sentinel message", content)

    def test_file_overwritten_on_second_call(self) -> None:
        """A second call to _configure_file_logging should overwrite the file."""
        test_logger = logging.getLogger("test_sts_main")

        _configure_file_logging(self.log_path)
        test_logger.info("first game message")
        for handler in logging.getLogger().handlers:
            handler.flush()

        # Simulate a new game starting: call configure again.
        _configure_file_logging(self.log_path)
        test_logger.info("second game message")
        for handler in logging.getLogger().handlers:
            handler.flush()

        with open(self.log_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertNotIn("first game message", content)
        self.assertIn("second game message", content)

    def test_no_duplicate_file_handlers(self) -> None:
        """Repeated calls must not accumulate multiple FileHandlers."""
        _configure_file_logging(self.log_path)
        _configure_file_logging(self.log_path)
        _configure_file_logging(self.log_path)
        file_handlers = [
            h
            for h in logging.getLogger().handlers
            if isinstance(h, logging.FileHandler)
        ]
        self.assertEqual(len(file_handlers), 1)

    def test_default_log_filename_constant(self) -> None:
        """_DEFAULT_LOG_FILENAME should be a non-empty string."""
        self.assertIsInstance(_DEFAULT_LOG_FILENAME, str)
        self.assertTrue(_DEFAULT_LOG_FILENAME)
