"""Tests for the game environment wrapper."""
from __future__ import annotations

import threading
import time
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_proc(stdout_lines: list[str], returncode: int = 0):
    """Build a mock subprocess.Popen compatible object."""
    proc = MagicMock()
    proc.poll.return_value = returncode  # process has exited
    # stdin is writable
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.flush = MagicMock()
    proc.stdin.close = MagicMock()
    proc.terminate = MagicMock()
    proc.wait = MagicMock()
    # stdout is an iterable of lines
    proc.stdout = iter(stdout_lines)
    return proc


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


class TestStripAnsi(unittest.TestCase):
    def test_strips_ansi_codes(self):
        from environment.game_env import _strip_ansi

        assert _strip_ansi("\x1b[31mRed\x1b[0m") == "Red"

    def test_strips_angle_bracket_markup(self):
        from environment.game_env import _strip_ansi

        assert _strip_ansi("<red>Enemy</red>") == "Enemy"

    def test_plain_text_unchanged(self):
        from environment.game_env import _strip_ansi

        plain = "Pick an option:\n1. Strike\n2. Defend"
        assert _strip_ansi(plain) == plain


class TestLooksComplete(unittest.TestCase):
    def test_numbered_option_detected(self):
        from environment.game_env import SlayTheSpireEnv

        text = "What do you do?\n1. Strike\n2. Defend"
        assert SlayTheSpireEnv._looks_complete(text) is True

    def test_type_prompt_detected(self):
        from environment.game_env import SlayTheSpireEnv

        text = "Type your choice."
        assert SlayTheSpireEnv._looks_complete(text) is True

    def test_empty_string_not_complete(self):
        from environment.game_env import SlayTheSpireEnv

        assert SlayTheSpireEnv._looks_complete("") is False


# ---------------------------------------------------------------------------
# compute_reward unit tests
# ---------------------------------------------------------------------------


class TestComputeReward(unittest.TestCase):
    def _env(self):
        from environment.game_env import SlayTheSpireEnv

        return SlayTheSpireEnv.__new__(SlayTheSpireEnv)

    def test_win_phrase_gives_positive_reward(self):
        env = self._env()
        reward, done = env._compute_reward("You won!")
        assert reward > 0
        assert done is True

    def test_loss_phrase_gives_negative_reward(self):
        env = self._env()
        env._won = False
        reward, done = env._compute_reward("You have lost!")
        assert reward < 0
        assert done is True

    def test_enemy_killed_gives_bonus(self):
        env = self._env()
        env._enemies_killed = 0
        env._floors_cleared = 0
        env._won = False
        reward, done = env._compute_reward("The Cultist has been defeated.")
        assert reward > 0
        assert done is False

    def test_floor_advance_gives_bonus(self):
        env = self._env()
        env._enemies_killed = 0
        env._floors_cleared = 0
        env._won = False
        reward, done = env._compute_reward("You have entered floor 3.")
        assert reward > 0
        assert done is False

    def test_invalid_input_gives_penalty(self):
        env = self._env()
        env._enemies_killed = 0
        env._floors_cleared = 0
        env._won = False
        reward, done = env._compute_reward("You have to type a number.")
        assert reward < 0
        assert done is False


# ---------------------------------------------------------------------------
# SlayTheSpireEnv lifecycle (subprocess mocked)
# ---------------------------------------------------------------------------


class TestEnvLifecycle(unittest.TestCase):
    @patch("environment.game_env.subprocess.Popen")
    def test_reset_starts_process(self, MockPopen):
        from environment.game_env import SlayTheSpireEnv

        lines = [
            "Slay the Spire\n",
            "Pick the Character you want to play.\n",
            "1. Silent\n",
            "2. Ironclad\n",
            "3. Defect\n",
        ]
        MockPopen.return_value = _make_fake_proc(lines)

        env = SlayTheSpireEnv(game_script="fake/main.py", read_timeout=1.0)
        state = env.reset()
        assert isinstance(state, str)
        env.close()

    @patch("environment.game_env.subprocess.Popen")
    def test_step_sends_action(self, MockPopen):
        from environment.game_env import SlayTheSpireEnv

        lines = [
            "Pick your character.\n",
            "1. Silent\n",
            "You chose the Silent.\n",
        ]
        proc = _make_fake_proc(lines)
        MockPopen.return_value = proc

        env = SlayTheSpireEnv(game_script="fake/main.py", read_timeout=1.0)
        env.reset()
        state, reward, done, info = env.step("1")
        proc.stdin.write.assert_called()
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        env.close()

    def test_close_is_idempotent(self):
        from environment.game_env import SlayTheSpireEnv

        env = SlayTheSpireEnv(game_script="fake/main.py")
        env.close()  # no subprocess was started
        env.close()  # second call must not raise

    @patch("environment.game_env.subprocess.Popen")
    def test_max_turns_ends_episode(self, MockPopen):
        from environment.game_env import SlayTheSpireEnv

        proc = _make_fake_proc(["Game output\n1. Option\n"])
        MockPopen.return_value = proc

        env = SlayTheSpireEnv(game_script="fake/main.py", max_turns=1, read_timeout=0.5)
        env.reset()
        _, _, done, info = env.step("1")
        assert done is True
        assert info["turn"] == 1
        env.close()


if __name__ == "__main__":
    unittest.main()
