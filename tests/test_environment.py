"""Tests for the sts-agent game environment wrapper."""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Unit tests for _format_state helper
# ---------------------------------------------------------------------------


class TestFormatState(unittest.TestCase):
    def test_formats_player_window(self):
        from environment.game_env import _format_state

        windows = [
            {"window_title": "Player", "content": "Health: 80/80\nEnergy: 3", "error": None},
        ]
        result = _format_state(windows)
        assert "=== Player ===" in result
        assert "Health: 80/80" in result

    def test_formats_multiple_windows(self):
        from environment.game_env import _format_state

        windows = [
            {"window_title": "Player", "content": "Health: 80/80\nEnergy: 3", "error": None},
            {"window_title": "Hand", "content": "1:Strike 1\n2:Defend 1", "error": None},
        ]
        result = _format_state(windows)
        assert "Player" in result
        assert "Hand" in result
        assert "Strike" in result

    def test_marks_unavailable_window(self):
        from environment.game_env import _format_state

        windows = [
            {"window_title": "Monster", "content": "", "error": "Window not found"},
        ]
        result = _format_state(windows)
        assert "unavailable" in result
        assert "Monster" in result

    def test_skips_empty_content_no_error(self):
        from environment.game_env import _format_state

        windows = [
            {"window_title": "Map", "content": "", "error": None},
            {"window_title": "Player", "content": "Health: 50/80", "error": None},
        ]
        result = _format_state(windows)
        # Map has no content and no error — should be omitted from output
        assert "=== Map ===" not in result
        assert "Player" in result


# ---------------------------------------------------------------------------
# compute_reward unit tests
# ---------------------------------------------------------------------------


class TestComputeReward(unittest.TestCase):
    def _env(self):
        from environment.game_env import StsAgentEnv

        env = StsAgentEnv.__new__(StsAgentEnv)
        env._won = False
        env._floors_cleared = 0
        env._enemies_killed = 0
        return env

    def test_win_phrase_gives_positive_reward(self):
        env = self._env()
        reward, done = env._compute_reward("You won!")
        assert reward > 0
        assert done is True

    def test_loss_phrase_gives_negative_reward(self):
        env = self._env()
        reward, done = env._compute_reward("You have lost!")
        assert reward < 0
        assert done is True

    def test_enemy_killed_gives_bonus(self):
        env = self._env()
        reward, done = env._compute_reward("The Cultist has been defeated.")
        assert reward > 0
        assert done is False

    def test_floor_advance_gives_bonus(self):
        env = self._env()
        reward, done = env._compute_reward("You have entered floor 3.")
        assert reward > 0
        assert done is False

    def test_invalid_input_gives_penalty(self):
        env = self._env()
        reward, done = env._compute_reward("You have to type a number.")
        assert reward < 0
        assert done is False

    def test_player_hp_nonzero_no_death_penalty(self):
        env = self._env()
        state = "=== Player ===\nHealth: 55/80\nEnergy: 3"
        reward, done = env._compute_reward(state)
        # Non-zero HP should not trigger the death-penalty path
        assert done is False

    def test_player_hp_zero_gives_penalty(self):
        env = self._env()
        # Simulate Player window content with 0 HP
        state = "=== Player ===\nHealth: 0/80\nEnergy: 3"
        reward, done = env._compute_reward(state)
        assert reward < 0
        assert done is True


# ---------------------------------------------------------------------------
# StsAgentEnv lifecycle tests (CLI calls mocked)
# ---------------------------------------------------------------------------


class TestStsAgentEnvLifecycle(unittest.TestCase):
    def _multi_window_response(self):
        return {
            "windows": [
                {"window_title": "Player", "content": "Health: 80/80\nEnergy: 3", "error": None},
                {"window_title": "Hand", "content": "1:Strike 1\n2:Defend 1", "error": None},
                {"window_title": "Monster", "content": "Cultist HP: 50/50\nIntent: Ritual", "error": None},
                {"window_title": "Choices", "content": "", "error": None},
                {"window_title": "Map", "content": "Floor 1", "error": None},
            ]
        }

    @patch("environment.game_env.StsAgentEnv._run_tool")
    def test_reset_reads_windows(self, mock_run_tool):
        from environment.game_env import StsAgentEnv

        mock_run_tool.return_value = self._multi_window_response()
        env = StsAgentEnv()
        state = env.reset()

        assert isinstance(state, str)
        assert "Player" in state
        assert "Health: 80/80" in state
        mock_run_tool.assert_called_once()

    @patch("environment.game_env.StsAgentEnv._run_tool")
    def test_step_calls_execute_then_read(self, mock_run_tool):
        from environment.game_env import StsAgentEnv

        mock_run_tool.return_value = self._multi_window_response()
        env = StsAgentEnv()
        env.reset()

        mock_run_tool.reset_mock()
        mock_run_tool.return_value = self._multi_window_response()

        state, reward, done, info = env.step("1")

        # step() calls _execute_and_read which calls _run_tool once
        mock_run_tool.assert_called_once()
        args, _ = mock_run_tool.call_args
        cmd_args = args[0]
        assert "--execute" in cmd_args
        assert "1" in cmd_args
        assert "--read-window" in cmd_args

        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert info["turn"] == 1

    def test_close_is_noop(self):
        from environment.game_env import StsAgentEnv

        env = StsAgentEnv()
        env.close()  # should not raise
        env.close()  # idempotent

    @patch("environment.game_env.StsAgentEnv._run_tool")
    def test_max_turns_ends_episode(self, mock_run_tool):
        from environment.game_env import StsAgentEnv

        mock_run_tool.return_value = self._multi_window_response()
        env = StsAgentEnv(max_turns=1)
        env.reset()

        mock_run_tool.return_value = self._multi_window_response()
        _, _, done, info = env.step("1")

        assert done is True
        assert info["turn"] == 1

    @patch("environment.game_env.StsAgentEnv._run_tool")
    def test_step_after_done_returns_immediately(self, mock_run_tool):
        from environment.game_env import StsAgentEnv

        mock_run_tool.return_value = self._multi_window_response()
        env = StsAgentEnv(max_turns=1)
        env.reset()

        mock_run_tool.return_value = self._multi_window_response()
        env.step("1")  # triggers max_turns, done=True

        mock_run_tool.reset_mock()
        _, _, done, _ = env.step("end")  # should not call _run_tool again
        mock_run_tool.assert_not_called()
        assert done is True

    @patch("environment.game_env.StsAgentEnv._run_tool")
    def test_run_tool_timeout_returns_empty_state(self, mock_run_tool):
        from environment.game_env import StsAgentEnv

        mock_run_tool.return_value = {}  # simulate timeout / parse failure
        env = StsAgentEnv()
        state = env.reset()
        # Should not raise; returns empty string
        assert isinstance(state, str)

    @patch("environment.game_env.StsAgentEnv._run_tool")
    def test_info_tracks_enemies_killed(self, mock_run_tool):
        from environment.game_env import StsAgentEnv

        # First reset
        mock_run_tool.return_value = self._multi_window_response()
        env = StsAgentEnv()
        env.reset()

        # Step with an "enemy defeated" message
        kill_response = {
            "windows": [
                {
                    "window_title": "Player",
                    "content": "Health: 70/80\nEnergy: 3",
                    "error": None,
                },
                {
                    "window_title": "Hand",
                    "content": "The Cultist has been defeated.\n1:Strike 1",
                    "error": None,
                },
            ]
        }
        mock_run_tool.return_value = kill_response
        _, _, _, info = env.step("1")
        assert info["enemies_killed"] >= 1


if __name__ == "__main__":
    unittest.main()

