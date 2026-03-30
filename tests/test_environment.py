"""Tests for the sts-agent game environment wrapper."""
from __future__ import annotations

import json
import os
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


# ---------------------------------------------------------------------------
# TextTheSpireEnv tests (file-system calls mocked / using tmp dirs)
# ---------------------------------------------------------------------------


class TestTextTheSpireEnv(unittest.TestCase):
    """Tests for the cross-platform TextTheSpireEnv back-end."""

    def _write_window_files(self, tmp_dir: str, windows: dict[str, str]) -> None:
        """Helper: write window content to ``<tmp_dir>/<name>.txt``."""
        for name, content in windows.items():
            with open(os.path.join(tmp_dir, f"{name}.txt"), "w", encoding="utf-8") as fh:
                fh.write(content)

    def test_reset_reads_window_files(self):
        import tempfile
        from environment.game_env import TextTheSpireEnv

        with tempfile.TemporaryDirectory() as tmp:
            self._write_window_files(tmp, {
                "Player": "Health: 80/80\nEnergy: 3",
                "Hand": "1:Strike 1\n2:Defend 1",
                "Monster": "Cultist HP: 50/50\nIntent: Ritual",
                "Choices": "",
                "Map": "Floor 1",
            })
            env = TextTheSpireEnv(
                state_dir=tmp,
                input_file=os.path.join(tmp, "sts_input.txt"),
                windows=["Player", "Hand", "Monster", "Map"],
            )
            state = env.reset()

        assert "Player" in state
        assert "Health: 80/80" in state
        assert "Hand" in state
        assert "Strike" in state

    def test_step_writes_command_file(self):
        import tempfile
        from environment.game_env import TextTheSpireEnv

        with tempfile.TemporaryDirectory() as tmp:
            self._write_window_files(tmp, {
                "Player": "Health: 80/80\nEnergy: 3",
                "Hand": "1:Strike 1",
            })
            input_path = os.path.join(tmp, "sts_input.txt")
            env = TextTheSpireEnv(
                state_dir=tmp,
                input_file=input_path,
                windows=["Player", "Hand"],
            )
            env.reset()
            env.step("end")

            with open(input_path, encoding="utf-8") as fh:
                written = fh.read()
        assert written.strip() == "end"

    def test_missing_window_file_returns_error_in_state(self):
        import tempfile
        from environment.game_env import TextTheSpireEnv

        with tempfile.TemporaryDirectory() as tmp:
            # Only write Player; Monster is missing
            self._write_window_files(tmp, {"Player": "Health: 70/80\nEnergy: 3"})
            env = TextTheSpireEnv(
                state_dir=tmp,
                input_file=os.path.join(tmp, "sts_input.txt"),
                windows=["Player", "Monster"],
                command_timeout=0.1,  # short timeout so test runs fast
            )
            state = env.reset()

        assert "Player" in state
        assert "Monster" in state
        assert "unavailable" in state

    def test_max_turns_ends_episode(self):
        import tempfile
        from environment.game_env import TextTheSpireEnv

        with tempfile.TemporaryDirectory() as tmp:
            self._write_window_files(tmp, {"Player": "Health: 80/80\nEnergy: 3"})
            env = TextTheSpireEnv(
                state_dir=tmp,
                input_file=os.path.join(tmp, "sts_input.txt"),
                windows=["Player"],
                max_turns=1,
            )
            env.reset()
            _, _, done, info = env.step("1")

        assert done is True
        assert info["turn"] == 1

    def test_step_after_done_returns_immediately(self):
        import tempfile
        from environment.game_env import TextTheSpireEnv

        with tempfile.TemporaryDirectory() as tmp:
            self._write_window_files(tmp, {"Player": "Health: 80/80\nEnergy: 3"})
            input_path = os.path.join(tmp, "sts_input.txt")
            env = TextTheSpireEnv(
                state_dir=tmp,
                input_file=input_path,
                windows=["Player"],
                max_turns=1,
            )
            env.reset()
            env.step("1")  # triggers max_turns → done=True

            # Remove the input file so any write attempt would fail
            if os.path.exists(input_path):
                os.remove(input_path)

            # Step again — should not write a command or raise
            _, _, done, _ = env.step("end")

        assert done is True

    def test_close_is_noop(self):
        import tempfile
        from environment.game_env import TextTheSpireEnv

        with tempfile.TemporaryDirectory() as tmp:
            env = TextTheSpireEnv(
                state_dir=tmp,
                input_file=os.path.join(tmp, "sts_input.txt"),
            )
            env.close()  # should not raise


# ---------------------------------------------------------------------------
# make_env factory tests
# ---------------------------------------------------------------------------


class TestMakeEnv(unittest.TestCase):
    def test_default_returns_sts_agent_env(self):
        from environment.game_env import StsAgentEnv, make_env

        env = make_env({})
        assert isinstance(env, StsAgentEnv)

    def test_sts_agent_mode_explicit(self):
        from environment.game_env import StsAgentEnv, make_env

        env = make_env({"interface_mode": "sts_agent"})
        assert isinstance(env, StsAgentEnv)

    def test_text_the_spire_mode(self):
        import tempfile
        from environment.game_env import TextTheSpireEnv, make_env

        with tempfile.TemporaryDirectory() as tmp:
            env = make_env({
                "interface_mode": "text_the_spire",
                "text_the_spire_state_dir": tmp,
                "text_the_spire_input_file": os.path.join(tmp, "sts_input.txt"),
            })
        assert isinstance(env, TextTheSpireEnv)

    def test_unknown_mode_falls_back_to_sts_agent(self):
        from environment.game_env import StsAgentEnv, make_env

        env = make_env({"interface_mode": "unknown_value"})
        assert isinstance(env, StsAgentEnv)

    def test_communication_mod_mode(self):
        from environment.game_env import CommunicationModEnv, make_env

        env = make_env({"interface_mode": "communication_mod"})
        assert isinstance(env, CommunicationModEnv)

    def test_make_env_passes_windows_and_max_turns(self):
        from environment.game_env import StsAgentEnv, make_env

        env = make_env({"windows": ["Player", "Hand"], "max_turns": 50})
        assert isinstance(env, StsAgentEnv)
        assert env.windows == ["Player", "Hand"]
        assert env.max_turns == 50

    def test_make_env_text_the_spire_passes_state_dir(self):
        import tempfile
        from environment.game_env import TextTheSpireEnv, make_env

        with tempfile.TemporaryDirectory() as tmp:
            env = make_env({
                "interface_mode": "text_the_spire",
                "text_the_spire_state_dir": tmp,
                "text_the_spire_input_file": os.path.join(tmp, "sts_input.txt"),
                "windows": ["Player"],
                "max_turns": 10,
            })
            assert isinstance(env, TextTheSpireEnv)
            assert env.state_dir == tmp
            assert env.max_turns == 10


# ---------------------------------------------------------------------------
# CommunicationModEnv tests (stdin/stdout mocked with io.StringIO)
# ---------------------------------------------------------------------------


class TestCommunicationModEnv(unittest.TestCase):
    """Tests for the CommunicationModEnv Communication Mod back-end."""

    def _make_state_line(
        self,
        screen_type: str = "NONE",
        current_hp: int = 80,
        max_hp: int = 80,
        floor: int = 1,
        hand: list | None = None,
        monsters: list | None = None,
        choice_list: list | None = None,
    ) -> str:
        """Return a JSON line simulating a Communication Mod ready message."""
        import json

        combat_state: dict = {
            "player": {"block": 0, "energy": 3, "powers": []},
            "hand": hand or [],
            "monsters": monsters or [],
        }
        game_state: dict = {
            "screen_type": screen_type,
            "current_hp": current_hp,
            "max_hp": max_hp,
            "gold": 99,
            "floor": floor,
            "act": 1,
            "combat_state": combat_state,
            "choice_list": choice_list or [],
            "screen_state": {},
        }
        return json.dumps({"ready_for_command": True, "game_state": game_state}) + "\n"

    def _make_env(self, lines: list[str], max_turns: int = 200):
        """Build a CommunicationModEnv with mocked streams."""
        import io
        from environment.game_env import CommunicationModEnv

        in_stream = io.StringIO("".join(lines))
        out_stream = io.StringIO()
        env = CommunicationModEnv(
            max_turns=max_turns,
            input_stream=in_stream,
            output_stream=out_stream,
        )
        return env, out_stream

    # ------------------------------------------------------------------
    # reset / step lifecycle
    # ------------------------------------------------------------------

    def test_reset_returns_formatted_state(self):
        line = self._make_state_line(
            current_hp=80,
            max_hp=80,
            hand=[{"name": "Strike", "cost": 1}],
        )
        env, _ = self._make_env([line])
        state = env.reset()

        assert "=== Player ===" in state
        assert "Health: 80/80" in state
        assert "=== Hand ===" in state
        assert "Strike" in state

    def test_step_sends_json_command_and_reads_next_state(self):
        import io
        import json
        from environment.game_env import CommunicationModEnv

        state_line = self._make_state_line()
        next_line = self._make_state_line(current_hp=75, max_hp=80)

        in_stream = io.StringIO(state_line + next_line)
        out_stream = io.StringIO()
        env = CommunicationModEnv(input_stream=in_stream, output_stream=out_stream)
        env.reset()

        _, reward, done, info = env.step("end")
        out_stream.seek(0)
        written = [json.loads(ln) for ln in out_stream if ln.strip()]
        assert any(cmd.get("command") == "end" for cmd in written)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert info["turn"] == 1

    def test_max_turns_ends_episode(self):
        lines = [self._make_state_line() for _ in range(3)]
        env, _ = self._make_env(lines, max_turns=1)
        env.reset()
        _, _, done, info = env.step("end")
        assert done is True
        assert info["turn"] == 1

    def test_step_after_done_is_noop(self):
        import io
        import json
        from environment.game_env import CommunicationModEnv

        state_line = self._make_state_line()
        in_stream = io.StringIO(state_line)
        out_stream = io.StringIO()
        env = CommunicationModEnv(max_turns=1, input_stream=in_stream, output_stream=out_stream)
        env.reset()
        env.step("end")  # max_turns reached → done=True

        out_stream.seek(0)
        cmds_before = out_stream.read()

        # Second step should not send another command
        env.step("end")
        out_stream.seek(0)
        cmds_after = out_stream.read()
        assert cmds_before == cmds_after

    def test_close_is_noop(self):
        from environment.game_env import CommunicationModEnv
        import io

        env = CommunicationModEnv(input_stream=io.StringIO(), output_stream=io.StringIO())
        env.close()  # must not raise

    def test_eof_returns_empty_state(self):
        from environment.game_env import CommunicationModEnv
        import io

        env = CommunicationModEnv(input_stream=io.StringIO(""), output_stream=io.StringIO())
        state = env.reset()
        assert isinstance(state, str)

    # ------------------------------------------------------------------
    # _format_game_state
    # ------------------------------------------------------------------

    def test_format_includes_monsters(self):
        import json
        import io
        from environment.game_env import CommunicationModEnv

        monsters = [
            {
                "name": "Cultist", "current_hp": 50, "max_hp": 50,
                "intent": "DEBUFF", "block": 0, "is_gone": False,
            }
        ]
        line = self._make_state_line(monsters=monsters)
        env = CommunicationModEnv(input_stream=io.StringIO(line), output_stream=io.StringIO())
        state = env.reset()
        assert "=== Monster ===" in state
        assert "Cultist" in state

    def test_format_includes_choices(self):
        import io
        from environment.game_env import CommunicationModEnv

        line = self._make_state_line(choice_list=["Strike", "Defend", "Bash"])
        env = CommunicationModEnv(input_stream=io.StringIO(line), output_stream=io.StringIO())
        state = env.reset()
        assert "=== Choices ===" in state
        assert "Bash" in state

    def test_format_skips_dead_monsters(self):
        import io
        from environment.game_env import CommunicationModEnv

        monsters = [
            {"name": "Dead", "current_hp": 0, "max_hp": 50, "intent": "NONE", "block": 0, "is_gone": True},
        ]
        line = self._make_state_line(monsters=monsters)
        env = CommunicationModEnv(input_stream=io.StringIO(line), output_stream=io.StringIO())
        state = env.reset()
        assert "Dead" not in state

    # ------------------------------------------------------------------
    # _translate_command
    # ------------------------------------------------------------------

    def test_translate_end(self):
        from environment.game_env import CommunicationModEnv
        import io

        env = CommunicationModEnv(input_stream=io.StringIO(), output_stream=io.StringIO())
        assert env._translate_command("end") == {"command": "end"}

    def test_translate_proceed(self):
        from environment.game_env import CommunicationModEnv
        import io

        env = CommunicationModEnv(input_stream=io.StringIO(), output_stream=io.StringIO())
        assert env._translate_command("proceed") == {"command": "proceed"}
        assert env._translate_command("confirm") == {"command": "proceed"}

    def test_translate_choose(self):
        from environment.game_env import CommunicationModEnv
        import io

        env = CommunicationModEnv(input_stream=io.StringIO(), output_stream=io.StringIO())
        assert env._translate_command("choose 2") == {"command": "choose", "choice_index": 1}

    def test_translate_numeric_combat(self):
        from environment.game_env import CommunicationModEnv
        import io

        env = CommunicationModEnv(input_stream=io.StringIO(), output_stream=io.StringIO())
        env._screen_type = "NONE"  # combat screen
        assert env._translate_command("1") == {"command": "play", "hand_index": 0}
        assert env._translate_command("3") == {"command": "play", "hand_index": 2}

    def test_translate_numeric_combat_with_target(self):
        from environment.game_env import CommunicationModEnv
        import io

        env = CommunicationModEnv(input_stream=io.StringIO(), output_stream=io.StringIO())
        env._screen_type = "NONE"
        assert env._translate_command("1 2") == {"command": "play", "hand_index": 0, "target_index": 1}

    def test_translate_numeric_noncombat(self):
        from environment.game_env import CommunicationModEnv
        import io

        env = CommunicationModEnv(input_stream=io.StringIO(), output_stream=io.StringIO())
        env._screen_type = "MAP"
        assert env._translate_command("2") == {"command": "choose", "choice_index": 1}

    def test_translate_potion_use(self):
        from environment.game_env import CommunicationModEnv
        import io

        env = CommunicationModEnv(input_stream=io.StringIO(), output_stream=io.StringIO())
        assert env._translate_command("pot u 1") == {"command": "potion", "use": True, "slot": 0}
        assert env._translate_command("pot u 2 1") == {"command": "potion", "use": True, "slot": 1, "target_index": 0}

    def test_translate_potion_discard(self):
        from environment.game_env import CommunicationModEnv
        import io

        env = CommunicationModEnv(input_stream=io.StringIO(), output_stream=io.StringIO())
        assert env._translate_command("pot d 1") == {"command": "potion", "use": False, "slot": 0}

    def test_translate_multi_action_uses_first(self):
        from environment.game_env import CommunicationModEnv
        import io

        env = CommunicationModEnv(input_stream=io.StringIO(), output_stream=io.StringIO())
        env._screen_type = "NONE"
        result = env._translate_command("1,2,end")
        assert result == {"command": "play", "hand_index": 0}

    def test_translate_unknown_falls_back_to_proceed(self):
        from environment.game_env import CommunicationModEnv
        import io

        env = CommunicationModEnv(input_stream=io.StringIO(), output_stream=io.StringIO())
        assert env._translate_command("xyzzy") == {"command": "proceed"}

    # ------------------------------------------------------------------
    # _read_state skips non-ready messages
    # ------------------------------------------------------------------

    def test_read_state_skips_non_ready_messages(self):
        import io
        import json
        from environment.game_env import CommunicationModEnv

        informational = json.dumps({"ready_for_command": False, "in_game": True}) + "\n"
        ready = self._make_state_line(current_hp=70, max_hp=80)
        env = CommunicationModEnv(
            input_stream=io.StringIO(informational + informational + ready),
            output_stream=io.StringIO(),
        )
        state = env.reset()
        assert "Health: 70/80" in state

    def test_read_state_skips_malformed_json(self):
        import io
        from environment.game_env import CommunicationModEnv

        bad_line = "not valid json\n"
        ready = self._make_state_line()
        env = CommunicationModEnv(
            input_stream=io.StringIO(bad_line + ready),
            output_stream=io.StringIO(),
        )
        state = env.reset()
        assert isinstance(state, str)


if __name__ == "__main__":
    unittest.main()

