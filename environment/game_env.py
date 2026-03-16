"""Gym-style environment that interfaces with Slay the Spire via sts-agent.

`sts-agent <https://github.com/ohylli/sts-agent>`_ bridges AI agents to a
running Slay the Spire instance through the Text the Spire accessibility mod.
This environment calls the ``sts_tool.py`` CLI on each step — there is no
long-running game subprocess to manage.

Requirements
------------
* Windows OS (sts-agent uses pywinauto/pywin32 for UI automation).
* Slay the Spire with the Text the Spire mod installed and running.
* sts-agent cloned: ``git clone https://github.com/ohylli/sts-agent``.

Usage::

    from environment.game_env import StsAgentEnv

    env = StsAgentEnv(sts_tool_path="sts-agent/src/sts_tool.py")
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done, info = env.step(action)
    env.close()
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from typing import Optional

from training.reward import RewardConfig, compute_step_reward

logger = logging.getLogger(__name__)

# Windows to read from Text the Spire on each step (order determines the
# formatted game-state string shown to the model).
_DEFAULT_WINDOWS = ["Player", "Hand", "Monster", "Choices", "Map"]

# Extra seconds added to command_timeout when setting the subprocess hard
# deadline; ensures the tool has time to report a timeout internally before
# the subprocess is killed from the outside.
_SUBPROCESS_TIMEOUT_BUFFER = 10

# Pattern to detect player HP from the Player window, used for done detection.
_HP_RE = re.compile(r"Health:\s*(\d+)/(\d+)", re.IGNORECASE)


def _format_state(windows_data: list[dict]) -> str:
    """Convert a list of window-content dicts into a single human-readable string.

    Each window becomes a labelled section; windows that are unavailable or empty
    are marked but still included so the model is aware of the context.
    """
    parts: list[str] = []
    for window in windows_data:
        title = window.get("window_title", "Unknown")
        error = window.get("error")
        content = window.get("content", "")
        if error:
            parts.append(f"=== {title} ===\n[unavailable: {error}]")
        elif content.strip():
            parts.append(f"=== {title} ===\n{content.strip()}")
    return "\n\n".join(parts)


class StsAgentEnv:
    """A Gym-inspired environment wrapping the sts-agent CLI tool.

    Each :meth:`reset` and :meth:`step` call invokes ``sts_tool.py`` as a
    subprocess (using ``--json`` for machine-readable output).  The resulting
    game state is the formatted text of all relevant Text the Spire windows.

    Parameters
    ----------
    sts_tool_path:
        Path to ``sts-agent/src/sts_tool.py``.
    python_executable:
        Python interpreter used to invoke the tool.  On Windows/WSL this
        should be ``python.exe``; defaults to the current interpreter.
    windows:
        List of Text the Spire window names to read on each step.
    max_turns:
        Maximum agent turns before the episode is forcibly ended.
    command_timeout:
        Per-command timeout (seconds) forwarded to ``sts_tool.py --timeout``.
    reward_config:
        Optional :class:`~training.reward.RewardConfig`.
    """

    def __init__(
        self,
        sts_tool_path: str = "sts-agent/src/sts_tool.py",
        python_executable: Optional[str] = None,
        windows: Optional[list[str]] = None,
        max_turns: int = 200,
        command_timeout: float = 5.0,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        self.sts_tool_path = sts_tool_path
        self.python_executable = python_executable or sys.executable
        self.windows = windows or list(_DEFAULT_WINDOWS)
        self.max_turns = max_turns
        self.command_timeout = command_timeout
        self._reward_config: RewardConfig = reward_config or RewardConfig()

        self._turn_count: int = 0
        self._done: bool = False
        self._last_state: str = ""

        # Episode-level statistics used by :meth:`_info`.
        self._floors_cleared: int = 0
        self._enemies_killed: int = 0
        self._won: bool = False

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(self) -> str:
        """Read all windows and return the current game state as a string.

        Unlike the previous slaythetext env this does **not** launch a new
        game process; Slay the Spire must already be running with Text the
        Spire active.
        """
        self._turn_count = 0
        self._done = False
        self._floors_cleared = 0
        self._enemies_killed = 0
        self._won = False

        state = self._read_windows()
        self._last_state = state
        logger.info("Environment reset; initial state length=%d chars", len(state))
        return state

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Execute *action* and return the next observation.

        Parameters
        ----------
        action:
            A command string accepted by sts-agent, e.g. ``"1"``, ``"end"``,
            ``"choose 1"``, ``"1,2,end"``, ``"pot u 1"``.

        Returns
        -------
        state:
            Formatted game state after the action.
        reward:
            Scalar reward for this step.
        done:
            Whether the episode has ended.
        info:
            Diagnostic dictionary (floor count, enemy kills, etc.).
        """
        if self._done:
            return self._last_state, 0.0, True, self._info()

        self._turn_count += 1

        new_state = self._execute_and_read(action)
        self._last_state = new_state

        reward, done = self._compute_reward(new_state)
        if self._turn_count >= self.max_turns:
            logger.info("Max turns (%d) reached; ending episode.", self.max_turns)
            done = True

        self._done = done
        return new_state, reward, done, self._info()

    def close(self) -> None:
        """No-op: sts-agent does not maintain a persistent subprocess."""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_tool(self, extra_args: list[str]) -> dict:
        """Run ``sts_tool.py`` with *extra_args* and return the parsed JSON.

        Returns an empty dict on timeout or parse failure.
        """
        cmd = [
            self.python_executable,
            self.sts_tool_path,
            "--json",
            "--timeout", str(self.command_timeout),
        ] + extra_args
        logger.debug("sts_tool cmd: %s", cmd)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.command_timeout + _SUBPROCESS_TIMEOUT_BUFFER,
            )
            if result.stdout.strip():
                return json.loads(result.stdout)
            logger.warning(
                "sts_tool produced no stdout; stderr=%r", result.stderr[:200]
            )
            return {}
        except subprocess.TimeoutExpired:
            logger.warning("sts_tool timed out for args %s", extra_args)
            return {}
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse sts_tool JSON: %s", exc)
            return {}

    def _read_windows(self) -> str:
        """Read all configured windows and return the formatted state."""
        windows_str = ",".join(self.windows)
        data = self._run_tool(["--read-window", windows_str])
        windows_list = data.get("windows", [])
        if not windows_list and "window_title" in data:
            # Single-window response — wrap for uniform handling.
            windows_list = [data]
        return _format_state(windows_list)

    def _execute_and_read(self, action: str) -> str:
        """Execute *action* and read all windows in a single CLI invocation."""
        windows_str = ",".join(self.windows)
        data = self._run_tool(["--execute", action, "--read-window", windows_str])
        windows_list = data.get("windows", [])
        if not windows_list and "window_title" in data:
            windows_list = [data]
        return _format_state(windows_list)

    def _compute_reward(self, state: str) -> tuple[float, bool]:
        """Delegate reward computation to :mod:`training.reward`.

        Also updates episode-level statistics (``_won``, ``_floors_cleared``,
        ``_enemies_killed``) used by :meth:`_info`.  In addition, detects
        player death via the Player window's ``Health: 0/N`` pattern.
        """
        cfg = getattr(self, "_reward_config", None) or RewardConfig()
        reward, done = compute_step_reward(state, cfg)

        if done and reward > 0:
            self._won = True
            logger.info("Episode ended: WIN")
        elif done and reward < 0:
            logger.info("Episode ended: LOSS")
        else:
            from training.reward import _ENEMY_KILLED_RE, _FLOOR_ADVANCE_RE

            if _ENEMY_KILLED_RE.search(state):
                self._enemies_killed += len(_ENEMY_KILLED_RE.findall(state))
            if _FLOOR_ADVANCE_RE.search(state):
                self._floors_cleared += 1

            # Detect player death from the Player window HP field.
            hp_match = _HP_RE.search(state)
            if hp_match and int(hp_match.group(1)) == 0:
                reward += cfg.loss_penalty
                done = True
                logger.info("Episode ended: player HP reached 0")

        return reward, done

    def _info(self) -> dict:
        """Return a diagnostic dictionary for the current episode state."""
        return {
            "turn": self._turn_count,
            "floors_cleared": self._floors_cleared,
            "enemies_killed": self._enemies_killed,
            "won": self._won,
            "done": self._done,
        }

