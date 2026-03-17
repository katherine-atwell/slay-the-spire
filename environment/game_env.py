"""Gym-style environments that interface with Slay the Spire.

Two back-ends are available; select via ``interface_mode`` in ``config.yaml``:

``sts_agent`` (default)
    Uses `sts-agent <https://github.com/ohylli/sts-agent>`_ to bridge to a
    running Slay the Spire instance via the Text the Spire accessibility mod.
    Calls ``sts_tool.py`` as a subprocess on each step.
    **Windows only** — sts-agent relies on pywinauto/pywin32 for UI automation.

``text_the_spire``
    Interfaces directly with the
    `Text the Spire <https://github.com/Wensber/TextTheSpire>`_ mod by reading
    per-window state files the mod writes and sending commands via a shared
    input file.  Works on **any platform** (Linux, macOS, Windows).

Use :func:`make_env` to construct the correct environment from a config dict.

Usage::

    from environment.game_env import make_env

    env = make_env(cfg["environment"])
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
import os
import re
import subprocess
import sys
import time
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

            kills = _ENEMY_KILLED_RE.findall(state)
            if kills:
                self._enemies_killed += len(kills)
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


# ---------------------------------------------------------------------------
# TextTheSpireEnv — cross-platform direct interface with the mod
# ---------------------------------------------------------------------------


class TextTheSpireEnv:
    """A Gym-inspired environment that interfaces directly with the Text the
    Spire mod via the file system.

    The Text the Spire mod writes the current game-window content to
    individual text files in a configurable *state directory*.  One file is
    created per window (e.g. ``Player.txt``, ``Hand.txt``).  Commands are
    sent to the mod by writing a single line to a shared *input file* which
    the mod reads and executes.

    This back-end does **not** depend on pywinauto / pywin32, so it works on
    Linux and macOS as well as Windows.

    Parameters
    ----------
    state_dir:
        Directory where Text the Spire writes window state files.  Each
        window named ``W`` should produce a file ``<state_dir>/<W>.txt``.
    input_file:
        Path to the command input file that Text the Spire monitors for
        incoming actions.
    windows:
        List of Text the Spire window names to read on each step.
    max_turns:
        Maximum agent turns before the episode is forcibly ended.
    command_timeout:
        Seconds to wait for each window file to be updated after sending an
        action before giving up and returning the last-known content.
    reward_config:
        Optional :class:`~training.reward.RewardConfig`.
    """

    def __init__(
        self,
        state_dir: str = ".",
        input_file: str = "sts_input.txt",
        windows: Optional[list[str]] = None,
        max_turns: int = 200,
        command_timeout: float = 5.0,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        self.state_dir = state_dir
        self.input_file = input_file
        self.windows = windows or list(_DEFAULT_WINDOWS)
        self.max_turns = max_turns
        self.command_timeout = command_timeout
        self._reward_config: RewardConfig = reward_config or RewardConfig()

        self._turn_count: int = 0
        self._done: bool = False
        self._last_state: str = ""

        self._floors_cleared: int = 0
        self._enemies_killed: int = 0
        self._won: bool = False

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(self) -> str:
        """Read all window files and return the current game state as a string.

        Slay the Spire must already be running with the Text the Spire mod
        active so that window state files are present.
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
        """Send *action* to the mod and return the next observation.

        Parameters
        ----------
        action:
            A command string understood by Text the Spire, e.g. ``"1"``,
            ``"end"``, ``"choose 1"``.

        Returns
        -------
        state:
            Formatted game state after the action.
        reward:
            Scalar reward for this step.
        done:
            Whether the episode has ended.
        info:
            Diagnostic dictionary.
        """
        if self._done:
            return self._last_state, 0.0, True, self._info()

        self._turn_count += 1

        self._send_command(action)
        new_state = self._read_windows()
        self._last_state = new_state

        reward, done = self._compute_reward(new_state)
        if self._turn_count >= self.max_turns:
            logger.info("Max turns (%d) reached; ending episode.", self.max_turns)
            done = True

        self._done = done
        return new_state, reward, done, self._info()

    def close(self) -> None:
        """No-op: no persistent subprocess to clean up."""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _send_command(self, action: str) -> None:
        """Write *action* to the mod's input file."""
        try:
            with open(self.input_file, "w", encoding="utf-8") as fh:
                fh.write(action.strip() + "\n")
            logger.debug("Sent command %r to %s", action, self.input_file)
        except OSError as exc:
            logger.warning("Failed to write command to %s: %s", self.input_file, exc)

    def _read_window_file(self, window_name: str) -> dict:
        """Return a window-content dict for *window_name*.

        Polls until the state file exists (or ``command_timeout`` expires).
        Returns an error dict if the file is not found or unreadable.
        """
        path = os.path.join(self.state_dir, f"{window_name}.txt")
        deadline = time.monotonic() + self.command_timeout
        while not os.path.exists(path):
            if time.monotonic() >= deadline:
                logger.warning("State file not found for window %r: %s", window_name, path)
                return {"window_title": window_name, "content": "", "error": "file not found"}
            time.sleep(0.1)

        try:
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            return {"window_title": window_name, "content": content, "error": None}
        except OSError as exc:
            logger.warning("Failed to read state file %s: %s", path, exc)
            return {"window_title": window_name, "content": "", "error": str(exc)}

    def _read_windows(self) -> str:
        """Read all configured window files and return the formatted state."""
        windows_data = [self._read_window_file(w) for w in self.windows]
        return _format_state(windows_data)

    # Delegate reward / info to the same helpers used by StsAgentEnv so the
    # two environments behave identically from the training loop's perspective.

    def _compute_reward(self, state: str) -> tuple[float, bool]:
        """Identical reward logic to :class:`StsAgentEnv`."""
        cfg = self._reward_config
        reward, done = compute_step_reward(state, cfg)

        if done and reward > 0:
            self._won = True
            logger.info("Episode ended: WIN")
        elif done and reward < 0:
            logger.info("Episode ended: LOSS")
        else:
            from training.reward import _ENEMY_KILLED_RE, _FLOOR_ADVANCE_RE

            kills = _ENEMY_KILLED_RE.findall(state)
            if kills:
                self._enemies_killed += len(kills)
            if _FLOOR_ADVANCE_RE.search(state):
                self._floors_cleared += 1

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_env(env_cfg: dict, reward_config: Optional[RewardConfig] = None):
    """Construct and return the appropriate environment from *env_cfg*.

    The ``interface_mode`` key in *env_cfg* controls which back-end is used:

    ``"sts_agent"`` (default)
        Returns a :class:`StsAgentEnv` that calls ``sts_tool.py`` via
        subprocess.  Requires Windows and the sts-agent submodule.

    ``"text_the_spire"``
        Returns a :class:`TextTheSpireEnv` that reads window state files
        written by the Text the Spire mod.  Works on any platform.

    Parameters
    ----------
    env_cfg:
        The ``environment`` section of ``config.yaml`` (or an equivalent
        dict with the same keys).
    reward_config:
        Optional :class:`~training.reward.RewardConfig` forwarded to the
        constructed environment.

    Returns
    -------
    StsAgentEnv | TextTheSpireEnv
    """
    windows = env_cfg.get("windows", None)
    max_turns = int(env_cfg.get("max_turns", 200))
    command_timeout = float(env_cfg.get("command_timeout", 5.0))
    mode = env_cfg.get("interface_mode", "sts_agent")

    if mode == "text_the_spire":
        state_dir = env_cfg.get("text_the_spire_state_dir", ".")
        input_file = env_cfg.get("text_the_spire_input_file", "sts_input.txt")
        logger.info(
            "Using TextTheSpireEnv (state_dir=%r, input_file=%r)", state_dir, input_file
        )
        return TextTheSpireEnv(
            state_dir=state_dir,
            input_file=input_file,
            windows=windows,
            max_turns=max_turns,
            command_timeout=command_timeout,
            reward_config=reward_config,
        )

    if mode != "sts_agent":
        logger.warning(
            "Unknown interface_mode %r; falling back to 'sts_agent'.", mode
        )

    sts_tool_path = env_cfg.get("sts_tool_path", "sts-agent/src/sts_tool.py")
    python_executable = env_cfg.get("python_executable", None)
    logger.info(
        "Using StsAgentEnv (sts_tool_path=%r, python_executable=%r)",
        sts_tool_path,
        python_executable,
    )
    return StsAgentEnv(
        sts_tool_path=sts_tool_path,
        python_executable=python_executable,
        windows=windows,
        max_turns=max_turns,
        command_timeout=command_timeout,
        reward_config=reward_config,
    )

