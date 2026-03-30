"""Gym-style environments that interface with Slay the Spire.

Three back-ends are available; select via ``interface_mode`` in ``config.yaml``:

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

``communication_mod``
    Communicates with the
    `Communication Mod <https://github.com/ForgottenArbiter/CommunicationMod>`_
    via stdin/stdout using newline-delimited JSON messages.  The mod launches
    ``main.py`` as an external process and pipes game state to its stdin;
    commands are written back to stdout.  Works on **any platform** including
    macOS with no platform-specific dependencies.

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
# CommunicationModEnv — cross-platform stdin/stdout JSON interface
# ---------------------------------------------------------------------------


class CommunicationModEnv:
    """Gym-inspired environment that communicates with the Communication Mod.

    The `Communication Mod
    <https://github.com/ForgottenArbiter/CommunicationMod>`_ launches an
    external process (``main.py``) and exchanges newline-delimited JSON
    messages with it via stdin/stdout.  This back-end has **no
    platform-specific dependencies** and works on macOS, Linux, and Windows.

    Protocol
    --------
    * The mod writes a JSON object to the process's stdin whenever the game
      state changes and the bot should act.  The object contains a
      ``"ready_for_command": true`` field together with a ``"game_state"``
      sub-object describing the full game state.
    * The bot writes a JSON command object to stdout, e.g.
      ``{"command": "end"}`` or ``{"command": "play", "hand_index": 0}``.
    * Each message is terminated by a newline.

    This environment translates the sts-agent text command format produced by
    :class:`~agent.agent.SlayTheSpireAgent` (e.g. ``"1"``, ``"end"``,
    ``"choose 2"``, ``"pot u 1"``) into the JSON commands the Communication
    Mod expects.

    Parameters
    ----------
    max_turns:
        Maximum agent turns before the episode is forcibly ended.
    reward_config:
        Optional :class:`~training.reward.RewardConfig`.
    input_stream:
        Readable text stream to receive game-state JSON from the mod.
        Defaults to :data:`sys.stdin`.
    output_stream:
        Writable text stream to send commands to the mod.
        Defaults to :data:`sys.stdout`.
    """

    def __init__(
        self,
        max_turns: int = 200,
        reward_config: Optional[RewardConfig] = None,
        input_stream=None,
        output_stream=None,
    ) -> None:
        self._in = input_stream if input_stream is not None else sys.stdin
        self._out = output_stream if output_stream is not None else sys.stdout
        self.max_turns = max_turns
        self._reward_config: RewardConfig = reward_config or RewardConfig()

        self._turn_count: int = 0
        self._done: bool = False
        self._last_state: str = ""

        self._floors_cleared: int = 0
        self._enemies_killed: int = 0
        self._won: bool = False
        # Track screen type so numeric commands are mapped to the right JSON.
        self._screen_type: str = ""

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(self) -> str:
        """Wait for the mod to signal readiness and return the initial state.

        Slay the Spire must already be running with the Communication Mod
        active.  The mod will send the first ready-for-command message as
        soon as the bot process connects.
        """
        self._turn_count = 0
        self._done = False
        self._floors_cleared = 0
        self._enemies_killed = 0
        self._won = False

        state_data = self._read_state()
        state = self._format_game_state(state_data)
        self._last_state = state
        logger.info("Environment reset; initial state length=%d chars", len(state))
        return state

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Translate *action* to a JSON command, send it, and read the next state.

        Parameters
        ----------
        action:
            A command string in sts-agent format, e.g. ``"1"``, ``"end"``,
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
            Diagnostic dictionary.
        """
        if self._done:
            return self._last_state, 0.0, True, self._info()

        self._turn_count += 1

        cmd = self._translate_command(action)
        self._send_command(cmd)

        state_data = self._read_state()
        new_state = self._format_game_state(state_data)
        self._last_state = new_state

        reward, done = self._compute_reward(new_state)
        if self._turn_count >= self.max_turns:
            logger.info("Max turns (%d) reached; ending episode.", self.max_turns)
            done = True

        self._done = done
        return new_state, reward, done, self._info()

    def close(self) -> None:
        """No-op: the Communication Mod manages the process lifecycle."""

    # ------------------------------------------------------------------
    # Communication helpers
    # ------------------------------------------------------------------

    def _send_command(self, command: dict) -> None:
        """Serialise *command* as JSON and write it to the output stream."""
        self._out.write(json.dumps(command) + "\n")
        self._out.flush()
        logger.debug("Sent command: %s", command)

    def _read_state(self) -> dict:
        """Read lines from the input stream until a ready-for-command message.

        Informational messages (e.g. game loading) are skipped.  Returns the
        parsed JSON dict, or an empty dict on EOF.
        """
        while True:
            line = self._in.readline()
            if not line:
                logger.warning("Input stream closed (EOF).")
                return {}
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse Communication Mod JSON: %s", exc)
                continue
            if data.get("error"):
                logger.warning("Communication Mod error: %s", data["error"])
            if data.get("ready_for_command", False):
                return data
            # Informational message — keep waiting.

    # ------------------------------------------------------------------
    # State formatting
    # ------------------------------------------------------------------

    def _format_game_state(self, data: dict) -> str:
        """Convert a Communication Mod JSON payload to a human-readable state string.

        Produces sections in the same labelled format as the Text the Spire
        windows (``=== Player ===``, ``=== Hand ===``, etc.) so the LLM
        agent's system prompt and action parser require no changes.
        """
        if not data:
            return ""

        game_state = data.get("game_state", {})
        if not game_state:
            return ""

        self._screen_type = game_state.get("screen_type", "")
        combat = game_state.get("combat_state", {}) or {}
        player_combat = combat.get("player", {}) or {}
        parts: list[str] = []

        # --- Player window ---
        current_hp = game_state.get("current_hp", 0)
        max_hp = game_state.get("max_hp", 0)
        block = player_combat.get("block", 0)
        energy = player_combat.get("energy", 0)
        gold = game_state.get("gold", 0)
        floor_num = game_state.get("floor", 0)
        player_lines = [
            f"Health: {current_hp}/{max_hp}",
            f"Block: {block}",
            f"Energy: {energy}",
            f"Gold: {gold}",
            f"Floor: {floor_num}",
        ]
        for power in player_combat.get("powers", []):
            player_lines.append(
                f"Power: {power.get('name', '?')} ({power.get('amount', '')})"
            )
        parts.append("=== Player ===\n" + "\n".join(player_lines))

        # --- Hand window ---
        hand = combat.get("hand", [])
        if hand:
            hand_lines = []
            for i, card in enumerate(hand, 1):
                name = card.get("name", "Unknown")
                cost = card.get("cost", "?")
                hand_lines.append(f"{i}: {name} ({cost} energy)")
            parts.append("=== Hand ===\n" + "\n".join(hand_lines))

        # --- Monster window ---
        monsters = [m for m in combat.get("monsters", []) if not m.get("is_gone", False)]
        if monsters:
            monster_lines = []
            for m in monsters:
                name = m.get("name", "Unknown")
                hp = m.get("current_hp", 0)
                max_hp_m = m.get("max_hp", 0)
                intent = m.get("intent", "Unknown")
                block_m = m.get("block", 0)
                monster_lines.append(
                    f"{name} HP: {hp}/{max_hp_m} Block: {block_m} Intent: {intent}"
                )
            parts.append("=== Monster ===\n" + "\n".join(monster_lines))

        # --- Choices window ---
        choice_list = game_state.get("choice_list", []) or []
        screen_state = game_state.get("screen_state") or {}
        options = screen_state.get("options", []) if isinstance(screen_state, dict) else []
        choices = choice_list or options
        if choices:
            choice_lines = []
            for i, choice in enumerate(choices, 1):
                if isinstance(choice, str):
                    choice_lines.append(f"{i}: {choice}")
                elif isinstance(choice, dict):
                    label = choice.get("text", choice.get("name", str(choice)))
                    choice_lines.append(f"{i}: {label}")
            parts.append("=== Choices ===\n" + "\n".join(choice_lines))

        # --- Map window ---
        map_lines = [f"Floor: {floor_num}"]
        act = game_state.get("act", 0)
        if act:
            map_lines.append(f"Act: {act}")
        parts.append("=== Map ===\n" + "\n".join(map_lines))

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Command translation
    # ------------------------------------------------------------------

    def _translate_command(self, action: str) -> dict:
        """Translate an sts-agent text action to a Communication Mod JSON command.

        The LLM agent produces commands in sts-agent format.  This method maps
        them to the JSON commands the Communication Mod expects.  Numeric
        indices are converted from 1-based (sts-agent) to 0-based (mod).

        Mapping
        -------
        * ``"end"`` → ``{"command": "end"}``
        * ``"proceed"`` / ``"confirm"`` / ``"skip"`` → ``{"command": "proceed"}``
        * ``"choose N"`` → ``{"command": "choose", "choice_index": N-1}``
        * ``"pot u N"`` → ``{"command": "potion", "use": true, "slot": N-1}``
        * ``"pot u N M"`` → adds ``"target_index": M-1``
        * ``"pot d N"`` → ``{"command": "potion", "use": false, "slot": N-1}``
        * ``"N"`` in combat (screen_type ``"NONE"``) → ``{"command": "play", "hand_index": N-1}``
        * ``"N M"`` in combat → adds ``"target_index": M-1``
        * ``"N"`` non-combat → ``{"command": "choose", "choice_index": N-1}``
        * ``"N,M,…"`` multi-action → only the first token is sent per step
        * Unrecognised → ``{"command": "proceed"}``
        """
        action = action.strip()

        # Comma-separated multi-action: only the first token is sent per step.
        if "," in action:
            first = action.split(",")[0].strip()
            return self._translate_command(first)

        lower = action.lower()

        if lower == "end":
            return {"command": "end"}

        if lower in ("proceed", "confirm", "skip"):
            return {"command": "proceed"}

        # "choose N"
        m = re.match(r"^choose\s+(\d+)$", action, re.IGNORECASE)
        if m:
            return {"command": "choose", "choice_index": int(m.group(1)) - 1}

        # "pot u N" or "pot u N M"
        m = re.match(r"^pot\s+u\s+(\d+)(?:\s+(\d+))?$", action, re.IGNORECASE)
        if m:
            cmd: dict = {"command": "potion", "use": True, "slot": int(m.group(1)) - 1}
            if m.group(2):
                cmd["target_index"] = int(m.group(2)) - 1
            return cmd

        # "pot d N"
        m = re.match(r"^pot\s+d\s+(\d+)$", action, re.IGNORECASE)
        if m:
            return {"command": "potion", "use": False, "slot": int(m.group(1)) - 1}

        # "map N …" — map navigation uses choice_index
        m = re.match(r"^map\s+(\d+)", action, re.IGNORECASE)
        if m:
            return {"command": "choose", "choice_index": int(m.group(1)) - 1}

        # "N" or "N M"
        m = re.match(r"^(\d+)(?:\s+(\d+))?$", action)
        if m:
            idx = int(m.group(1)) - 1
            if self._screen_type == "NONE":
                # In combat — play card from hand.
                cmd = {"command": "play", "hand_index": idx}
                if m.group(2):
                    cmd["target_index"] = int(m.group(2)) - 1
                return cmd
            else:
                # Non-combat screen — choose from options list.
                return {"command": "choose", "choice_index": idx}

        logger.warning("Unrecognised action %r; sending 'proceed'.", action)
        return {"command": "proceed"}

    # ------------------------------------------------------------------
    # Reward / info
    # ------------------------------------------------------------------

    def _compute_reward(self, state: str) -> tuple[float, bool]:
        """Identical reward logic to the other environment back-ends."""
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

    ``"communication_mod"``
        Returns a :class:`CommunicationModEnv` that communicates with the
        `Communication Mod <https://github.com/ForgottenArbiter/CommunicationMod>`_
        via stdin/stdout JSON.  Works on any platform including macOS.

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
    StsAgentEnv | TextTheSpireEnv | CommunicationModEnv
    """
    windows = env_cfg.get("windows", None)
    max_turns = int(env_cfg.get("max_turns", 200))
    command_timeout = float(env_cfg.get("command_timeout", 5.0))
    mode = env_cfg.get("interface_mode", "sts_agent")

    if mode == "communication_mod":
        logger.info("Using CommunicationModEnv (stdin/stdout JSON)")
        return CommunicationModEnv(
            max_turns=max_turns,
            reward_config=reward_config,
        )

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

