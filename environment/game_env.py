"""Gym-style environment that wraps the slaythetext game as a subprocess.

The game communicates via stdin/stdout; this module manages the subprocess
lifecycle, feeds actions, reads state updates, and computes step-level reward
signals passed to the RL trainer.

Reward computation is fully delegated to :mod:`training.reward` so that the
environment and the trainer always use the same signals and weights.

Usage::

    from environment.game_env import SlayTheSpireEnv

    env = SlayTheSpireEnv(game_script="slaythetext/main.py")
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done, info = env.step(action)
    env.close()
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import time
from typing import Optional

from training.reward import RewardConfig, compute_step_reward

logger = logging.getLogger(__name__)

# ANSI escape code pattern for stripping colour markup from game output.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHFABCDJsu]|<[^>]+>")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes and angle-bracket markup from *text*."""
    return _ANSI_RE.sub("", text)


class SlayTheSpireEnv:
    """A Gym-inspired environment wrapping the slaythetext CLI game.

    Parameters
    ----------
    game_script:
        Path to ``slaythetext/main.py`` (or any compatible entry point).
    python_executable:
        Python interpreter used to launch the game subprocess.
        Defaults to the same interpreter running this module.
    max_turns:
        Maximum number of agent turns before the episode is forcibly ended
        (prevents infinite loops in degenerate game states).
    read_timeout:
        Seconds to wait for the game process to emit output before giving up.
    """

    def __init__(
        self,
        game_script: str = "slaythetext/main.py",
        python_executable: Optional[str] = None,
        max_turns: int = 200,
        read_timeout: float = 30.0,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        self.game_script = game_script
        self.python_executable = python_executable or sys.executable
        self.max_turns = max_turns
        self.read_timeout = read_timeout
        self._reward_config: RewardConfig = reward_config or RewardConfig()

        self._proc: Optional[subprocess.Popen] = None
        self._output_buffer: list[str] = []
        self._reader_thread: Optional[threading.Thread] = None
        self._turn_count: int = 0
        self._done: bool = False
        self._last_state: str = ""

        # Episode-level statistics for reward computation.
        self._floors_cleared: int = 0
        self._enemies_killed: int = 0
        self._won: bool = False

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(self) -> str:
        """Start a new game episode and return the initial game state text.

        Any running subprocess from a previous episode is terminated first.
        """
        self.close()

        self._output_buffer = []
        self._turn_count = 0
        self._done = False
        self._floors_cleared = 0
        self._enemies_killed = 0
        self._won = False

        logger.info("Launching game: %s %s", self.python_executable, self.game_script)
        self._proc = subprocess.Popen(
            [self.python_executable, self.game_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        # Background thread collects stdout lines into the shared buffer.
        self._reader_thread = threading.Thread(
            target=self._read_output_loop, daemon=True
        )
        self._reader_thread.start()

        initial_state = self._collect_output()
        self._last_state = initial_state
        return initial_state

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Send *action* to the game and return the next observation.

        Parameters
        ----------
        action:
            A string action to send to the game (e.g. ``"1"`` or ``"Save"``).

        Returns
        -------
        state:
            The new game state text after the action.
        reward:
            Scalar reward for this step.
        done:
            Whether the episode has ended.
        info:
            Diagnostic dictionary (floor count, enemy kills, etc.).
        """
        if self._done or self._proc is None:
            return self._last_state, 0.0, True, self._info()

        self._turn_count += 1

        # Send action to game.
        try:
            self._proc.stdin.write(action + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError:
            logger.warning("Game process pipe broken; ending episode.")
            self._done = True
            return self._last_state, 0.0, True, self._info()

        new_output = self._collect_output()
        self._last_state = new_output

        reward, done = self._compute_reward(new_output)
        if self._turn_count >= self.max_turns:
            logger.info("Max turns (%d) reached; ending episode.", self.max_turns)
            done = True

        self._done = done
        return new_output, reward, done, self._info()

    def close(self) -> None:
        """Terminate the game subprocess if running."""
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)
            self._reader_thread = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_output_loop(self) -> None:
        """Continuously read lines from the game's stdout into the buffer."""
        try:
            for line in self._proc.stdout:
                self._output_buffer.append(line)
        except Exception:
            pass

    def _collect_output(self) -> str:
        """Wait for the game to emit a prompt and return all buffered text.

        Waits until the output looks like a menu/prompt (ends with a question
        or number-list pattern) or the read timeout expires.
        """
        deadline = time.monotonic() + self.read_timeout
        collected: list[str] = []

        while time.monotonic() < deadline:
            while self._output_buffer:
                collected.append(self._output_buffer.pop(0))
            combined = "".join(collected)
            # Consider output complete when it ends with a prompt-like fragment.
            if self._looks_complete(combined):
                break
            # Also stop if the process has exited.
            if self._proc is not None and self._proc.poll() is not None:
                time.sleep(0.1)
                while self._output_buffer:
                    collected.append(self._output_buffer.pop(0))
                break
            time.sleep(0.05)

        return _strip_ansi("".join(collected)).strip()

    @staticmethod
    def _looks_complete(text: str) -> bool:
        """Return True if *text* appears to be a complete prompt ready for input."""
        stripped = text.strip()
        if not stripped:
            return False
        # The game ends lines with a newline followed by a blank line or a
        # numbered option list.  Detect a trailing newline as the simplest
        # proxy — combined with a brief poll-sleep loop, this works well.
        last_lines = stripped.rsplit("\n", 5)
        last_chunk = "\n".join(last_lines[-3:])
        # Numbered option present or direct question line.
        if re.search(r"\b\d+\.", last_chunk):
            return True
        # "Type" / "Press" / "Pick" indicators.
        if re.search(r"(Type|Press|Pick|Choose|Enter|input)", last_chunk, re.IGNORECASE):
            return True
        return False

    def _compute_reward(self, output: str) -> tuple[float, bool]:
        """Delegate reward computation to :mod:`training.reward`.

        Also updates episode-level statistics (``_won``, ``_floors_cleared``,
        ``_enemies_killed``) used by :meth:`_info`.
        """
        cfg = getattr(self, "_reward_config", None) or RewardConfig()
        reward, done = compute_step_reward(output, cfg)
        if done and reward > 0:
            self._won = True
            logger.info("Episode ended: WIN")
        elif done and reward < 0:
            logger.info("Episode ended: LOSS")
        else:
            # Update cumulative statistics from the reward module's patterns.
            from training.reward import (
                _ENEMY_KILLED_RE,
                _FLOOR_ADVANCE_RE,
            )
            if _ENEMY_KILLED_RE.search(output):
                self._enemies_killed += len(_ENEMY_KILLED_RE.findall(output))
            if _FLOOR_ADVANCE_RE.search(output):
                self._floors_cleared += 1
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
