"""Reward functions for the Slay the Spire RL agent.

Each reward function takes the raw output text from a game step and
returns a scalar reward signal.  Functions are composable — the trainer
calls :func:`compute_step_reward` which aggregates all signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class RewardConfig:
    """Weights for each reward component.

    Attributes
    ----------
    win_bonus:
        Large positive reward for winning the game.
    loss_penalty:
        Large negative reward for losing.
    floor_cleared_bonus:
        Reward for each floor the player advances.
    hp_fraction_bonus_scale:
        Multiplied by (current_hp / max_hp) when available in output.
        Encourages finishing with high HP.
    enemy_killed_bonus:
        Small bonus for each enemy kill detected in the output.
    invalid_action_penalty:
        Penalty whenever the game indicates the last input was not understood.
    """

    win_bonus: float = 200.0
    loss_penalty: float = -100.0
    floor_cleared_bonus: float = 10.0
    hp_fraction_bonus_scale: float = 20.0
    enemy_killed_bonus: float = 5.0
    invalid_action_penalty: float = -2.0


# ---------------------------------------------------------------------------
# Pattern constants
# ---------------------------------------------------------------------------

_WIN_RE = re.compile(r"you won", re.IGNORECASE)
_LOSS_RE = re.compile(r"(you have lost|thanks so much for playing)", re.IGNORECASE)
_ENEMY_KILLED_RE = re.compile(
    r"(has been defeated|all other minions are fleeing)", re.IGNORECASE
)
_FLOOR_ADVANCE_RE = re.compile(
    r"(you have entered|you arrive at|floor \d+|after the battle|you proceed)",
    re.IGNORECASE,
)
_INVALID_RE = re.compile(
    r"(you have to type|invalid|that is not|i didn.t understand)",
    re.IGNORECASE,
)
# Match "HP: 45/66" or "45 / 66 HP" or "45/66 hp".
_HP_RE = re.compile(
    r"hp[:\s]+(\d+)\s*/\s*(\d+)|(\d+)\s*/\s*(\d+)\s*hp",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Individual signal extractors
# ---------------------------------------------------------------------------


def win_loss_reward(output: str, cfg: RewardConfig) -> tuple[float, bool]:
    """Return ``(reward, is_terminal)`` based on win/loss detection."""
    if _WIN_RE.search(output):
        return cfg.win_bonus, True
    if _LOSS_RE.search(output):
        return cfg.loss_penalty, True
    return 0.0, False


def enemy_kill_reward(output: str, cfg: RewardConfig) -> float:
    """Count enemy-kill events and return cumulative reward."""
    kills = len(_ENEMY_KILLED_RE.findall(output))
    return kills * cfg.enemy_killed_bonus


def floor_advance_reward(output: str, cfg: RewardConfig) -> float:
    """Return a single floor-advance bonus if a floor-advance event is detected.

    Each game step represents at most one floor transition, so we use
    ``re.search`` (boolean) rather than ``findall`` to avoid double-counting
    when the output contains multiple matching phrases (e.g. both
    "You have entered" and "floor 3" in the same line).
    """
    if _FLOOR_ADVANCE_RE.search(output):
        return cfg.floor_cleared_bonus
    return 0.0


def hp_fraction_reward(output: str, cfg: RewardConfig) -> float:
    """Reward the agent proportionally to the player's remaining HP fraction.

    Parses strings like ``"HP: 45/66"`` or ``"45/66 HP"`` from the game output.
    Returns 0 if no HP information is found.
    """
    match = _HP_RE.search(output)
    if match:
        # Group layout: (grp1, grp2) for "HP: X/Y", (grp3, grp4) for "X/Y HP"
        if match.group(1) is not None:
            current_hp, max_hp = int(match.group(1)), int(match.group(2))
        else:
            current_hp, max_hp = int(match.group(3)), int(match.group(4))
        if max_hp > 0:
            return cfg.hp_fraction_bonus_scale * (current_hp / max_hp)
    return 0.0


def invalid_action_penalty(output: str, cfg: RewardConfig) -> float:
    """Return a negative reward if the game reported an invalid input."""
    if _INVALID_RE.search(output):
        return cfg.invalid_action_penalty
    return 0.0


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------


def compute_step_reward(
    output: str,
    cfg: RewardConfig | None = None,
) -> tuple[float, bool]:
    """Aggregate all reward signals for a single game step.

    Args:
        output: Raw (ANSI-stripped) text from one game step.
        cfg: Reward weights.  Uses :class:`RewardConfig` defaults when ``None``.

    Returns:
        ``(total_reward, is_terminal)`` where ``is_terminal`` is ``True`` if
        a win or loss was detected.
    """
    if cfg is None:
        cfg = RewardConfig()

    terminal_reward, is_terminal = win_loss_reward(output, cfg)
    if is_terminal:
        return terminal_reward, True

    reward = 0.0
    reward += enemy_kill_reward(output, cfg)
    reward += floor_advance_reward(output, cfg)
    reward += hp_fraction_reward(output, cfg)
    reward += invalid_action_penalty(output, cfg)
    return reward, False
