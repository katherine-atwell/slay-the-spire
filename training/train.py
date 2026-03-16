"""Reinforcement-learning training loop for the Slay the Spire LLM agent.

Algorithm
---------
We use **GRPO (Group Relative Policy Optimisation)** from TRL, which:
1. Samples *G* completions from the policy for each game-state prompt.
2. Scores each completion with a reward function derived from the game.
3. Optimises the policy to increase the probability of high-reward actions
   relative to the group average — without requiring a separate critic model.

This choice keeps the memory footprint well within the 64 GB RAM budget
because we never load a second reference model during PPO clip computation.

Running
-------
::

    python -m training.train --config config.yaml

Optional arguments
~~~~~~~~~~~~~~~~~~
--config   Path to the YAML config file (default: ``config.yaml``).
--episodes Number of game episodes to collect before each gradient update.
           Defaults to the value in ``config.yaml``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_cfg(path: str) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _build_reward_fn(reward_cfg: dict):
    """Return a TRL-compatible reward function closed over *reward_cfg*.

    TRL's GRPOTrainer expects a reward function with the signature::

        reward_fn(prompts, completions, **kwargs) -> list[float]
    """
    from training.reward import RewardConfig, compute_step_reward

    cfg = RewardConfig(
        win_bonus=reward_cfg.get("win_bonus", 200.0),
        loss_penalty=reward_cfg.get("loss_penalty", -50.0),
        floor_cleared_bonus=reward_cfg.get("floor_cleared_bonus", 10.0),
        hp_fraction_bonus_scale=reward_cfg.get("hp_fraction_bonus_scale", 20.0),
        enemy_killed_bonus=reward_cfg.get("enemy_killed_bonus", 5.0),
        invalid_action_penalty=reward_cfg.get("invalid_action_penalty", -2.0),
    )

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        """Score each (prompt, completion) pair.

        During GRPO, the *completion* is the action string the model produced.
        We use the *prompt* (the game-state text) combined with the completion
        as a proxy for what the game would display next, and score it with the
        step-level reward function.

        In a full online RL setup the environment would actually be stepped
        and the resulting output scored.  Here we use a lightweight heuristic
        that rewards grammatical, numeric responses and penalises obviously
        invalid ones — suitable for supervised warm-up / offline RL.
        """
        rewards = []
        for prompt, completion in zip(prompts, completions, strict=True):
            # Reward the completion as if it were the game output that followed.
            combined = prompt + "\n" + completion
            score, _ = compute_step_reward(combined, cfg)
            # Minimum reward for any valid-looking numeric response (exploration bonus).
            import re
            if re.match(r"^\s*\d+", completion.strip()):
                score += 1.0
            rewards.append(score)
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def _build_dataset(env_cfg: dict, n_episodes: int = 10, system_prompt: str = ""):
    """Roll out *n_episodes* of the game with a random policy to collect prompts.

    Returns a HuggingFace ``Dataset`` with a single ``"prompt"`` column
    containing the game-state strings observed during those episodes.
    Each entry is formatted as a chat-style message list so that the GRPO
    trainer can apply the tokenizer's chat template directly.

    In production, replace this with the online GRPO loop that rolls out the
    current policy.
    """
    import random
    from datasets import Dataset
    from environment.game_env import StsAgentEnv

    sts_tool_path = env_cfg.get("sts_tool_path", "sts-agent/src/sts_tool.py")
    python_executable = env_cfg.get("python_executable", "python.exe")
    windows = env_cfg.get("windows", None)
    max_turns = env_cfg.get("max_turns", 200)
    command_timeout = float(env_cfg.get("command_timeout", 5.0))

    env = StsAgentEnv(
        sts_tool_path=sts_tool_path,
        python_executable=python_executable,
        windows=windows,
        max_turns=max_turns,
        command_timeout=command_timeout,
    )

    # Simple random actions compatible with sts-agent.
    _random_actions = ["1", "2", "3", "4", "5", "end"]

    prompts: list[list[dict]] = []
    for ep in range(n_episodes):
        logger.info("Collecting episode %d / %d …", ep + 1, n_episodes)
        try:
            state = env.reset()
            done = False
            while not done:
                messages: list[dict] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": state})
                prompts.append(messages)
                action = random.choice(_random_actions)
                state, _, done, _ = env.step(action)
        except Exception as exc:
            logger.warning("Episode %d failed: %s", ep + 1, exc)
    env.close()

    return Dataset.from_dict({"prompt": prompts})


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(config_path: str = "config.yaml") -> None:
    """Load config, initialise model + trainer, and run GRPO training.

    Args:
        config_path: Path to the YAML configuration file.
    """
    cfg = _load_cfg(config_path)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    env_cfg = cfg["environment"]
    reward_cfg = cfg["reward"]

    # ------------------------------------------------------------------
    # 1. Load model + tokeniser
    # ------------------------------------------------------------------
    from agent.model import load_model_and_tokenizer, prepare_model_for_training
    from agent.system_prompt import get_character_prompt

    character = cfg.get("agent", {}).get("character", "")
    system_prompt = get_character_prompt(character)
    resolved = character.lower() if character.lower() in ("ironclad", "silent", "defect") else "generic"
    logger.info("Using system prompt for character: %s", resolved)

    logger.info("Loading model …")
    model, tokenizer = load_model_and_tokenizer(config_path=config_path)
    model = prepare_model_for_training(model, config_path=config_path)

    # ------------------------------------------------------------------
    # 2. Build dataset (offline rollouts with random policy)
    # ------------------------------------------------------------------
    logger.info("Collecting game rollouts for dataset …")
    dataset = _build_dataset(env_cfg, n_episodes=10, system_prompt=system_prompt)
    logger.info("Dataset size: %d prompts", len(dataset))

    # ------------------------------------------------------------------
    # 3. Configure GRPOTrainer
    # ------------------------------------------------------------------
    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        output_dir=train_cfg.get("output_dir", "./checkpoints"),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        num_generations=train_cfg.get("num_generations", 4),
        max_prompt_length=train_cfg.get("max_prompt_length", 1024),
        max_completion_length=train_cfg.get("max_completion_length", 256),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        seed=train_cfg.get("seed", 42),
    )

    reward_fn = _build_reward_fn(reward_cfg)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
    )

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    logger.info("Starting GRPO training …")
    trainer.train()

    # ------------------------------------------------------------------
    # 5. Save the final adapter
    # ------------------------------------------------------------------
    output_dir = train_cfg.get("output_dir", "./checkpoints")
    final_dir = os.path.join(output_dir, "final_adapter")
    logger.info("Saving adapter to %s", final_dir)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Slay the Spire LLM agent with GRPO."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    train(config_path=args.config)


if __name__ == "__main__":
    main()
