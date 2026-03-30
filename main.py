"""Entry point executed by the Communication Mod on startup.

The `Communication Mod <https://github.com/ForgottenArbiter/CommunicationMod>`_
is configured to launch an external process when Slay the Spire starts.  This
script is that process.  It communicates with the mod via stdin/stdout using
newline-delimited JSON messages, making it compatible with **macOS, Linux, and
Windows** — no platform-specific UI-automation libraries are required.

Configure the Communication Mod to run::

    python main.py

or with a custom config file::

    python main.py --config /path/to/config.yaml

What this script does
---------------------
1. Parse ``config.yaml`` (or the path supplied via ``--config``).
2. Load the quantised LLM and attach the LoRA adapter (if configured).
3. Create a :class:`~environment.game_env.CommunicationModEnv` that reads game
   state from stdin and writes commands to stdout.
4. Play one episode: call the agent on each state until the game ends.

Logging is written to **stderr** so it does not interfere with the
stdin/stdout pipe used for mod communication.
"""

from __future__ import annotations

import argparse
import logging
import sys

import yaml

from agent.agent import SlayTheSpireAgent
from agent.model import load_model_and_tokenizer
from environment.game_env import make_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,  # stdout is reserved for the Communication Mod pipe
)
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slay the Spire bot — Communication Mod entry point",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to config.yaml (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load the agent and run one game episode via the Communication Mod."""
    args = _parse_args(argv)

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Always use the communication_mod back-end when launched from this entry
    # point, regardless of what interface_mode is set to in config.yaml.
    cfg.setdefault("environment", {})["interface_mode"] = "communication_mod"

    logger.info("Loading model from %s …", cfg.get("model", {}).get("name", "config"))
    model, tokenizer = load_model_and_tokenizer(config_path=args.config)
    agent = SlayTheSpireAgent(model, tokenizer)
    logger.info("Model loaded.")

    env = make_env(cfg["environment"])

    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        logger.info("Action: %r", action)
        state, reward, done, info = env.step(action)
        logger.info("Reward: %.2f  Done: %s  Info: %s", reward, done, info)

    env.close()
    logger.info("Episode finished: %s", info)


if __name__ == "__main__":
    main()
