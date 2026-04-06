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
stdin/stdout pipe used for mod communication.  Each game episode also writes
a temporary log file (overwritten at the start of every new game) so that
outputs and errors are available for debugging after the episode completes.
The log-file path defaults to ``<OS temp dir>/slay_the_spire_game.log`` and
can be overridden via the ``logging.log_file`` key in ``config.yaml``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,  # stdout is reserved for the Communication Mod pipe
)
logger = logging.getLogger(__name__)

# Default log-file name placed in the OS temporary directory.
_DEFAULT_LOG_FILENAME = "slay_the_spire_game.log"


def _configure_file_logging(log_path: str) -> None:
    """Attach (or replace) a :class:`logging.FileHandler` on the root logger.

    The file is opened with ``mode='w'`` so it is **overwritten** at the start
    of every game episode, keeping only the most recent run for debugging.
    Any existing :class:`~logging.FileHandler` attached to the root logger is
    removed first to prevent duplicate handlers when :func:`main` is invoked
    more than once in the same process.

    Args:
        log_path: Absolute or relative path of the log file to write.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(file_handler)


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
    # Defer heavy imports so the module can be imported without torch/GPU libs
    # (e.g. in tests or environments that only need the logging helpers).
    from agent.agent import SlayTheSpireAgent  # noqa: PLC0415
    from agent.model import load_model_and_tokenizer  # noqa: PLC0415
    from environment.game_env import make_env  # noqa: PLC0415

    args = _parse_args(argv)

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Always use the communication_mod back-end when launched from this entry
    # point, regardless of what interface_mode is set to in config.yaml.
    cfg.setdefault("environment", {})["interface_mode"] = "communication_mod"

    # Set up per-game file logging.  The file is overwritten at the start of
    # each new game so only the most recent episode is retained for debugging.
    log_path: str = cfg.get("logging", {}).get("log_file") or os.path.join(
        tempfile.gettempdir(), _DEFAULT_LOG_FILENAME
    )
    _configure_file_logging(log_path)
    logger.info("Game log: %s", log_path)

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
