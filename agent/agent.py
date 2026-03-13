"""LLM agent that interacts with the slaythetext game environment.

The agent receives the current game state as a text string, prepends the system
prompt, and uses the loaded Llama 3.2:3B model to generate the next action.
The action is expected to be a single number (or short keyword) matching one of
the numbered options presented by slaythetext.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Maximum new tokens for the action response.
# Actions are short (usually a single digit), so we cap generation early.
_MAX_NEW_TOKENS = 64
_TEMPERATURE = 0.7
_TOP_P = 0.9


class SlayTheSpireAgent:
    """Wraps a causal LM to produce game actions from raw game-state text.

    Args:
        model: A loaded (optionally LoRA-adapted) causal language model.
        tokenizer: The corresponding tokeniser.
        system_prompt: Override the default system prompt if desired.
        max_new_tokens: Maximum tokens to generate per action.
        temperature: Sampling temperature (lower → more deterministic).
        top_p: Nucleus sampling probability threshold.
        device: Explicit device string (``"cuda"``, ``"cpu"`` …).
            Defaults to the model's first parameter device.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: int = _MAX_NEW_TOKENS,
        temperature: float = _TEMPERATURE,
        top_p: float = _TOP_P,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, game_state: str) -> str:
        """Given the current game-state text, return the chosen action string.

        The returned string is the raw model output, trimmed of whitespace.
        The caller (or the environment) is responsible for parsing it into a
        valid game command.

        Args:
            game_state: Raw text emitted by slaythetext representing the current
                        game state and available choices.

        Returns:
            The agent's action as a short string (typically a digit).
        """
        prompt = self._build_prompt(game_state)
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).input_ids.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (skip the prompt).
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        raw_response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        action = self._parse_action(raw_response)
        logger.debug("Game state snippet: %.120s … → action: %r", game_state, action)
        return action

    def build_messages(self, game_state: str) -> list[dict]:
        """Build a chat-style message list for use with TRL trainers.

        Returns a list in the OpenAI chat format:
        ``[{"role": "system", ...}, {"role": "user", ...}]``
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": game_state},
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, game_state: str) -> str:
        """Assemble the full prompt string sent to the model.

        Uses the chat template when the tokenizer exposes one; falls back to
        a simple concatenation otherwise.
        """
        messages = self.build_messages(game_state)
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        # Fallback
        return (
            f"<|system|>\n{self.system_prompt}\n"
            f"<|user|>\n{game_state}\n"
            "<|assistant|>\n"
        )

    @staticmethod
    def _parse_action(raw: str) -> str:
        """Extract the first valid action token from the model's raw output.

        Preference order:
        1. Leading digit (most common — numbered choice).
        2. First alphabetic word (e.g. "Save", "End").
        3. The full stripped response as a fallback.
        """
        stripped = raw.strip()
        # Prefer a leading number (possibly multi-digit for shops/large lists).
        m = re.match(r"^\s*(\d+)", stripped)
        if m:
            return m.group(1)
        # Accept a leading keyword if no digit found.
        m = re.match(r"^\s*([A-Za-z]+)", stripped)
        if m:
            return m.group(1)
        return stripped
