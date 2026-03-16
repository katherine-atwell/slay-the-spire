"""LLM agent that interacts with the Slay the Spire environment via sts-agent.

The agent receives the current game state (formatted Text the Spire window
content) as a text string, prepends the system prompt, and uses the loaded
Llama 3.2:3B model to generate the next command for sts-agent.

Commands follow the sts-agent CLI format, e.g.::

    "1"         — select option / play card 1
    "end"       — end turn
    "choose 1"  — choose option 1
    "1,2,end"   — play cards 1 and 2 then end turn
    "pot u 1"   — use potion slot 1
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.system_prompt import SYSTEM_PROMPT, get_character_prompt  # noqa: F401

logger = logging.getLogger(__name__)

# Maximum new tokens for the action response.
# sts-agent commands are short (single-line), so we cap generation early.
_MAX_NEW_TOKENS = 64
_TEMPERATURE = 0.7
_TOP_P = 0.9

# Valid single-word sts-agent keywords the model may produce.
_KEYWORD_RE = re.compile(
    r"^(end|proceed|choose|play|map|path|pot)\b", re.IGNORECASE
)


class SlayTheSpireAgent:
    """Wraps a causal LM to produce sts-agent commands from game-state text.

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

        The returned string is a command suitable for ``sts_tool.py --execute``,
        e.g. ``"1"``, ``"end"``, ``"1,2,end"``, ``"pot u 1"``.

        Args:
            game_state: Formatted Text the Spire window content representing
                        the current game state and available choices.

        Returns:
            The agent's action as a short command string.
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
        """Extract the first valid sts-agent command from the model's raw output.

        The model may produce explanatory prose followed by the command on a
        later line.  This method scans all non-empty lines for the first that
        looks like a valid sts-agent command.

        Priority order (applied to each line in order)
        -----------------------------------------------
        1. A comma-separated sequence of sts-agent tokens
           (digits, ``end``, ``choose N``, ``pot …``, etc.) — multi-action.
        2. A line starting with a known keyword (``end``, ``choose``, …).
        3. A leading digit on any line.

        If no line matches the above, the first word of the first non-empty
        line is returned as a fallback.
        """
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        if not lines:
            return raw.strip()

        for candidate in lines:
            # 1. Comma-separated multi-command (e.g. "1,2,end", "1,2,3").
            #    Each comma-separated token must start with a digit or keyword.
            if "," in candidate:
                tokens = [t.strip() for t in candidate.split(",")]
                if tokens and all(
                    t and (t[0].isdigit() or _KEYWORD_RE.match(t)) for t in tokens
                ):
                    return candidate
            # 2. Keyword-led command (e.g. "end", "choose 1", "pot u 1 2", "map 6 4").
            if _KEYWORD_RE.match(candidate):
                return candidate
            # 3. Leading digit — single card/choice or targeted play (e.g. "1", "1 2").
            m = re.match(r"^\s*(\d+(?:\s+\d+)*)", candidate)
            if m:
                return m.group(1).strip()

        # Fallback: first word of first line.
        candidate = lines[0]
        m = re.match(r"^\s*(\S+)", candidate)
        if m:
            return m.group(1)
        return candidate
