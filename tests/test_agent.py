"""Tests for the agent module."""
from __future__ import annotations

import re
import types
import unittest
from unittest.mock import MagicMock, patch

from agent.system_prompt import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _make_fake_model_and_tokenizer(response_text: str = "3"):
    """Return minimal mock objects that satisfy SlayTheSpireAgent's interface."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = 1
    # apply_chat_template returns a simple string.
    tokenizer.apply_chat_template.return_value = "PROMPT"

    # tokenizer(...) returns an object with .input_ids attribute.
    mock_enc = MagicMock()
    mock_enc.input_ids.shape = (1, 10)

    def fake_call(text, **kwargs):
        mock_enc.input_ids = MagicMock()
        mock_enc.input_ids.__getitem__ = lambda self, idx: MagicMock()
        mock_enc.input_ids.shape = [1, 10]
        mock_enc.input_ids.to = lambda dev: mock_enc.input_ids
        return mock_enc

    tokenizer.side_effect = fake_call

    # Decoding returns the desired response text.
    tokenizer.decode.return_value = response_text

    model = MagicMock()
    # model.generate returns a tensor-like object.
    import torch

    generated = torch.zeros((1, 12), dtype=torch.long)
    model.generate.return_value = generated
    # make next(model.parameters()) work
    fake_param = torch.zeros(1)
    model.parameters.return_value = iter([fake_param])

    return model, tokenizer


# ---------------------------------------------------------------------------
# System-prompt tests
# ---------------------------------------------------------------------------


class TestSystemPrompt(unittest.TestCase):
    def test_contains_core_mechanics(self):
        assert "CORE MECHANICS" in SYSTEM_PROMPT

    def test_contains_characters(self):
        for char in ("Ironclad", "Silent", "Defect"):
            assert char in SYSTEM_PROMPT, f"{char} missing from system prompt"

    def test_contains_strategy_tips(self):
        assert "STRATEGY TIPS" in SYSTEM_PROMPT

    def test_navigation_section(self):
        assert "NAVIGATION" in SYSTEM_PROMPT

    def test_non_empty(self):
        assert len(SYSTEM_PROMPT) > 500


# ---------------------------------------------------------------------------
# Agent._parse_action tests
# ---------------------------------------------------------------------------


class TestParseAction(unittest.TestCase):
    def setUp(self):
        from agent.agent import SlayTheSpireAgent

        self.parse = SlayTheSpireAgent._parse_action

    def test_leading_digit(self):
        assert self.parse("3") == "3"

    def test_digit_with_trailing_text(self):
        assert self.parse("2 — play Strike") == "2"

    def test_keyword(self):
        assert self.parse("Save") == "Save"

    def test_whitespace_stripped(self):
        assert self.parse("  1  ") == "1"

    def test_empty_string(self):
        # Empty string should return empty (no crash).
        result = self.parse("")
        assert isinstance(result, str)

    def test_multi_digit(self):
        assert self.parse("12") == "12"


# ---------------------------------------------------------------------------
# Agent.act integration (mocked model)
# ---------------------------------------------------------------------------


class TestSlayTheSpireAgentAct(unittest.TestCase):
    def test_act_returns_string(self):
        import torch
        from agent.agent import SlayTheSpireAgent

        model, tokenizer = _make_fake_model_and_tokenizer("2")
        # Patch torch.no_grad to be a no-op context manager.
        agent = SlayTheSpireAgent(model, tokenizer, device="cpu")

        game_state = "You see 3 options:\n1. Strike\n2. Defend\n3. End Turn"
        result = agent.act(game_state)
        assert isinstance(result, str)

    def test_build_messages_structure(self):
        from agent.agent import SlayTheSpireAgent

        model, tokenizer = _make_fake_model_and_tokenizer()
        agent = SlayTheSpireAgent(model, tokenizer, device="cpu")
        msgs = agent.build_messages("some game state")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "some game state"

    def test_custom_system_prompt(self):
        from agent.agent import SlayTheSpireAgent

        model, tokenizer = _make_fake_model_and_tokenizer()
        custom = "Custom prompt."
        agent = SlayTheSpireAgent(model, tokenizer, system_prompt=custom, device="cpu")
        msgs = agent.build_messages("state")
        assert msgs[0]["content"] == custom


if __name__ == "__main__":
    unittest.main()
