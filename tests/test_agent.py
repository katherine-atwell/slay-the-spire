"""Tests for the agent module."""
from __future__ import annotations

import re
import types
import unittest
from unittest.mock import MagicMock, patch

from agent.system_prompt import (
    CHARACTER_PROMPTS,
    DEFECT_SYSTEM_PROMPT,
    IRONCLAD_SYSTEM_PROMPT,
    SILENT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    get_character_prompt,
)


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
        # System prompt now describes sts-agent command syntax instead of a
        # generic navigation section.
        assert "COMMAND FORMAT" in SYSTEM_PROMPT

    def test_non_empty(self):
        assert len(SYSTEM_PROMPT) > 500


# ---------------------------------------------------------------------------
# Character-specific system-prompt tests
# ---------------------------------------------------------------------------


class TestCharacterSystemPrompts(unittest.TestCase):
    """Each character prompt must contain tailored content and shared sections."""

    def _assert_common_sections(self, prompt: str, label: str) -> None:
        for section in ("COMMAND FORMAT", "CORE MECHANICS", "STRATEGY TIPS"):
            assert section in prompt, f"{label}: '{section}' missing"

    def test_ironclad_prompt_non_empty(self):
        assert len(IRONCLAD_SYSTEM_PROMPT) > 500

    def test_ironclad_prompt_contains_common_sections(self):
        self._assert_common_sections(IRONCLAD_SYSTEM_PROMPT, "IRONCLAD")

    def test_ironclad_prompt_contains_character_section(self):
        assert "IRONCLAD" in IRONCLAD_SYSTEM_PROMPT

    def test_ironclad_prompt_contains_key_mechanics(self):
        for keyword in ("Burning Blood", "Strength", "Exhaust", "Barricade"):
            assert keyword in IRONCLAD_SYSTEM_PROMPT, (
                f"Ironclad prompt missing keyword: {keyword}"
            )

    def test_silent_prompt_non_empty(self):
        assert len(SILENT_SYSTEM_PROMPT) > 500

    def test_silent_prompt_contains_common_sections(self):
        self._assert_common_sections(SILENT_SYSTEM_PROMPT, "SILENT")

    def test_silent_prompt_contains_character_section(self):
        assert "SILENT" in SILENT_SYSTEM_PROMPT

    def test_silent_prompt_contains_key_mechanics(self):
        for keyword in ("Ring of the Snake", "Poison", "Shiv", "Catalyst"):
            assert keyword in SILENT_SYSTEM_PROMPT, (
                f"Silent prompt missing keyword: {keyword}"
            )

    def test_defect_prompt_non_empty(self):
        assert len(DEFECT_SYSTEM_PROMPT) > 500

    def test_defect_prompt_contains_common_sections(self):
        self._assert_common_sections(DEFECT_SYSTEM_PROMPT, "DEFECT")

    def test_defect_prompt_contains_character_section(self):
        assert "DEFECT" in DEFECT_SYSTEM_PROMPT

    def test_defect_prompt_contains_key_mechanics(self):
        for keyword in ("Cracked Core", "Orb", "Focus", "Lightning", "Frost"):
            assert keyword in DEFECT_SYSTEM_PROMPT, (
                f"Defect prompt missing keyword: {keyword}"
            )

    def test_character_prompts_dict_has_all_three(self):
        assert set(CHARACTER_PROMPTS.keys()) == {"ironclad", "silent", "defect"}

    def test_character_prompts_dict_values_match_constants(self):
        assert CHARACTER_PROMPTS["ironclad"] is IRONCLAD_SYSTEM_PROMPT
        assert CHARACTER_PROMPTS["silent"] is SILENT_SYSTEM_PROMPT
        assert CHARACTER_PROMPTS["defect"] is DEFECT_SYSTEM_PROMPT

    def test_prompts_are_distinct(self):
        prompts = [IRONCLAD_SYSTEM_PROMPT, SILENT_SYSTEM_PROMPT, DEFECT_SYSTEM_PROMPT]
        for i, p1 in enumerate(prompts):
            for j, p2 in enumerate(prompts):
                if i != j:
                    assert p1 != p2, "Character prompts must not be identical"


class TestGetCharacterPrompt(unittest.TestCase):
    """get_character_prompt() should select the right prompt or fall back."""

    def test_ironclad_exact(self):
        assert get_character_prompt("ironclad") is IRONCLAD_SYSTEM_PROMPT

    def test_silent_exact(self):
        assert get_character_prompt("silent") is SILENT_SYSTEM_PROMPT

    def test_defect_exact(self):
        assert get_character_prompt("defect") is DEFECT_SYSTEM_PROMPT

    def test_case_insensitive_upper(self):
        assert get_character_prompt("Ironclad") is IRONCLAD_SYSTEM_PROMPT
        assert get_character_prompt("SILENT") is SILENT_SYSTEM_PROMPT
        assert get_character_prompt("Defect") is DEFECT_SYSTEM_PROMPT

    def test_unknown_character_falls_back_to_generic(self):
        assert get_character_prompt("watcher") is SYSTEM_PROMPT

    def test_empty_string_falls_back_to_generic(self):
        assert get_character_prompt("") is SYSTEM_PROMPT


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

    def test_end_keyword(self):
        assert self.parse("end") == "end"

    def test_comma_separated_multi_action(self):
        assert self.parse("1,2,end") == "1,2,end"

    def test_choose_command(self):
        assert self.parse("choose 1") == "choose 1"

    def test_pot_command(self):
        assert self.parse("pot u 1") == "pot u 1"

    def test_prose_before_command_stripped(self):
        # Model may explain reasoning then give the command on a new line.
        result = self.parse("I will play Strike first.\nend")
        assert result == "end"


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
