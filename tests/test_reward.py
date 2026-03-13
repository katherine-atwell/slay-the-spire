"""Tests for the reward functions."""
from __future__ import annotations

import unittest

from training.reward import (
    RewardConfig,
    compute_step_reward,
    enemy_kill_reward,
    floor_advance_reward,
    hp_fraction_reward,
    invalid_action_penalty,
    win_loss_reward,
)


class TestWinLossReward(unittest.TestCase):
    def setUp(self):
        self.cfg = RewardConfig()

    def test_win_detected(self):
        reward, terminal = win_loss_reward("You won!", self.cfg)
        self.assertGreater(reward, 0)
        self.assertTrue(terminal)

    def test_win_heart(self):
        reward, terminal = win_loss_reward("You won and beat the Heart!!", self.cfg)
        self.assertGreater(reward, 0)
        self.assertTrue(terminal)

    def test_loss_detected(self):
        reward, terminal = win_loss_reward("You have lost!", self.cfg)
        self.assertLess(reward, 0)
        self.assertTrue(terminal)

    def test_thanks_for_playing(self):
        reward, terminal = win_loss_reward("Thanks so much for playing!", self.cfg)
        self.assertLess(reward, 0)
        self.assertTrue(terminal)

    def test_neutral_output(self):
        reward, terminal = win_loss_reward("The Cultist attacks for 6 damage.", self.cfg)
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminal)

    def test_case_insensitive(self):
        reward, terminal = win_loss_reward("you won!", self.cfg)
        self.assertTrue(terminal)


class TestEnemyKillReward(unittest.TestCase):
    def setUp(self):
        self.cfg = RewardConfig(enemy_killed_bonus=5.0)

    def test_single_kill(self):
        reward = enemy_kill_reward("The Cultist has been defeated.", self.cfg)
        self.assertEqual(reward, 5.0)

    def test_multiple_kills(self):
        text = (
            "The Cultist has been defeated.\n"
            "The Jaw Worm has been defeated."
        )
        reward = enemy_kill_reward(text, self.cfg)
        self.assertEqual(reward, 10.0)

    def test_minions_flee(self):
        reward = enemy_kill_reward("All other Minions are fleeing!", self.cfg)
        self.assertEqual(reward, 5.0)

    def test_no_kill(self):
        reward = enemy_kill_reward("You play Strike for 6 damage.", self.cfg)
        self.assertEqual(reward, 0.0)


class TestFloorAdvanceReward(unittest.TestCase):
    def setUp(self):
        self.cfg = RewardConfig(floor_cleared_bonus=10.0)

    def test_floor_advance(self):
        reward = floor_advance_reward("You have entered floor 3.", self.cfg)
        self.assertEqual(reward, 10.0)

    def test_no_advance(self):
        reward = floor_advance_reward("You gain 3 gold.", self.cfg)
        self.assertEqual(reward, 0.0)


class TestHpFractionReward(unittest.TestCase):
    def setUp(self):
        self.cfg = RewardConfig(hp_fraction_bonus_scale=20.0)

    def test_full_hp(self):
        reward = hp_fraction_reward("HP: 66/66", self.cfg)
        self.assertAlmostEqual(reward, 20.0)

    def test_half_hp(self):
        reward = hp_fraction_reward("HP: 33/66", self.cfg)
        self.assertAlmostEqual(reward, 10.0)

    def test_no_hp_info(self):
        reward = hp_fraction_reward("You see a corridor.", self.cfg)
        self.assertEqual(reward, 0.0)

    def test_zero_max_hp_no_crash(self):
        reward = hp_fraction_reward("HP: 0/0", self.cfg)
        self.assertEqual(reward, 0.0)


class TestInvalidActionPenalty(unittest.TestCase):
    def setUp(self):
        self.cfg = RewardConfig(invalid_action_penalty=-2.0)

    def test_invalid_detected(self):
        reward = invalid_action_penalty("You have to type a number.", self.cfg)
        self.assertEqual(reward, -2.0)

    def test_valid_output(self):
        reward = invalid_action_penalty("You play Defend for 5 block.", self.cfg)
        self.assertEqual(reward, 0.0)


class TestComputeStepReward(unittest.TestCase):
    def test_win_short_circuits(self):
        reward, terminal = compute_step_reward("You won!")
        self.assertGreater(reward, 0)
        self.assertTrue(terminal)

    def test_loss_short_circuits(self):
        reward, terminal = compute_step_reward("You have lost!")
        self.assertLess(reward, 0)
        self.assertTrue(terminal)

    def test_incremental_rewards_combined(self):
        text = "The Cultist has been defeated.\nHP: 50/66"
        reward, terminal = compute_step_reward(text)
        self.assertFalse(terminal)
        # At minimum the kill bonus should be present.
        self.assertGreater(reward, 0)

    def test_default_config_used_when_none(self):
        # Should not raise even with default config.
        reward, terminal = compute_step_reward("Random game text.", cfg=None)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminal, bool)

    def test_custom_config(self):
        cfg = RewardConfig(enemy_killed_bonus=100.0)
        reward, _ = compute_step_reward("The Cultist has been defeated.", cfg=cfg)
        self.assertEqual(reward, 100.0)


if __name__ == "__main__":
    unittest.main()
