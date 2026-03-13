# slay-the-spire
Finetune an LLM agent to play Slay the Spire

## Overview

This project trains a **Llama 3.2:3B (4-bit quantised)** LLM agent to play the
text-based Slay the Spire clone [slaythetext](https://github.com/Difio3333/slaythetext)
using **reinforcement learning (GRPO — Group Relative Policy Optimisation)**.

The agent:
- Receives the current game-state text from slaythetext as its observation.
- Decides which numbered action to take guided by a system prompt describing
  the game mechanics.
- Is trained with GRPO (via [TRL](https://github.com/huggingface/trl)) to
  maximise a reward signal derived from in-game outcomes (floors cleared,
  enemy kills, HP fraction, win/loss).

## Repository structure

```
slay-the-spire/
├── config.yaml              # Model, LoRA, training and reward hyperparameters
├── requirements.txt         # Python dependencies
├── agent/
│   ├── agent.py             # SlayTheSpireAgent: wraps the LLM for action selection
│   ├── model.py             # 4-bit quantised Llama 3.2:3B loading + LoRA setup
│   └── system_prompt.py     # Detailed game-mechanics system prompt
├── environment/
│   └── game_env.py          # Gym-style subprocess wrapper for slaythetext
├── training/
│   ├── reward.py            # Composable reward functions
│   └── train.py             # GRPO training entry point
└── tests/
    ├── test_agent.py
    ├── test_environment.py
    └── test_reward.py
```

## Requirements

- Python 3.10 or later
- ~64 GB RAM (the 4-bit model uses ~1.7 GB for weights; the rest is for
  activations, the LoRA adapter, and the training loop)
- A copy of [slaythetext](https://github.com/Difio3333/slaythetext) (see below)

### Python packages

```bash
pip install -r requirements.txt
```

### slaythetext

Clone the game and install its dependency:

```bash
git clone https://github.com/Difio3333/slaythetext slaythetext
pip install ansimarkup colorama
```

Then update `config.yaml` → `environment.game_script` to point at
`slaythetext/main.py` (already the default).

### Llama 3.2:3B model access

Request access to the model on Hugging Face and log in:

```bash
huggingface-cli login
```

The model ID used is `meta-llama/Llama-3.2-3B-Instruct` (set in `config.yaml`).

## Running

### Training

```bash
python -m training.train --config config.yaml
```

The script will:
1. Load the 4-bit quantised Llama 3.2:3B model.
2. Attach a LoRA adapter for parameter-efficient training.
3. Collect game rollouts using a random policy to build an initial dataset.
4. Fine-tune the model with GRPO, saving checkpoints to `./checkpoints/`.

### Inference (playing a game)

```python
from agent.model import load_model_and_tokenizer
from agent.agent import SlayTheSpireAgent
from environment.game_env import SlayTheSpireEnv

model, tokenizer = load_model_and_tokenizer(
    config_path="config.yaml",
    adapter_path="./checkpoints/final_adapter",  # omit to use base model
)
agent = SlayTheSpireAgent(model, tokenizer)
env = SlayTheSpireEnv(game_script="slaythetext/main.py")

state = env.reset()
done = False
while not done:
    action = agent.act(state)
    state, reward, done, info = env.step(action)
    print(f"Action: {action!r}  Reward: {reward:.1f}  Info: {info}")

env.close()
```

## Running tests

```bash
python -m pytest tests/ -v
```

## Configuration

All hyperparameters live in `config.yaml`.  Key sections:

| Section | Notable keys |
|---------|-------------|
| `model` | `name`, `load_in_4bit`, `bnb_4bit_quant_type` |
| `peft` | `r` (LoRA rank), `lora_alpha`, `target_modules` |
| `training` | `num_train_epochs`, `num_generations` (GRPO group size), `learning_rate` |
| `environment` | `game_script`, `max_turns`, `read_timeout` |
| `reward` | `win_bonus`, `floor_cleared_bonus`, `enemy_killed_bonus`, … |

## System prompt

The system prompt in `agent/system_prompt.py` covers:
- All three characters (Ironclad, Silent, Defect) and their starter relics
- Combat fundamentals: energy, cards, block, turn structure
- Status effects: Vulnerable, Weak, Frail, Poison, Strength, Dexterity
- Orb mechanics (Defect)
- Map navigation, shops, rest sites, elites, bosses, potions, and keys
- Strategy tips for optimal play
