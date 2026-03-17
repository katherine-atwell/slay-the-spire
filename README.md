# slay-the-spire
Finetune an LLM agent to play Slay the Spire

## Overview

This project trains a **Llama 3.2:3B (4-bit quantised)** LLM agent to play
Slay the Spire using **reinforcement learning (GRPO — Group Relative Policy
Optimisation)**.

The agent interfaces with the game through
[sts-agent](https://github.com/ohylli/sts-agent), which bridges AI agents to
a running Slay the Spire instance via the
[Text the Spire](https://github.com/Wensber/TextTheSpire) accessibility mod.
Game state is read from multiple labelled text windows (Player, Hand, Monster,
Choices, Map) and commands are sent through the mod's prompt interface.

The agent:
- Receives formatted Text the Spire window content as its observation.
- Decides which sts-agent command to issue (e.g. `"1"`, `"end"`, `"1,2,end"`)
  guided by a system prompt describing the game mechanics and command format.
- Is trained with GRPO (via [TRL](https://github.com/huggingface/trl)) to
  maximise a reward signal derived from in-game outcomes (floors cleared,
  enemy kills, HP fraction, win/loss).

## Quick Start

The steps below cover everything from a fresh clone to a running training loop.

**Prerequisites:** Windows (or WSL with a Windows Python) **for the default
`sts_agent` interface mode**.  If you set `interface_mode: "text_the_spire"` in
`config.yaml`, the agent communicates with the mod through plain files and runs
on **any platform** (Linux, macOS, Windows) — see
[Interface modes](#interface-modes) below.

### 1. Clone this repo and install Python dependencies

```bash
git clone https://github.com/katherine-atwell/slay-the-spire
cd slay-the-spire
pip install -r requirements.txt
```

### 2. Set up sts-agent

```bash
git clone https://github.com/ohylli/sts-agent sts-agent
# On Windows:
pip install -r sts-agent/requirements.txt
# On WSL (use the Windows Python):
pip.exe install -r sts-agent/requirements.txt
```

### 3. Authenticate with Hugging Face (for the Llama model)

```bash
huggingface-cli login
```

Request access to `meta-llama/Llama-3.2-3B-Instruct` on Hugging Face if you
have not already done so.

### 4. Configure paths

Open `config.yaml` and verify (or update) the `environment` section:

```yaml
environment:
  # "sts_agent" (default, Windows only) or "text_the_spire" (cross-platform)
  interface_mode: "sts_agent"

  # sts_agent mode — path to sts_tool.py and Python executable
  sts_tool_path: "sts-agent/src/sts_tool.py"
  python_executable: "python.exe"   # "python.exe" on WSL, "python" on native Windows

  # text_the_spire mode — directory of state files and command input file
  text_the_spire_state_dir: "."
  text_the_spire_input_file: "sts_input.txt"
```

### 5. Start the game

Launch Slay the Spire with the **Text the Spire** mod active.  The mod creates
the accessibility windows that sts-agent reads from.

### 6. Train

```bash
python -m training.train --config config.yaml
```

Checkpoints are written to `./checkpoints/`.  To resume or use a different
config, pass `--config path/to/config.yaml`.

### 7. Play (inference)

```python
import yaml
from agent.model import load_model_and_tokenizer
from agent.agent import SlayTheSpireAgent
from environment.game_env import make_env

model, tokenizer = load_model_and_tokenizer(
    config_path="config.yaml",
    adapter_path="./checkpoints/final_adapter",  # omit to use the base model
)
agent = SlayTheSpireAgent(model, tokenizer)
cfg = yaml.safe_load(open("config.yaml"))
env = make_env(cfg["environment"])   # respects interface_mode in config.yaml

state = env.reset()
done = False
while not done:
    action = agent.act(state)
    state, reward, done, info = env.step(action)
    print(f"Action: {action!r}  Reward: {reward:.1f}  Info: {info}")

env.close()
```



```
slay-the-spire/
├── config.yaml              # Model, LoRA, training and reward hyperparameters
├── requirements.txt         # Python dependencies
├── agent/
│   ├── agent.py             # SlayTheSpireAgent: wraps the LLM for action selection
│   ├── model.py             # 4-bit quantised Llama 3.2:3B loading + LoRA setup
│   └── system_prompt.py     # Game-mechanics system prompt + sts-agent command syntax
├── environment/
│   └── game_env.py          # Gym-style env; supports sts_agent and text_the_spire modes
├── training/
│   ├── reward.py            # Composable reward functions
│   └── train.py             # GRPO training entry point
└── tests/
    ├── test_agent.py
    ├── test_environment.py
    └── test_reward.py
```

## Interface modes

The agent can communicate with Slay the Spire in two ways, controlled by
`environment.interface_mode` in `config.yaml`.

### `sts_agent` (default, Windows only)

Uses the [sts-agent](https://github.com/ohylli/sts-agent) CLI tool, which
reads game state from the Windows accessibility windows created by the Text
the Spire mod (via pywinauto / pywin32).

Set up:

```bash
git clone https://github.com/ohylli/sts-agent sts-agent
pip install -r sts-agent/requirements.txt    # or pip.exe in WSL
```

Then in `config.yaml`:

```yaml
environment:
  interface_mode: "sts_agent"
  sts_tool_path: "sts-agent/src/sts_tool.py"
  python_executable: "python.exe"   # "python.exe" on WSL, "python" on Windows
```

### `text_the_spire` (cross-platform)

Interfaces directly with the [Text the Spire](https://github.com/Wensber/TextTheSpire)
mod by reading per-window state files the mod writes to a directory and by
writing commands to a shared input file that the mod monitors.  This back-end
has **no Windows-only dependencies** and works on Linux, macOS, and Windows.

Configure the mod to write state files to a directory (e.g. `/tmp/sts_state`)
and monitor a command file (e.g. `/tmp/sts_state/sts_input.txt`), then update
`config.yaml`:

```yaml
environment:
  interface_mode: "text_the_spire"
  text_the_spire_state_dir: "/tmp/sts_state"    # directory with Player.txt, Hand.txt, …
  text_the_spire_input_file: "/tmp/sts_state/sts_input.txt"
```

The mod writes one `.txt` file per window (e.g. `Player.txt`, `Hand.txt`,
`Monster.txt`, `Choices.txt`, `Map.txt`) and the agent writes the chosen
action as a single line to `sts_input.txt`.

## Requirements

- **Windows** required only for the default `sts_agent` interface mode
  (sts-agent uses pywinauto/pywin32 for UI automation).  The
  `text_the_spire` mode works on any OS.
  On WSL, invoke sts-agent with `python.exe` (the Windows Python).
- Python 3.10 or later
- ~64 GB RAM (the 4-bit model uses ~1.7 GB for weights; the rest is for
  activations, the LoRA adapter, and the training loop)
- Slay the Spire with [Text the Spire mod](https://github.com/Wensber/TextTheSpire) installed and running

### Python packages

```bash
pip install -r requirements.txt
```

### sts-agent

Only needed for the `sts_agent` interface mode.  Clone the interface tool and
install its dependencies:

```bash
git clone https://github.com/ohylli/sts-agent sts-agent
pip.exe install -r sts-agent/requirements.txt   # use pip.exe in WSL
```

Then update `config.yaml` → `environment.sts_tool_path` (already defaults to
`sts-agent/src/sts_tool.py`) and set `environment.python_executable` to
`python.exe` if running from WSL.

### Llama 3.2:3B model access

Request access to the model on Hugging Face and log in:

```bash
huggingface-cli login
```

The model ID used is `meta-llama/Llama-3.2-3B-Instruct` (set in `config.yaml`).

## Running

### Training

Start Slay the Spire with Text the Spire, then:

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
import yaml
from agent.model import load_model_and_tokenizer
from agent.agent import SlayTheSpireAgent
from environment.game_env import make_env

model, tokenizer = load_model_and_tokenizer(
    config_path="config.yaml",
    adapter_path="./checkpoints/final_adapter",  # omit to use base model
)
agent = SlayTheSpireAgent(model, tokenizer)

cfg = yaml.safe_load(open("config.yaml"))
env = make_env(cfg["environment"])   # respects interface_mode in config.yaml

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
| `environment` | `interface_mode`, `sts_tool_path`, `python_executable`, `text_the_spire_state_dir`, `text_the_spire_input_file`, `windows`, `max_turns`, `command_timeout` |
| `reward` | `win_bonus`, `floor_cleared_bonus`, `enemy_killed_bonus`, … |

## System prompt

The system prompt in `agent/system_prompt.py` covers:
- sts-agent command format and the labelled Text the Spire windows
- All three characters (Ironclad, Silent, Defect) and their starter relics
- Combat fundamentals: energy, cards, block, turn structure
- Status effects: Vulnerable, Weak, Frail, Poison, Strength, Dexterity
- Orb mechanics (Defect)
- Map navigation, shops, rest sites, elites, bosses, potions, and keys
- Strategy tips for optimal play

