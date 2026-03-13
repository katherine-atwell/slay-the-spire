"""Load a 4-bit quantised Llama 3.2:3B model together with a LoRA adapter.

The quantisation configuration targets machines with ~64 GB of system RAM.
NF4 4-bit quantisation roughly halves the memory footprint (≈1.7 GB for the
base weights) while the LoRA adapter adds minimal overhead.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load training/model configuration from a YAML file."""
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def build_bnb_config(model_cfg: dict) -> BitsAndBytesConfig:
    """Construct a BitsAndBytesConfig from the ``model`` section of config."""
    compute_dtype_str = model_cfg.get("bnb_4bit_compute_dtype", "float16")
    compute_dtype = getattr(torch, compute_dtype_str)

    return BitsAndBytesConfig(
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
    )


def load_model_and_tokenizer(
    config_path: str = "config.yaml",
    device_map: str = "auto",
    adapter_path: Optional[str] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the 4-bit quantised base model and its tokeniser.

    Args:
        config_path: Path to ``config.yaml``.
        device_map: Device placement strategy passed to ``from_pretrained``.
            Use ``"auto"`` to let Accelerate distribute layers across available
            hardware, or ``"cpu"`` for CPU-only inference.
        adapter_path: Optional path to a saved LoRA adapter checkpoint to merge
            on top of the base model for inference.

    Returns:
        A ``(model, tokenizer)`` tuple ready for generation.
    """
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    model_name = model_cfg["name"]

    logger.info("Loading tokeniser from %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = build_bnb_config(model_cfg)

    logger.info(
        "Loading 4-bit quantised model %s (device_map=%s)", model_name, device_map
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=False,
    )
    model.config.use_cache = False

    if adapter_path is not None:
        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def prepare_model_for_training(
    model: AutoModelForCausalLM,
    config_path: str = "config.yaml",
) -> AutoModelForCausalLM:
    """Attach a LoRA adapter to the model for parameter-efficient fine-tuning.

    Args:
        model: The base model returned by :func:`load_model_and_tokenizer`.
        config_path: Path to ``config.yaml``.

    Returns:
        The model with a LoRA adapter ready for gradient-based training.
    """
    cfg = load_config(config_path)
    peft_cfg = cfg["peft"]

    lora_config = LoraConfig(
        r=peft_cfg["r"],
        lora_alpha=peft_cfg["lora_alpha"],
        lora_dropout=peft_cfg["lora_dropout"],
        target_modules=peft_cfg["target_modules"],
        bias=peft_cfg["bias"],
        task_type=peft_cfg["task_type"],
    )

    logger.info(
        "Attaching LoRA adapter (r=%d, alpha=%d)",
        lora_config.r,
        lora_config.lora_alpha,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
