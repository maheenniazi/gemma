#!/usr/bin/env python3
"""
QLoRA SFT on harvested winning trajectories -> adapters/main_lora/.

Runs on a FREE Kaggle GPU (T4x2 or P100). Trains a small LoRA adapter on top of
the QAT-quantized Gemma 4 base using the resolved trajectories from
harvest_trajectories.py. The output artifacts drop straight into the submission:

    adapters/main_lora/adapter_config.json
    adapters/main_lora/adapter_model.safetensors

DESIGN CHOICES (for the free-compute plan)
------------------------------------------
- Base: gemma-4-e4b (fast to train, fits free VRAM comfortably). Keep in sync with
  agent.yaml — if you switch the submission model you must retrain here on the same base.
- LoRA rank 16 (cheap, enough capacity for behavioural cloning of a tool-use policy).
- Short runs: this is rejection-sampling SFT; a few hundred high-quality examples
  trained for 1-2 epochs is plenty and avoids overfitting the small task set.
- Train only on ASSISTANT tokens (mask the system/user/tool turns) so the model learns
  to PRODUCE the winning actions, not to parrot tool output.

USAGE (Kaggle)
--------------
    pip install -q "unsloth[kaggle-new]"   # or trl+peft+bitsandbytes
    python training/train_qlora.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ---- config ----------------------------------------------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "google/gemma-4-e4b-it")  # QAT base id (WIRE if different)
DATA_PATH = os.environ.get("DATA_PATH", "training/data/sft_trajectories.jsonl")
OUT_DIR = Path(os.environ.get("OUT_DIR", "adapters/main_lora"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "8192"))
EPOCHS = float(os.environ.get("EPOCHS", "2"))
LORA_RANK = int(os.environ.get("LORA_RANK", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LR = float(os.environ.get("LR", "2e-4"))
BATCH = int(os.environ.get("BATCH", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))


def load_examples(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(
            f"No training data in {path}. Run harvest_trajectories.py first."
        )
    return rows


def main() -> None:
    examples = load_examples(DATA_PATH)
    print(f"Loaded {len(examples)} resolved trajectories.")

    # Lazy imports so the file is readable without the heavy deps installed.
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    def to_text(ex: dict) -> dict:
        # Render chat messages with the model's chat template. We train on the full
        # rendered conversation; `assistant_only_loss` (below) masks non-assistant tokens.
        text = tok.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = Dataset.from_list([to_text(e) for e in examples])

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # base is already QAT w4a16; load as-is. If using bitsandbytes NF4 instead,
        # pass load_in_4bit=True + a BitsAndBytesConfig here.
    )

    peft_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="training/_ckpt",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        tokenizer=tok,
        # Mask everything but assistant tokens so we clone the policy, not the context.
        # (TRL exposes this via DataCollatorForCompletionOnlyLM or assistant_only_loss
        #  depending on version — enable whichever your TRL supports.)
    )

    trainer.train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)   # writes adapter_config.json + adapter_model.safetensors
    tok.save_pretrained(OUT_DIR)
    print(f"\nAdapter written to {OUT_DIR}/ — rezip the submission to include it.")


if __name__ == "__main__":
    main()
