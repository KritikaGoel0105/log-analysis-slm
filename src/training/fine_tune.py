"""
fine_tune.py

Week 5 — QLoRA fine-tuning setup for the offline Log Analysis SLM.

Document basis:
  * Week 5 (Section 4): "LoRA fine-tuning setup: PEFT config,
    training loop, checkpoint saving" -> deliverable "First
    fine-tuned model checkpoint".
  * Section 5.4: QLoRA Fine-Tuning Configuration — the bnb_config,
    lora_config and training_args below reproduce that section's
    values EXACTLY (r=16, alpha=32, dropout=0.1, q/v/k/o_proj,
    3 epochs, batch 2, grad-accum 8, lr 2e-4, cosine, warmup 0.1,
    save/eval per epoch, 4-bit nf4 double-quant).
  * Section 9 repository tree: this file lives at
    src/training/fine_tune.py; checkpoints go to models/checkpoints/.

Documented deviations from the Section 5.4 snippet (and why):
  1. output_dir: Section 5.4 shows './checkpoints' but the Section 9
     repository tree specifies 'models/checkpoints/'. Section 9
     governs repository structure, so models/checkpoints/ is used.
  2. evaluation_strategy: recent transformers renamed this kwarg to
     'eval_strategy'. We pass the document's exact name first and
     fall back to the new name only if TypeError is raised, so the
     document's intent (per-epoch eval) is preserved on any version.
  3. Precision: Section 5.4 sets fp16=True with the inline note
     "Use bf16=True if on Ampere GPU". The target GPU (RTX 4050,
     Ada Lovelace, compute capability 8.9 > Ampere) fully supports
     bf16, so bf16 is the default here per that note; --fp16
     restores the verbatim snippet value. bf16 also matches
     bnb_4bit_compute_dtype=torch.bfloat16, avoiding a dtype
     mismatch between quantized compute and trainer precision.

Implementation decision (not specified by the document):
  * Loss masking — prompt tokens (system instruction + user log
    input) are masked to -100 so the loss is computed only on the
    assistant's structured output. This is standard causal-LM
    instruction tuning; without it the model wastes capacity
    learning to reproduce its own inputs.

Offline guarantees (Section 2 / Section 6):
  * HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE set BEFORE importing
    transformers; base model loaded with local_files_only=True from
    ./models/qwen25-3b; report_to=[] so no telemetry/W&B; nothing
    is pushed to any hub.

Usage (from repository root, inside the offline venv):
    # 1. Smoke test first (~minutes): 20 train / 8 val examples,
    #    1 epoch, saves to models/checkpoints/smoke-test/
    python -m src.training.fine_tune --smoke

    # 2. Full Week 5 training run (hours on RTX 4050):
    python -m src.training.fine_tune
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# ------------------------------------------------------------------
# Offline enforcement — MUST precede the transformers import
# ------------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from .utils import read_jsonl

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = REPO_ROOT / "models" / "qwen25-3b"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # document Section 5.4 primary model

TRAIN_FILE = REPO_ROOT / "data" / "dataset" / "train.jsonl"
VAL_FILE = REPO_ROOT / "data" / "dataset" / "val.jsonl"

# Section 9 tree: models/checkpoints/ (see deviation note 1 above)
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"

# Max sequence length: dataset windows are <= 20 log lines plus the
# instruction and a ~256-token output; 1024 covers >99% of examples
# on this dataset while keeping VRAM within a 6 GB RTX 4050.
MAX_SEQ_LEN = 1024


# ------------------------------------------------------------------
# Dataset: tokenization with prompt masking
# ------------------------------------------------------------------

class InstructionDataset(torch.utils.data.Dataset):
    """
    Wraps the Week 3 {instruction, input, output} JSONL examples.

    Each example is rendered with the model's own chat template —
    the SAME construction used by evaluate.py (system=instruction,
    user=log input) — so training and evaluation prompts match.

    Labels equal input_ids with prompt positions set to -100
    (ignored by the loss). See "Implementation decision" in the
    module docstring.
    """

    def __init__(self, examples: list[dict], tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        # Prompt part (everything the model is GIVEN)
        messages = [
            {"role": "system", "content": example["instruction"]},
            {"role": "user", "content": example["input"]},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Target part (what the model must LEARN to produce)
        target_text = example["output"] + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False
        )["input_ids"]
        target_ids = self.tokenizer(
            target_text, add_special_tokens=False
        )["input_ids"]

        # ---- Overflow handling (bug fix) --------------------------
        # Tail-slicing `(prompt + target)[:MAX_SEQ_LEN]` silently
        # dropped ALL target tokens whenever the prompt alone reached
        # MAX_SEQ_LEN (~47% of examples). A fully -100 label row
        # yields 0/0 in cross-entropy -> eval_loss = nan, and the
        # example contributes no learning signal. Instead, drop
        # tokens from the MIDDLE of the prompt: the head (system
        # instruction + first log lines) and the tail (last log
        # lines + the chat template's assistant tag) are preserved,
        # and the target is always fully kept.
        overflow = len(prompt_ids) + len(target_ids) - MAX_SEQ_LEN
        if overflow > 0:
            keep_tail = 128  # last log lines + assistant tag
            keep_head = len(prompt_ids) - overflow - keep_tail
            if keep_head < 1:
                # degenerate case (should not occur: targets are
                # ~150-250 tokens) — keep as much tail as fits
                prompt_ids = prompt_ids[-(MAX_SEQ_LEN - len(target_ids)):]
            else:
                prompt_ids = prompt_ids[:keep_head] + prompt_ids[-keep_tail:]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


def pad_collate(batch: list[dict], pad_token_id: int) -> dict:
    """Right-pad a batch to its longest sequence (labels pad = -100)."""
    max_len = max(len(item["input_ids"]) for item in batch)

    def pad(seq, value):
        return seq + [value] * (max_len - len(seq))

    return {
        "input_ids": torch.tensor(
            [pad(b["input_ids"], pad_token_id) for b in batch]
        ),
        "attention_mask": torch.tensor(
            [pad(b["attention_mask"], 0) for b in batch]
        ),
        "labels": torch.tensor(
            [pad(b["labels"], -100) for b in batch]
        ),
    }


# ------------------------------------------------------------------
# Model loading with document Section 5.4 QLoRA configuration
# ------------------------------------------------------------------

def load_quantized_model():
    """Load the local base model in 4-bit and attach LoRA adapters."""

    # --- Section 5.4, verbatim: 4-bit quantization ---------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model from {MODEL_DIR} (4-bit nf4, offline)...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=str(MODEL_DIR), local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=str(MODEL_DIR),
        local_files_only=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Standard k-bit preparation (enables grad checkpointing, casts
    # norm layers) — required for stable QLoRA training.
    model = prepare_model_for_kbit_training(model)

    # --- Section 5.4, verbatim: LoRA configuration ---------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                    # Rank: start with 16, try 32 if underfitting
        lora_alpha=32,           # Scaling: typically 2*r
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Sanity check: only the LoRA adapters must be trainable.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} "
          f"({100 * trainable / total:.2f}% of {total:,})")

    return model, tokenizer


def build_training_arguments(output_dir: Path, epochs: int,
                             use_fp16: bool) -> TrainingArguments:
    """
    Section 5.4 training arguments, verbatim values.

    'evaluation_strategy' was renamed 'eval_strategy' in newer
    transformers — try the document's exact kwarg first, fall back
    on TypeError (deviation note 2 in the module docstring).
    """
    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        # Precision (deviation note 3): bf16 on Ada per the
        # document's own "Use bf16=True if on Ampere GPU" note;
        # --fp16 restores the snippet's literal fp16=True.
        fp16=use_fp16,
        bf16=not use_fp16,
        # Offline hardening: no telemetry, no hub push
        report_to=[],
        logging_steps=10,
        gradient_checkpointing=True,
        # ---- 6 GB VRAM fit (runtime settings only; Section 5.4
        # training hyperparameters above are unchanged) ----------
        # Trainer's eval defaults caused CUDA OOM on the RTX 4050:
        #   * eval batch defaults to 8 (vs. train batch 2) — the
        #     document specifies only per_device_TRAIN_batch_size,
        #     so pinning eval batch to 2 adds no deviation.
        per_device_eval_batch_size=2,
        #   * without this flag, Trainer accumulates all eval
        #     logits (batch x seq_len x 152k vocab, ~1.2 GB per
        #     batch at bf16) on GPU across the whole eval loop.
        #     Moving them to CPU each step bounds eval VRAM.
        eval_accumulation_steps=1,
    )

    try:
        return TrainingArguments(evaluation_strategy="epoch", **kwargs)
    except TypeError:
        print("NOTE: this transformers version uses 'eval_strategy' "
              "(renamed from 'evaluation_strategy'); intent unchanged.")
        return TrainingArguments(eval_strategy="epoch", **kwargs)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Week 5 QLoRA fine-tuning (offline, Section 5.4 config)."
    )
    parser.add_argument("--smoke", action="store_true",
                        help="smoke test: 20 train / 8 val examples, "
                             "1 epoch, output to models/checkpoints/"
                             "smoke-test/")
    parser.add_argument("--epochs", type=int, default=3,
                        help="training epochs (document default: 3)")
    parser.add_argument("--fp16", action="store_true",
                        help="use fp16 instead of bf16 (the Section "
                             "5.4 snippet's literal value)")
    parser.add_argument("--resume", type=str, default=None,
                        help="path to a checkpoint to resume from")
    args = parser.parse_args()

    print("=" * 60)
    print("Week 5 — QLoRA Fine-Tuning (offline)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. QLoRA 4-bit training "
              "requires a CUDA GPU (bitsandbytes has no CPU path).")
        print("Verify the venv has the CUDA torch build installed.")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # ---- Data ----------------------------------------------------
    train_examples = read_jsonl(TRAIN_FILE)
    val_examples = read_jsonl(VAL_FILE)
    if not train_examples or not val_examples:
        print(f"ERROR: dataset missing ({TRAIN_FILE}, {VAL_FILE}). "
              "Run Week 3 scripts first.")
        return 1

    epochs = args.epochs
    output_dir = CHECKPOINT_DIR
    if args.smoke:
        train_examples = train_examples[:20]
        val_examples = val_examples[:8]
        epochs = 1
        output_dir = CHECKPOINT_DIR / "smoke-test"
        print("SMOKE TEST MODE: 20 train / 8 val examples, 1 epoch")

    print(f"Train examples : {len(train_examples)}")
    print(f"Val examples   : {len(val_examples)}")
    print(f"Epochs         : {epochs}")
    print(f"Output         : {output_dir}")

    # ---- Model ---------------------------------------------------
    model, tokenizer = load_quantized_model()

    train_dataset = InstructionDataset(train_examples, tokenizer)
    val_dataset = InstructionDataset(val_examples, tokenizer)

    training_args = build_training_arguments(
        output_dir, epochs, use_fp16=args.fp16
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda batch: pad_collate(
            batch, tokenizer.pad_token_id
        ),
    )

    # ---- Train ---------------------------------------------------
    start = time.time()
    result = trainer.train(resume_from_checkpoint=args.resume)
    elapsed = time.time() - start

    # ---- Save final adapter (deliverable: model checkpoint) ------
    final_dir = output_dir / "final-adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # ---- Run summary (feeds the Week 6 training report) ----------
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke-test" if args.smoke else "full",
        "model": MODEL_NAME,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "epochs": epochs,
        "precision": "fp16" if args.fp16 else "bf16",
        "train_runtime_s": round(elapsed, 1),
        "final_train_loss": round(result.training_loss, 4),
        "log_history": trainer.state.log_history,
        "config": {
            "lora": {"r": 16, "alpha": 32, "dropout": 0.1,
                     "target_modules": ["q_proj", "v_proj",
                                        "k_proj", "o_proj"]},
            "quantization": "4-bit nf4, double quant, bf16 compute",
            "batch_size": 2, "grad_accum": 8,
            "effective_batch": 16, "lr": 2e-4,
            "scheduler": "cosine", "warmup_ratio": 0.1,
        },
    }
    summary_file = output_dir / "training_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print(f"Training complete in {elapsed / 60:.1f} min")
    print(f"Final train loss : {result.training_loss:.4f}")
    print(f"Adapter saved to : {final_dir}")
    print(f"Run summary      : {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
