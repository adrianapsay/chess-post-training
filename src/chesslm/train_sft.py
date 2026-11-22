"""Supervised fine-tuning (SFT) with LoRA, driven by configs/sft.yaml.

The model sees a position (the same prompt used in evaluation) and learns to answer
with the move that was played. The loss is computed on the answer only, not the prompt.
"""

import time
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from chesslm.model import load_model, load_tokenizer, pick_device
from chesslm.prompts import build_messages


# Which column holds the answer the model learns (configs/sft.yaml, data.answer):
# "played" = the move the player made (sft_v1); "stockfish" = Stockfish's best move,
# added by data.add_stockfish_moves (sft_stockfish).
ANSWER_COLUMNS = {"played": "move", "stockfish": "best"}


def to_sft_example(row, prompt_cfg, answer="played"):
    """One training example: the question as the prompt, the answer move as the completion."""
    move = row[f"{ANSWER_COLUMNS[answer]}_{prompt_cfg['notation']}"]
    return {
        "prompt": build_messages(row, prompt_cfg),
        "completion": [{"role": "assistant", "content": move}],
    }


def sample_examples(positions, n, seed):
    """A seeded random sample of n positions. The same seed always gives the same positions, in the same order."""
    return positions.sample(n=min(n, len(positions)), random_state=seed)


def to_dataset(sample, prompt_cfg, answer="played"):
    """Turn already-sampled positions into SFT examples, keeping their order."""
    return Dataset.from_list([to_sft_example(row, prompt_cfg, answer) for row in sample.to_dict("records")])


def make_dataset(positions, n, prompt_cfg, seed, answer="played"):
    """A seeded random sample of n positions, as SFT examples."""
    return to_dataset(sample_examples(positions, n, seed), prompt_cfg, answer)


def examples_label(examples):
    """A short name for a number of training examples: 20000 -> "20k" (small numbers, as in a quick check, stay as they are)."""
    return f"{examples // 1000}k" if examples >= 1000 else str(examples)


def copy_dir(output_dir, examples):
    """Where the copy saved after `examples` training examples goes: checkpoints/sft_100k -> checkpoints/sft_100k_at_20k."""
    return output_dir.parent / f"{output_dir.name}_at_{examples_label(examples)}"


class SaveCopies(TrainerCallback):
    """Save a copy of the adapter after given numbers of training examples (for a data-scaling curve).

    One training step uses effective_batch_size examples, so the copy for N examples is saved
    after step N // effective_batch_size.
    """

    def __init__(self, output_dir, save_at, examples_per_step):
        self.dirs = {n // examples_per_step: copy_dir(output_dir, n) for n in save_at}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step in self.dirs:
            model.save_pretrained(str(self.dirs[state.global_step]))


def train(train_ds, val_ds, sft_cfg, output_dir, save_at=None):
    """Train a LoRA adapter, save it to output_dir, and return (loss log, compute used).

    save_at: optional numbers of training examples (e.g. [20000, 40000]) after which a copy of the
    adapter is also saved, to copy_dir(output_dir, n). Used for the data-scaling run (notebook 12).
    """
    t = sft_cfg["training"]
    device = pick_device()
    use_4bit = sft_cfg["lora"]["quantize_4bit"] and device == "cuda"
    output_dir = Path(output_dir)
    callbacks = [SaveCopies(output_dir, save_at, t["effective_batch_size"])] if save_at else None

    trainer = SFTTrainer(
        model=load_model(sft_cfg["model"], quantize_4bit=use_4bit),
        processing_class=load_tokenizer(sft_cfg["model"]),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=LoraConfig(
            r=sft_cfg["lora"]["rank"],
            lora_alpha=sft_cfg["lora"]["alpha"],
            lora_dropout=sft_cfg["lora"]["dropout"],
            target_modules=sft_cfg["lora"]["target_modules"],
            task_type="CAUSAL_LM",
        ),
        args=SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=t["epochs"],
            learning_rate=t["learning_rate"],
            lr_scheduler_type=t["lr_scheduler"],
            per_device_train_batch_size=t["batch_size"],
            per_device_eval_batch_size=t["batch_size"],
            gradient_accumulation_steps=t["effective_batch_size"] // t["batch_size"],
            gradient_checkpointing=t["gradient_checkpointing"],
            max_length=t["max_seq_length"],
            completion_only_loss=True,  # learn the answer, not the prompt
            bf16=device == "cuda",
            logging_steps=t["logging_steps"],
            eval_strategy="steps",
            eval_steps=t["eval_steps"],
            save_strategy="no",  # we save the final adapter ourselves
            report_to="none",
            seed=t["seed"],
        ),
        callbacks=callbacks,
    )

    start = time.time()
    trainer.train()
    minutes = (time.time() - start) / 60
    trainer.save_model(str(output_dir))  # saves the LoRA adapter only (a few MB), not the full model

    compute = {
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "quantized_4bit": use_4bit,
        "wall_clock_minutes": round(minutes, 1),
        "gpu_hours": round(minutes / 60, 3),
        "examples_seen": len(train_ds) * t["epochs"],
        "tokens_seen": sum(len(ids) for ids in trainer.train_dataset["input_ids"]) * t["epochs"],
    }
    return pd.DataFrame(trainer.state.log_history), compute
