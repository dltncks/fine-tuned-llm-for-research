import argparse, os, torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--model_name",
    default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--train_subset", type=float, default=1.0,
    help="Fraction of the *train* split to use (0-1).")
    p.add_argument("--max_steps", type=int, default=None)
    return p.parse_args()

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def format_sample(ex):
    return f"<problem>\n{ex['problem']}\n</problem>\n<approach>\n{ex['approach']}\n</approach>"

def build_label_mask(input_ids, tokenizer):
    labels = input_ids.clone()
    aid = tokenizer.convert_tokens_to_ids("<approach>")
    idx = (input_ids == aid).nonzero(as_tuple=True)
    if len(idx[1]):                           # found the tag
        labels[0, : idx[1][0].item() + 1] = -100
    return labels

# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------- dataset -------------------------------------------------
    raw = load_dataset(
        "json",
        data_files={
            "train": str(Path(args.data_dir) / "train.jsonl"),
            "validation": str(Path(args.data_dir) / "validation.jsonl"),
        },
    )

    if args.train_subset < 1.0:
        n = int(len(raw["train"]) * args.train_subset)
        raw["train"] = raw["train"].shuffle(seed=42).select(range(n))

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tok.add_special_tokens(
        {"additional_special_tokens":
            ["<problem>", "</problem>", "<approach>", "</approach>"]}
    )
    tok.pad_token = tok.eos_token

    def preprocess(ex):
        enc = tok(format_sample(ex),
        max_length=args.max_seq_length,
        truncation=True,
        padding="max_length")
        enc["labels"] = build_label_mask(
            torch.tensor([enc["input_ids"]]), tok
        )[0].tolist()
        return enc

    data = raw.map(preprocess, remove_columns=["problem", "approach"])

    # ---------------- model ---------------------------------------------------
    dtype = torch.float32                     ### FIX ② – always FP32
    device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tok))

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    # ---------------- training ------------------------------------------------
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        fp16=False, bf16=False,
        optim="adamw_torch",
        report_to="none",
        max_grad_norm=1.0,                    ### FIX ④ – gradient clipping
        remove_unused_columns=False,
        max_steps=args.max_steps,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=default_data_collator,  ### FIX ①
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    # ------------- merge LoRA for single-folder inference ---------------------
    merged = Path(args.output_dir) / "merged"
    model.merge_and_unload().save_pretrained(merged)
    print("✓ done →", merged.resolve())

if __name__ == "__main__":
    main()