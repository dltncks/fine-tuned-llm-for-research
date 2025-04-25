"""
run_evaluation.py
-----------------
Computes BLEU, ROUGE-L and BERTScore on the test split.
"""

import argparse, torch, tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path TO THE *MERGED* MODEL FOLDER")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--max_new_tokens", type=int, default=256)
    return p.parse_args()

# --------------------------------------------------------------------------- #
def generate(model, tok, prompt, max_new_tokens):
    input_ids = tok(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(outputs[0], skip_special_tokens=True)

# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)

    # ---------- load tokenizer from PARENT DIR ------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path.parent,          # <-- changed line
        trust_remote_code=True)

    # ---------- load merged LoRA model ------------------------------------- #
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        device_map="auto",   # use GPU if available
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,

        trust_remote_code=True,
    )
    model.eval()

    ds = load_dataset("json", data_files={"test": f"{args.data_dir}/test.jsonl"})["test"]
    ds = ds.select(range(100))  # <- just use 100 test samples

    preds, refs = [], []
    for row in tqdm.tqdm(ds, desc="Generating"):
        prompt  = f"<problem>\n{row['problem']}\n</problem>\n<approach>\n"
        out     = generate(model, tokenizer, prompt, args.max_new_tokens)
        approach = out.split("</approach>")[-1] if "</approach>" in out else out
        preds.append(approach.strip())
        refs.append(row["approach"].strip())

    # ---------- metrics ----------------------------------------------------- #
    bleu      = evaluate.load("sacrebleu")
    rouge     = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    bleu_res  = bleu.compute(predictions=preds, references=[[r] for r in refs])
    rouge_res = rouge.compute(predictions=preds, references=refs)
    bert_res  = bertscore.compute(predictions=preds, references=refs, lang="en")

    print("\n=== Automatic Evaluation Metrics ===")
    print(f"BLEU:       {bleu_res['score']:.2f}")
    print(f"ROUGE-L:    {rouge_res['rougeL']:.4f}")
    print(f"BERTScore (F1): {sum(bert_res['f1']) / len(bert_res['f1']):.4f}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
