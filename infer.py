"""
infer.py
--------
Interactive console: type a research problem, get a generated approach.
"""

import argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max_new_tokens", default=256, type=int)
    return p.parse_args()

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(Path(args.checkpoint).parent, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto", trust_remote_code=True
    )
    model.eval()

    print("\n=== Research-Approach Generator ===\nType 'exit' to quit.\n")
    while True:
        problem = input("Research problem: ").strip()
        if problem.lower() in {"exit", "quit"}:
            break
        prompt = f"<problem>\n{problem}\n</problem>\n<approach>\n"
        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.5,
        )
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        approach = result.split("</approach>")[-1] if "</approach>" in result else result
        print("\n--- Suggested approach ---\n")
        print(approach.strip(), "\n")

if __name__ == "__main__":
    main()
