"""
prepare_data.py (Regex-Based Version, No NLTK)
----------------------------------------------
Downloads the HuggingFace 'scientific_papers' (arxiv) split,
splits abstracts into problem-approach pairs using improved regex sentence splitting,
logs skip reasons, and writes train/val/test JSONL files.
"""

from datasets import load_dataset
from pathlib import Path
import json, random, tqdm, re

SAVE_DIR = Path("data")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

''' 
Tried adjusting MAX_PROBLEM_TOKENS and MIN_APPROACH_TOKENS many times, and this setup gave the best balance. 
Stricter values filtered out all data, leaving nothing to train on. Too lenient, and the resulting data lacked quality. 
This was the most satisfying middle ground.
'''
MAX_PROBLEM_TOKENS = 50
MIN_APPROACH_TOKENS = 90
VAL_RATIO = 0.05
TEST_RATIO = 0.05
random.seed(42)

# Tried to use NLTK, but couldn't make it to work
def custom_sent_tokenize(text):
    # Improved regex for academic-style sentence splitting
    pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+(?=[A-Z0-9“\(])')
    return pattern.split(text.strip())

def split_abstract(abstract: str):
    sents = custom_sent_tokenize(abstract)
    if not sents or len(sents) < 2:
        sents = [s.strip() for s in abstract.split("\n") if s.strip()]
    if not sents or len(sents) < 2:
        return None, None, "too_few_sentences"

    problem = sents[0].strip()
    approach = " ".join(sents[1:]).strip()

    if len(problem.split()) > MAX_PROBLEM_TOKENS:
        return None, None, "problem_too_long"

    if len(approach.split()) < MIN_APPROACH_TOKENS:
        return None, None, "approach_too_short"

    return problem, approach, None

def main():
    print("Downloading / loading scientific_papers:arxiv …")
    ds = load_dataset("scientific_papers", "arxiv", split="train", trust_remote_code=True)

    records = []
    skip_reasons = {"too_few_sentences": 0, "problem_too_long": 0, "approach_too_short": 0, "other": 0}

    for i, row in enumerate(tqdm.tqdm(ds, desc="Filtering")):
        abstract = row["abstract"]
        prob, app, reason = split_abstract(abstract)

        if prob and app:
            records.append({"problem": prob, "approach": app})
        else:
            skip_reasons[reason or "other"] += 1
            if i < 5:  # Log first few skipped cases
                print(f"\n--- Skipped Abstract {i} ---")
                print(f"Reason: {reason}")
                print(abstract[:500])

    print("\nSkipping Summary:")
    for reason, count in skip_reasons.items():
        print(f"{reason}: {count:,}")

    random.shuffle(records)
    n = len(records)
    test_size = int(n * TEST_RATIO)
    val_size = int(n * VAL_RATIO)

    splits = {
        "train": records[: n - val_size - test_size],
        "validation": records[n - val_size - test_size : n - test_size],
        "test": records[n - test_size :],
    }

    for split, rows in splits.items():
        path = SAVE_DIR / f"{split}.jsonl"
        with path.open("w", encoding="utf8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows):,} lines ➜ {path}")

    print("\nDataset ready in", SAVE_DIR.resolve())

if __name__ == "__main__":
    main()
