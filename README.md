# Fine-Tuned-LLM-for-Research
Generate full research **approaches** from short **problem statements**  
→ tiny model, Windows-friendly, 100 % open-source.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-lightgrey)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

---

## ✨ Project overview
| Stage | Tooling | Notes |
|-------|---------|-------|
| **Dataset** | HF `scientific_papers:arxiv` | Abstract → ✂︎ 1st sentence = *problem*, rest = *approach*. |
| **Model** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1 B params, runs on CPU-only if needed. |
| **Fine-tuning** | LoRA (`peft`) | Rank 8 adapters, AdamW, LR 1e-4. |
| **Evaluation** | ROUGE-L, BLEU, BERTScore | Optional GPTScore for LLM-based eval. |
| **Inference** | CLI chat (`infer.py`) | Type any research question, get an approach. |

---

## 🖥️ Quick start (Windows + VS Code)

```powershell
# 0. clone repo
git clone https://github.com/<your-handle>/fine-tuned-llm-for-research.git
cd fine-tuned-llm-for-research

# 1. create & activate venv
python -m venv llm-research-env
.\llm-research-env\Scripts\activate

# 2. install deps
pip install -r requirements.txt

# 3. build dataset  (~10 min / 1.4 GB)
python prepare_data.py

## 🖥️ Train (LoRA)

```powershell
# full data, 1 epoch (≈ 8 h CPU, 90 min RTX 3060)
python finetune.py --data_dir data --output_dir runs/lora-r8 --num_train_epochs 1 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --learning_rate 5e-5
```

---

## 🧪 Experiment tips

- **Baseline (no training):** skip `finetune.py` and run `run_evaluation.py` directly on a raw model.
- **Hyper-param sweep:** adjust `--lora_r` (e.g., 8 → 16) or `--learning_rate` (e.g., 5e-5 → 1e-4).
- **RAM-bound?** Add `--train_subset 0.2` (use only 20% of training data).

---

## 📏 Evaluate

```powershell
python run_evaluation.py --checkpoint runs/lora-r8/merged --data_dir data --max_new_tokens 256
```

Sample result (on 100 test examples):

| Metric        | Score |
|---------------|-------|
| BLEU          | 3.5   |
| ROUGE-L       | 0.17  |
| BERTScore (F1) | 0.82  |

*(See `reports/metrics*.json` for full runs and baselines.)*

---

## 💬 Interactive demo

```powershell
python infer.py --checkpoint runs/lora-r8/merged
```

Example:

```
Research problem: How can renewable energy sources be efficiently integrated into existing power grids?
--- Suggested approach ---
1. Deploy distributed energy resources …
2. Introduce smart-grid forecasting …
⋯
```

---

## 📂 Repository structure

```typescript
fine-tuned-llm-for-research/
│
├─ data/                ← JSONL splits after `prepare_data.py`
├─ runs/                ← checkpoints & logs
│   └─ lora-r8/
│       ├─ adapter_config.json
│       ├─ merged/      ← single-folder model for inference
│       └─ trainer_state.json
│
├─ prepare_data.py      ← build problem/approach pairs
├─ finetune.py          ← LoRA training script
├─ run_evaluation.py    ← ROUGE/BLEU/BERTScore (opt GPTScore)
├─ infer.py             ← chat interface
└─ requirements.txt
```

