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
