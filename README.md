# Latent Skill Selection

This repository contains code for the paper **"[Refer, Infer, or Abstain: Investigating Latent Skill Selection of Large Language Models under Contextual Question Answering Uncertainty](https://aclanthology.org/2026.uncertainlp-main.39/)"**.

## Overview

When answering a contextual question, a language model can **_Refer_** to the context, **_Infer_** an answer from it, or **_Abstain_** when the answer is not available. This repository contains the code to study how models implicitly select between these skills under uncertainty. It covers _Prompting_ and _Probing_ experiments.

## Quick Start

### 1. Setup
Create a virtual environment

```bash
module load cuda/11.7.0
module load python/3.9.9
python3 -m venv venv
source venv/bin/activate
```
---

Install dependencies from the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```


### 2. Prepare the Dataset

We use [QuAIL](https://huggingface.co/datasets/textmachinelab/quail). The following script maps question types to skill types and saves the result to the dataset folder:

```bash
python dataset.py
```

### 3. Models

We evaluate 9 language models from 3 model families available in 🤗 Hugging Face: 

- Llama 3: `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`
- Qwen 2.5: `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`
- Gemma 3: `google/gemma-3-1b-it`, `google/gemma-3-4b-it`, `google/gemma-3-12b-it`
  
## Running Experiments

### Prompting

From the `prompting/` folder, run:

```bash
python qa.py \
  --dataset_path ..dataset/challenge.json \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --results_folder challenge \
  --apply_chat_template
```

| Argument                 | Description                                                    |
|--------------------------|----------------------------------------------------------------|
| `--dataset_path`         | Path to the local JSON dataset created in step 2               |
| `--model_name`           | Hugging Face model name                                        |
| `--results_folder`       | Folder where results are stored, use the dataset split name    |
| `--apply_chat_template`  | Apply the model's chat template when prompting (optional flag) |

### Probing

Both steps are run from the `probing/` folder.

**Step 1: Extract features** from each layer of the model:

```bash
python extract_features.py --model_name meta-llama/Llama-3.2-1B-Instruct
```

**Step 2: Train probes**, one per layer:

```bash
python train_probes.py --model_name Llama-3.2-1B-Instruct
```

## Citation

If you use or extend this work, please cite:

```bibtex
@inproceedings{kalbaliyev-sirts-2026-refer,
    title = "Refer, Infer, or Abstain: Investigating Latent Skill Selection of Large Language Models under Contextual Question Answering Uncertainty",
    author = "Kalbaliyev, Emil  and
      Sirts, Kairit",
    booktitle = "The Proceedings of the Third Workshop on Uncertainty-Aware {NLP}",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.uncertainlp-main.39/",
    pages = "657--671",
    ISBN = "979-8-89176-486-6"
}
```
