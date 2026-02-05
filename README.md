# MetaGen-AI

MetaGen-AI is a lightweight, training-free framework that adapts role specifications and graph-structured LLM collaborations at inference time, enabling dynamic multi-role workflows for complex reasoning and code tasks without updating base model weights.

It provides:
- A configurable **graph builder** (task hub → role nodes → exit nodes)
- Built-in **role libraries** and baseline graph templates
- Dataset preparation scripts that normalize sources into a simple **JSONL** format
- CLI scripts for running **baselines** and **graph-based evaluation**, with metrics exported to CSV

## Installation

```bash
git clone <your-fork-or-repo-url>
cd metagen-ai
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## API setup

MetaGen-AI uses an OpenAI-compatible chat completion endpoint.

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

You can change the model and endpoint in `configs/default.yaml`:

- `llm.model`: model name
- `llm.base_url`: optional (OpenAI-compatible gateway)

## Data format

All datasets are loaded from JSONL under:

```
data/datasets/<name>.jsonl
```

Each line is one example. Common schemas:

- Multiple-choice QA: `{"question": "...", "choices": [...], "answer": "..."}`
- NLI: `{"premise": "...", "hypothesis": "...", "label": "..."}`
- Extractive QA: `{"passage": "...", "question": "...", "answers": [...]}`

## Prepare datasets

Use the preparation scripts to download and convert datasets into `data/datasets/*.jsonl`.

Examples:

```bash
python scripts/prepare_mnli.py --help
python scripts/prepare_mmlu.py --help
python scripts/prepare_drop.py --help
python scripts/prepare_humaneval.py --help
```

## Run graph-based evaluation

Run one or more datasets (comma-separated) and save metrics to CSV:

```bash
python scripts/run_dataset_eval.py \
  --config configs/default.yaml \
  --datasets dataset_name \
  --max_examples -1 \
  --out_csv logs/metrics/run_dataset_name.csv
```

## Run baselines

Baselines generate predictions for a single dataset and write a CSV with per-example results and aggregate metrics:

```bash
python scripts/run_baselines.py \
  --config configs/default.yaml \
  --dataset mnli_dev_matched \
  --baseline cot \
  --out_csv logs/metrics/baseline_cot_mnli.csv
```


## Repository layout

- `src/metagen_ai/` — core library
- `configs/` — runtime configuration (LLM, controller, evaluation)
- `scripts/` — dataset preparation and evaluation entrypoints
