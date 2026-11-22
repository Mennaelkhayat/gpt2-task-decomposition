# GPT-2 Task Decomposition

Graduation project: fine-tuning GPT-2 to take a high-level task (e.g. "study english",
"plan a birthday party") and generate a breakdown into actionable subtasks.

## What it does

Given a short task description, the model generates an ordered list of subtasks that
accomplish it. The model is a `gpt2` checkpoint fine-tuned as a causal language model on a
custom dataset of tasks paired with their reference decompositions.

Example:

```
Task: "study english"
Generated Subtasks: " [SEP] learn 20 new vocabulary words [SEP] practice grammar exercises [SEP] ..."
```

## Dataset

`data/tasks.json` contains 189 high-level tasks with their reference subtasks
(1,188 subtasks in total), collected and annotated for this project. Each training example
is serialized as a single sequence:

```
<task> [SEP] <subtask 1> [SEP] <subtask 2> [SEP] ...
```

## Architecture and training

- Base model: `gpt2` (117M parameters) via Hugging Face Transformers
- Objective: causal language modeling over the `task [SEP] subtasks` sequences
- Optimizer: AdamW, learning rate 5e-5, cosine schedule with warmup
- Batch size 4, 3 epochs, gradient clipping at 1.0
- Generation: sampling with `top_k=30`, `top_p=0.7`, `temperature=0.9`

## Project structure

```
├── data/
│   └── tasks.json                    # 189 tasks with reference subtasks
├── notebooks/
│   └── task_decomposition.ipynb      # the original experiment, cleaned up
├── src/
│   ├── dataset.py                    # TaskDataset: task+subtasks -> tokenized sequences
│   ├── train.py                      # fine-tuning loop (saves a Hugging Face checkpoint)
│   └── generate.py                   # subtask generation from a fine-tuned checkpoint
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# Fine-tune (GPU recommended; saves to checkpoints/)
python -m src.train --epochs 3 --batch-size 4 --output-dir checkpoints/

# Generate subtasks for a new task
python -m src.generate --task "study english" --model-dir checkpoints/
```

Or walk through `notebooks/task_decomposition.ipynb`, which contains the same pipeline in
notebook form.

## Results and notes

- With only 189 training tasks this is a proof of concept: the fine-tuned model learns the
  `task -> subtask list` format and produces plausible decompositions for tasks similar to
  the training data, but generalization to arbitrary domains is limited by dataset size.
- The original notebook set `max_length=50000` in `generate()`; GPT-2's context window is
  1,024 tokens, so generation here caps at a sensible default (100 tokens, configurable).
- Model artifacts were originally passed around with `pickle`; training now saves a standard
  Hugging Face checkpoint (`save_pretrained`), which `generate.py` loads directly.
