# MiniMind Datasets

将所有下载的数据集文件放置到当前目录.

Place the downloaded dataset file in the current directory.

## Code-task dataset for RL

We provide a starter dataset for tool-augmented RL in `code_tasks/`:

```
dataset/
  └── code_tasks/
      ├── train.jsonl
      └── eval.jsonl
```

Each line in the JSONL files has the following schema:

```json
{
  "task_id": "add_two_numbers",
  "prompt": "Problem statement shown to the agent.",
  "reference_solution": "All tests passed.",
  "tests": [
    {"input": "3 5\n", "output": "8\n", "description": "Two positives"}
  ],
  "starter_code": "... optional helper scaffold ...",
  "ground_truth": "... canonical answer for offline validation ...",
  "metadata": {"difficulty": "easy", "topic": "arithmetic"}
}
```

- `prompt`: natural-language instructions delivered to the agent/environment.
- `reference_solution`: sentinel string that the current `CodeExecEnv` uses to
  detect success (the prompts ask the agent to print this string after running
  tests). Future reward functions can instead evaluate `tests` directly.
- `tests`: optional IO-based checks; each entry provides raw `input`, expected
  `output`, and an optional description/timeout.
- `starter_code`/`ground_truth`: optional scaffolding for IDE-like experiences or
  offline regression tests.
- `metadata`: arbitrary tags (difficulty, domain, source, etc.).

To load the dataset programmatically:

```python
from dataset import CodeTaskDataset, load_code_task_splits

splits = load_code_task_splits("dataset/code_tasks", splits=("train", "eval"))
train_ds = splits["train"]
sample = train_ds.sample(1)[0]
env_payload = sample.to_env_task()
```

Custom datasets can follow the same JSONL format; drop the files into
`dataset/code_tasks/` (or another directory) and point
`load_code_task_splits(..., data_dir=...)` to the new path.