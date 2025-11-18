# Miniagent

Miniagent extends the original minimind project with a veRL-style reinforcement
learning stack and VerlTool-inspired tool environments. The `agent_rl` package
allows MiniMind checkpoints to interact with code execution tools, run automated
tests, and train via Group Relative Policy Optimization (GRPO).

---

## Features

- **RL Core (`agent_rl/core/`)**
  - `policy_base.py`: abstract interface for policies.
  - `minimind_policy.py`: production-grade MiniMind wrapper with chat templating,
    streaming logprobs, and tool-call extraction.
  - `trajectory.py`: `Episode`/`Step` dataclasses + buffer helpers.
  - `rollout.py`: synchronous rollout worker that records env/tool/policy info.
  - `trainer_grpo.py`: outcome-based GRPO trainer with per-task advantage
    normalization.

- **Environments & Tools**
  - `envs/code_exec_env.py`: multi-turn code sandbox that runs dataset-provided
    tests via a tool call and rewards only after all checks pass.
  - `tools/code_exec_tool.py`: subprocess-backed Python executor with stdin &
    timeout controls.
  - `server/tool_server.py`: optional HTTP façade for exposing tools remotely.

- **Data Pipeline (`agent_rl/data/`)**
  - `CodeTaskDataset` (in `dataset/`) defines JSONL schema for code tasks.
  - `task_sampler.py` provides deterministic batch sampling + conversion to env
    payloads.

- **Training Scripts**
  - `agent_rl/scripts/train_code_exec_agent.py` wires everything together:
    dataset loading, policy initialization, rollout loop, and GRPO updates with
    live metrics.

---

## Repository Layout

```
agent_rl/
  core/        # policies, trajectories, rollout worker, GRPO trainer
  envs/        # CodeExecEnv + other task environments
  tools/       # exec/search tools and tool server
  agent/       # chat template, message schema, tool parser, agent loop
  data/        # task sampler utilities
  configs/     # YAML configs (model/env/train hyper-params)
  scripts/     # entry points, e.g., train_code_exec_agent.py
dataset/
  code_tasks/  # train/eval JSONL files following the CodeTaskDataset schema
model/
  model_minimind.py + tokenizer assets
```

---

## Quickstart

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

`dataset/code_tasks/` ships with `train.jsonl` and `eval.jsonl`. Each record:

```json
{
  "task_id": "add_two_numbers",
  "prompt": "... natural language description ...",
  "reference_solution": "All tests passed.",
  "tests": [{"input": "3 5\n", "output": "8\n"}],
  "starter_code": "... optional scaffold ...",
  "ground_truth": "... canonical answer ...",
  "metadata": {"difficulty": "easy", "topic": "arithmetic"}
}
```

Load programmatically:

```python
from dataset import load_code_task_splits
splits = load_code_task_splits("dataset/code_tasks")
sample = splits["train"][0]
env_payload = sample.to_env_task()
```

### 3. Run Training Loop

```bash
python agent_rl/scripts/train_code_exec_agent.py \
  --config agent_rl/configs/code_exec_grpo.yaml \
  --dataset-dir dataset/code_tasks \
  --model-path model \
  --device cuda:0 \
  --preview-tasks 2
```

The script:
1. Loads YAML config (model/env/train hyper-params).
2. Builds MiniMind policy + AdamW optimizer.
3. Samples tasks via `CodeTaskSampler`.
4. Runs synchronous rollouts through `CodeExecEnv`.
5. Calls `GRPOTrainer.update` and logs loss/reward/advantage each iteration.

---

## Architecture Highlights

1. **Tool-Aware Env**  
   `CodeExecEnv` automatically feeds tool executions with test fixtures, captures
   per-case results, and grants reward only when all tests pass or the sentinel
   `reference_solution` appears (legacy mode).

2. **Policy Metadata for GRPO**  
   `MiniMindPolicy.generate_action` returns decoded text, `<tool_call>` payloads,
   response token IDs, and logprobs. Rollout steps embed this metadata so the
   trainer can recompute sequence logprobs during updates.

3. **Outcome Advantage Computation**  
   `GRPOTrainer` groups episodes by `task_id`, subtracts group mean reward, and
   optionally normalizes by per-group std dev (DrGRPO style). No value network
   is required.

4. **Dataset-Driven Sampling**  
   A lightweight sampler reshuffles deterministic indices per epoch and produces
   env-ready dictionaries. The CLI optionally prints sample prompts to validate
   wiring.

---

## Configuration & CLI

- `agent_rl/configs/code_exec_grpo.yaml`
  - `model`: name/dtype, informs the policy wrapper.
  - `policy`: generation hyper-params (max tokens, temperature, top-p).
  - `env`: system prompt, max turns, reward scaling, timeouts.
  - `rollout`: number of episodes per iteration.
  - `train`: optimizer/GRPO settings (iters, LR, grad-clip, advantage norm).
  - `eval`: placeholder for future evaluation cadence.

- CLI arguments (see `python agent_rl/scripts/train_code_exec_agent.py -h`)
  - `--dataset-dir`, `--train-split`, `--eval-split`
  - `--model-path`, `--device`
  - `--preview-tasks`

---

## TODO

- [x] Hook `BasePolicy` into the existing MiniMind model/tokenizer stack so
      rollouts can produce real logits/logprobs.
- [x] Implement the full GRPO update path inside `core/trainer_grpo.py`,
      including per-task advantages and optimizer steps.
- [ ] Add larger task corpora plus richer reward sources (e.g., external graders,
      search tools) and integrate them with training/eval scripts.