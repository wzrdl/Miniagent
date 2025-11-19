# Miniagent RL & Tool-Usage Training Framework

Miniagent extends the original MiniMind project with a veRL-style reinforcement-learning
stack that can train tool-using agents end to end. The `agent_rl` package wires policies,
tool-capable environments, and GRPO training so that LLM checkpoints can safely call tools
(code execution, live web search, etc.) and learn from sparse rewards.

The README below walks you through setup, datasets, configuration, training flows,
observability, and extension hooks so you can operate the framework without reading
source code first.

---

## Table of Contents

1. [Core Capabilities](#core-capabilities)
2. [Repository Layout](#repository-layout)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Datasets](#datasets)
6. [Available Tools & Environments](#available-tools--environments)
7. [Configuration Files](#configuration-files)
8. [Running Training Jobs](#running-training-jobs)
9. [Monitoring & Debugging](#monitoring--debugging)
10. [Extending the Framework](#extending-the-framework)
11. [Troubleshooting FAQ](#troubleshooting-faq)
12. [Contributing](#contributing)

---

## Core Capabilities

- **GRPO Training Loop** – Implements outcome-based Group Relative Policy Optimization
  with per-task normalization (`agent_rl/core/trainer_grpo.py`).
- **Policy Backends** – MiniMind and HuggingFace-compatible chat policies with tool-call
  parsing (`agent_rl/core/minimind_policy.py`, `agent_rl/core/hf_policy.py`).
- **Tool-Aware Environments** – Multi-turn environments that enforce tool usage semantics,
  e.g., `code_exec_env` for Python execution and `search_env` for real web search.
- **Async + Sync Rollouts** – Thread-based async executor for high-throughput data
  collection and a simple synchronous worker for debugging.
- **Tool Registry** – Register any number of tools (code execution, Tavily/Serper/Bing
  search, custom HTTP proxies) and expose them to policies.

---

## Repository Layout

```
agent_rl/
  agent/      # Chat message schema + tool-call parser
  core/       # Policies, rollouts, trajectories, GRPO trainer
  data/       # Deterministic task sampler utilities
  envs/       # CodeExecEnv, SearchEnv, base env abstractions
  tools/      # Tool registry + code exec + web search tools
  configs/    # YAML configs for code/search agents
  scripts/    # CLI entry points (train_code_exec_agent.py, train_search_agent.py)
dataset/
  code_tasks/    # JSONL splits for code execution tasks
  search_tasks/  # JSONL splits for web-search QA tasks
model/
  model_minimind.py + tokenizer assets (MiniMind checkpoints)
```

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Tested on 3.10/3.11. |
| CUDA Toolkit 12.x (optional) | Needed when training on GPU. |
| Web search API key | Required for real-time search (`TAVILY_API_KEY`, `SERPER_API_KEY`, etc.). |
| Git + virtualenv | Recommended for environment isolation. |

---

## Installation

```bash
git clone https://github.com/your-org/miniagent.git
cd miniagent
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: `requests` is included for the HTTP-backed `WebSearchTool`. If you deploy in
> a restricted environment, ensure outbound HTTPS is allowed or swap the provider to a
> company-internal proxy.

---

## Datasets

### Code Execution Tasks

- Location: `dataset/code_tasks/train.jsonl` & `eval.jsonl`
- Fields: `task_id`, `prompt`, `starter_code`, `tests`, `reference_solution`, `metadata`
- Use `CodeTaskDataset` / `load_code_task_splits` helper to load them into memory.
- Rewards: success is granted when the `code_exec` tool reports that all tests pass.

### Search QA Tasks

- Location: `dataset/search_tasks/*.jsonl`
- Required fields: `task_id`, `question`, `answer`
- Optional fields: `documents` (seed passages), `metadata` (keywords, hints)
- At training time, the environment only injects the question; the agent must call the
  `search` tool to fetch live snippets and then produce the final answer.

### Validating Dataset Wiring

```python
from dataset import SearchTaskDataset
ds = SearchTaskDataset.from_path("dataset/search_tasks/train.jsonl")
sample = ds[0].to_env_task()
print(sample["question"], "->", sample["answer"])
```

---

## Available Tools & Environments

| Tool | File | Description |
|------|------|-------------|
| `code_exec` | `agent_rl/tools/code_exec_tool.py` | Executes user code in a sandbox, captures stdout/stderr, enforces timeouts. |
| `search` (web) | `agent_rl/tools/web_search_tool.py` | Real HTTP search client (Tavily/Serper/Bing/custom). Handles retries, rate limits, snippet formatting. |
| `search` (corpus) | `agent_rl/tools/search_tool.py` | Offline in-memory search used for regression tests or when no API key is available. |

| Environment | File | Reward Signal |
|-------------|------|---------------|
| `CodeExecEnv` | `agent_rl/envs/code_exec_env.py` | Reward only when code tool returns all tests passing (`success_reward`) or penalties on failure/timeouts. |
| `SearchEnv` | `agent_rl/envs/search_env.py` | Tracks multi-turn interactions with the `search` tool; rewards if the final answer matches the ground truth. |

---

## Configuration Files

### Code Execution (`agent_rl/configs/code_exec_grpo.yaml`)

- `model`: MiniMind checkpoint metadata.
- `policy`: generation hyper-parameters (max tokens, temperature, top-p, etc.).
- `env`: system prompt, max tool turns, reward shaping, timeout penalties.
- `rollout`: number of episodes per GRPO iteration.
- `train`: optimizer LR, number of iterations, clipping, normalization flags.

### Web Search QA (`agent_rl/configs/search_qa_grpo.yaml`)

- `env.system_prompt`: instructs policies to search before answering.
- `env.max_turns`: maximum assistant turns before a timeout penalty kicks in.
- `tools.search`:
  - `mode`: `web` (default) or `corpus`.
  - `provider`: `tavily`, `serper`, `bing`, or a custom string + `endpoint`.
  - `api_key_env`: env var read at runtime (`TAVILY_API_KEY` by default).
  - `default_params`: provider-specific knobs (search depth, result counts, etc.).
  - `timeout`, `max_retries`, `backoff_factor`, `rate_limit_per_min`, `max_snippet_chars`.

To supply credentials:

```bash
export TAVILY_API_KEY="sk-..."        # macOS/Linux
setx TAVILY_API_KEY "sk-..."          # Windows PowerShell (persistent)
```

If `mode: corpus`, the registry falls back to the offline `SearchTool` and uses any
documents embedded in the dataset record.

---

## Running Training Jobs

### 1. Code Execution Agent (MiniMind policy)

```bash
python agent_rl/scripts/train_code_exec_agent.py \
  --config agent_rl/configs/code_exec_grpo.yaml \
  --dataset-dir dataset/code_tasks \
  --model-path model \
  --device cuda:0 \
  --preview-tasks 2 \
  --async-rollout-workers 4
```

### 2. Web Search QA Agent (HuggingFace policy example)

```bash
python agent_rl/scripts/train_search_agent.py \
  --config agent_rl/configs/search_qa_grpo.yaml \
  --dataset-dir dataset/search_tasks \
  --policy-type huggingface \
  --hf-model-name meta-llama/Llama-3-8b-instruct \
  --device cuda:0 \
  --preview-tasks 2 \
  --async-rollout-workers 2
```

Workflow outline for both scripts:

1. Parse YAML config + CLI overrides.
2. Build the requested policy (`MiniMindPolicy` or `HuggingFacePolicy`) and AdamW optimizer.
3. Create tool registry (code exec / web search) injected into the environment.
4. Instantiate deterministic train/eval samplers from JSONL datasets.
5. Run synchronous or asynchronous rollouts to collect episodes.
6. Call `GRPOTrainer.update` per iteration; log metrics to stdout or your preferred logger.

> Tip: use `--preview-tasks N` to log sample prompts before training and validate that
> datasets/tooling are wired correctly.

---

## Monitoring & Debugging

- **Stdout Logs** – Each rollout reports reward, turn counts, and tool-call traces. Search
  history from `SearchEnv` is exposed via `info["search_history"]` for inspection.
- **Async Executor** – If you enable `--async-rollout-workers > 1`, set `PYTHONWARNINGS=ignore`
  or tune thread counts to match CPU/network capacity.
- **Linter** – Run `python -m ruff check agent_rl` if you modify core modules.
- **Repro Seeds** – CLI `--seed` argument ensures deterministic dataset ordering and
  rollouts (policy sampling still depends on RNG state).

---

## Extending the Framework

1. **Add a New Tool**
   - Subclass `BaseTool`, implement `__call__`, and register it in `ToolRegistry`.
   - Update environment prompts so the agent knows when to use it.

2. **New Environment**
   - Derive from `AgentEnv`, handle `reset/step`, and provide reward logic.
   - Extend `agent_rl/scripts/train_*` with a new entry point if needed.

3. **Custom Policies**
   - Implement `BasePolicy` interface (see `minimind_policy.py` for reference).
   - Ensure `generate_action` returns text, metadata, and optional `<tool_call>` blocks.

4. **Evaluation Loops**
   - Hook into `agent_rl/core/rollout.RolloutWorker` to collect eval episodes.
   - Use `CodeTaskSampler` with `shuffle=False` for deterministic evaluation sweeps.

---

## Troubleshooting FAQ

| Symptom | Potential Fix |
|---------|---------------|
| `KeyError: Tool 'search' is not registered.` | Check your YAML config; ensure `tools.search.mode` is set and API keys are loaded. |
| `Search failed after multiple retries` | Verify outbound network access and provider status. Increase `timeout` or `max_retries`. |
| Rollouts end immediately with timeout penalty | Increase `env.max_turns` or ensure the model emits valid `<tool_call>` JSON. |
| `RuntimeError: policy.generate_action` | Confirm the model path/device is correct and that HuggingFace weights fit in GPU memory. |
| Tool-call parsing errors | The policy must wrap JSON payloads inside `<tool_call>{...}</tool_call>`. See `agent/tool_parser.py`. |

---

## Contributing

1. Fork & clone the repository.
2. Create a feature branch: `git checkout -b feature/my-change`.
3. Run formatting/linting (e.g., `ruff`, `black`) before pushing.
4. Submit a PR with:
   - Summary of changes.
   - Testing evidence (commands/logs).
   - Any config or credential impacts (e.g., new env vars).

Open issues or feature ideas are welcome—see `design.md` for roadmap discussions.

---

Happy training!