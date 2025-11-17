# Miniagent

Miniagent extends the original minimind project with a veRL-style reinforcement
learning stack plus VerlTool-inspired tool environments. The new `agent_rl`
package provides:

- `core/`: policy wrappers, trajectory buffers, rollout workers, GRPO trainer
- `envs/`: task environments such as `CodeExecEnv` and `SearchEnv`
- `tools/`: sandboxed tool definitions (`CodeExecTool`, `SearchTool`), registry,
  and an optional HTTP `ToolServer`
- `agent/`: message schema, chat template utilities, tool-call parser, and a
  tool-aware agent loop
- `configs/` & `scripts/`: YAML configs and CLI entry points for training
  agents (e.g., `train_code_exec_agent.py`)

The goal is to make it easy to plug minimind LLMs into multi-turn,
tool-augmented RL training loops. Start by exploring `agent_rl/configs` and
running the corresponding scripts to wire up your model, tasks, and reward
functions.