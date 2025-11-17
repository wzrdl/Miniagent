
---

## 0. 总体目标（你要复刻的核心能力）

结合 veRL + VerlTool + 你现有 minimind（已支持 PPO/GRPO 等 RLAIF、并且 chat_template 已有 `<tool_call>` / `<think>` 标签）这几块能力([GitHub][1])，你的新系统目标可以概括为：

1. **veRL-like RL 核心**

   * 支持 GRPO / PPO / 其他 RL 算法（minimind 已有 GRPO，可扩展）。([GitHub][2])
   * 有清晰的“**trajectory / rollout / trainer / policy**”抽象。([GitHub][2])

2. **VerlTool-like 工具 Agent 层**

   * 工具即环境（tool-as-environment），多轮交互、可持久化状态。([GitHub][3])
   * Actor 和 Env 完全解耦，让 rollout 可以异步 / 分布式。([GitHub][3])

3. **可插拔的 Tool 沙盒**

   * 至少：代码执行工具（本地 / Docker 沙箱），后面可以拓展 search / SQL / web 等。([GitHub][3])

所有东西都“挂”在 minimind 现有的 **model + trainer（RLAIF/GRPO）** 能力上去。

---

## 1. 顶层目录设计（基于 minimind）

在 minimind 现有结构的基础上，目前大致是：([GitHub][1])

```text
minimind/
  dataset/
  images/
  model/
  scripts/
  trainer/
  eval_llm.py
  README*.md
  ...
```

建议新增一套用于 Agent-RL 的子系统（名字你可以自己定，我先用 `agent_rl`）：

```text
minimind/
  agent_rl/
    __init__.py

    # 1. 核心抽象层：环境 / 工具 / 轨迹 / policy
    core/
      policy_base.py        # 把 minimind LLM 封成 RL Policy (logits, logprob, value)
      trajectory.py         # Step / Episode / TrajectoryBuffer 定义
      rollout.py            # RolloutWorker / Sampler（同步版起步，后面再做异步）
      trainer_grpo.py       # 复用你现有 GRPO，封装成通用 Trainer 接口
      trainer_ppo.py        # 以后可以加
      reward_fn.py          # 各任务的 reward 计算（Code / Math / SQL 等）

    # 2. 环境层：task 封装 + 多轮对话逻辑
    envs/
      base_env.py           # 类似 Gym Env：reset/step，面向“一个任务”
      code_exec_env.py      # Code 任务环境（调用 CodeExecTool）
      search_env.py         # 示例：检索任务环境
      ...
    
    # 3. 工具 & 工具服务器
    tools/
      base_tool.py          # Tool 抽象类：name/description/call()
      registry.py           # ToolRegistry，集中管理所有工具
      code_exec_tool.py     # 本地/容器代码沙箱（Python 起步）
      search_tool.py        # 简单的本地文档检索/HTTP search
      ...
    server/
      tool_server.py        # 可选：把工具通过本地 RPC/HTTP 暴露出来（类似 VerlTool 的 tool server）:contentReference[oaicite:7]{index=7}

    # 4. Agent 封装：prompt + tool 调用解析
    agent/
      messages.py           # Message / Role / ToolCall 数据结构
      chat_template.py      # 复用 minimind 已有 chat_template，加入 tool_call 格式
      tool_parser.py        # 从 LLM 输出中解析 tool 调用（JSON/tag 解析）
      agent_loop.py         # “多轮对话 + 工具调用”的主循环（一个 episode 内）

    # 5. 配置 & 训练脚本
    configs/
      code_exec_grpo.yaml   # 训练 code-exec agent 的配置
      search_qa_grpo.yaml
      ...
    scripts/
      train_code_exec_agent.py  # CLI 脚本，读取 config，启动 rollout + trainer
      train_search_agent.py
```

这套结构基本就是：

> minimind 的 **model + RLAIF 算法**
> ＋ veRL 的 **RL 核心抽象**([GitHub][2])
> ＋ VerlTool 的 **tool-as-environment + tool server** 设计([GitHub][3])

---

## 2. 各层职责拆解

### 2.1 Model / Policy 层（复用 minimind）

目标：把 minimind 的 LLM 变成一个 RL Policy，暴露统一接口。

**文件：`agent_rl/core/policy_base.py`**

关键接口：

```python
class PolicyOutput(NamedTuple):
    logprobs: torch.Tensor   # [B, T]
    values: torch.Tensor | None
    logits: torch.Tensor     # [B, T, Vocab]

class BasePolicy(nn.Module):
    def forward(self, input_ids, attention_mask=None) -> PolicyOutput:
        raise NotImplementedError

    @torch.no_grad()
    def generate_action(self, messages, max_new_tokens=128, **gen_kwargs):
        """
        - 将 messages 转成 chat_template（含 <tool_call> 的格式）
        - 调用 minimind 的 generate，得到输出
        - 同时保留生成 token 的 logprobs，供 RL 用
        """
```

实现上你可以直接在这里引用 minimind 的 `model/`、`trainer/` 里已有的模型加载 / 生成逻辑，尤其是 RLAIF 部分已经有 logprob 计算和 GRPO 的样板。([GitHub][4])

---

### 2.2 Trajectory / Rollout / Trainer（veRL 核心）

借鉴 veRL 的设计，你需要明确三个概念：trajectory、rollout worker、trainer。([GitHub][2])

**`trajectory.py`：**

```python
@dataclass
class Step:
    obs: dict            # 例如：tokenized input_ids, attention_mask
    action: dict         # 例如：生成的 token ids / tool_call 对象
    logprob: torch.Tensor
    reward: float
    done: bool
    info: dict           # 可存工具名、错误类型、耗时等

@dataclass
class Episode:
    steps: list[Step]
    episode_id: str
    task_id: str
    final_reward: float | None = None

class TrajectoryBuffer:
    def add(self, episode: Episode): ...
    def sample_batch(self, batch_size): ...
```

**`rollout.py`：**

```python
class RolloutWorker:
    def __init__(self, policy, env_cls, env_config, max_steps):
        self.policy = policy
        self.env_cls = env_cls
        self.env_config = env_config
        self.max_steps = max_steps

    def run_episode(self, task_sample) -> Episode:
        env = self.env_cls(task_sample, self.env_config)
        obs = env.reset()
        steps = []

        for t in range(self.max_steps):
            pi_out = self.policy.forward(**obs)
            action = self._sample_action(pi_out)          # 生成文本 & 工具调用
            next_obs, reward, done, info = env.step(action)
            steps.append(Step(obs, action, pi_out.logprobs, reward, done, info))
            obs = next_obs
            if done:
                break

        ep = Episode(steps=steps, episode_id=..., task_id=...)
        ep.final_reward = env.compute_final_reward()
        return ep
```

**`trainer_grpo.py`：**

这里直接复用你在 minimind 里已经写好的 GRPO 算法，只是给它一个统一的 Trainer 外壳，例如：

```python
class GRPOTrainer:
    def __init__(self, policy, optimizer, config):
        ...

    def update(self, episodes: list[Episode]):
        """
        - 把 episode 展开成 token 级别的 (obs, action, logprob_old, reward)
        - 计算 advantage, 组装 GRPO loss
        - 反向传播，更新 policy 参数
        """
```

这样你的 Agent-RL 框架就有了 veRL 那种：**rollout → trajectory → trainer(GRPO/PPO)** 的闭环。

---

### 2.3 Env 层：多轮对话 + Tool-as-environment

参考 VerlTool 的设计，它把“工具交互”视作环境的一部分，每个 `step()` 既可能是对话、也可能是工具调用。([GitHub][3])

**`envs/base_env.py`：**

```python
class AgentEnv(ABC):
    def __init__(self, task_sample, config, tool_manager):
        self.task_sample = task_sample
        self.config = config
        self.tool_manager = tool_manager

    @abstractmethod
    def reset(self) -> dict:
        """返回初始 obs（通常是 messages → tokenized）"""

    @abstractmethod
    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """
        action: 来自 LLM 的输出（文本 + 可能的 tool_call）
        - 解析 tool_call，调用 tool_manager
        - 更新对话状态
        - 返回下一轮 obs、reward、done、info
        """
    
    def compute_final_reward(self) -> float:
        """有的任务在最后一次统一给总 reward，可在此实现"""
```

**`envs/code_exec_env.py`：**

* `task_sample`：可以是一道“修 bug / 写函数”的问题 + 测试用例。
* `reset()`：构造系统提示 + 用户问题，初始化 messages。
* `step(action)`：

  1. 用 `tool_parser` 从 action 中解析出是否有 `<tool_call name="exec_code">...`。
  2. 如果有 → 调 `tool_manager.call("exec_code", code_str)`。
  3. 把执行结果作为“工具消息” append 到 messages，作为下一轮 obs 的输入。
  4. 根据是否通过测试、是否崩溃等信息给中间 reward。

---

### 2.4 Tool 层 + Tool Server

VerlTool 明确提到：**工具调用通过统一 API 接入，工具可以单独测试；Actor 和环境交互完全解耦**。([GitHub][3])

**`tools/base_tool.py`：**

```python
class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def __call__(self, **kwargs) -> str:
        """执行工具，返回字符串结果（会插入到对话里）"""
```

**`tools/registry.py`：**

```python
class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        return self._tools[name]

    def list_tools(self):
        return list(self._tools.values())
```

**`tools/code_exec_tool.py`：**

* 先做一个最简单版：在本地用 `subprocess` 执行 Python 代码（加一些超时 & 安全限制），后面你可以改成 Docker 沙箱。

```python
class CodeExecTool(BaseTool):
    name = "exec_code"
    description = "Execute Python code in a sandbox and return stdout/stderr."

    def __call__(self, code: str, timeout: float = 5.0) -> str:
        # 调用本地 python 或 docker
        ...
```

**`server/tool_server.py`（可选，但建议预留）**

* 把 ToolRegistry 暴露成一个本地 HTTP / Unix socket 服务，便于未来做 **异步 rollout / 分布式工具执行**（VerlTool 有同步 & 异步 rollout 设计）。([GitHub][3])

---

### 2.5 Agent 层：消息结构 & tool_call 解析

你已经有 minimind 的 chat_template，并且 changelog 中提到支持 `<tool_call>` / `<think>` 标签，这非常适合做 agent 层。([GitHub][1])

**`agent/messages.py`：**

```python
@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_name: str | None = None   # tool 消息时用

@dataclass
class ToolCall:
    name: str
    arguments: dict
```

**`agent/chat_template.py`：**

* 封装两件事：

  1. 把 `List[Message]` 转换成带 `<tool_call>` 等 tag 的文本，用 minimind 的 tokenizer 编码；
  2. 反向：从模型输出的 token 反解析出 tool 调用对象（可以交给 `tool_parser`）。

**`agent/tool_parser.py`：**

* 根据你设计的 schema（推荐 JSON 或 `<tool_call>` 标记）解析出 `ToolCall`：

  * 如：模型输出中一段 `<tool_call>{"name": "exec_code", "arguments": {"code": "print(1)"}}</tool_call>`。

**`agent/agent_loop.py`：**

* 这是“一个 episode 内的 Agent 行为”，RolloutWorker 的 `run_episode` 可以复用它：

```python
class ToolCallingAgent:
    def __init__(self, policy, tool_registry, max_turns):
        ...

    def one_turn(self, messages: list[Message]) -> tuple[list[Message], ToolCall | None]:
        # 1. 用 chat_template 打包 messages
        # 2. 调 policy.generate_action()
        # 3. 用 tool_parser 解析 tool_call
        # 4. 返回新的 messages（加上 assistant 回复）和解析出的 tool_call
```

---

## 3. 主训练脚本的整体数据流

以 `scripts/train_code_exec_agent.py` 为例，你的主训练流程可以是：

```python
def main(config_path):
    cfg = load_yaml(config_path)

    # 1. 初始化模型 & policy
    base_model = load_minimind_model(cfg.model)        # 复用 minimind 现有加载逻辑
    policy = MinimindPolicyWrapper(base_model, cfg.policy)

    # 2. 初始化工具
    registry = ToolRegistry()
    registry.register(CodeExecTool(...))
    # 未来可以 registry.register(SearchTool(...))

    # 3. 初始化 env factory
    env_cls = CodeExecEnv

    # 4. 初始化 RL 组件
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.train.lr)
    trainer = GRPOTrainer(policy, optimizer, cfg.train)
    buffer = TrajectoryBuffer()

    # 5. 训练 loop
    for it in range(cfg.train.num_iters):
        episodes = []
        for _ in range(cfg.rollout.num_episodes_per_iter):
            task_sample = sample_code_task(...)
            worker = RolloutWorker(policy, env_cls, cfg.env, max_steps=cfg.env.max_steps)
            ep = worker.run_episode(task_sample)
            episodes.append(ep)

        trainer.update(episodes)   # 用 GRPO 更新 minimind 模型参数

        if it % cfg.eval.eval_interval == 0:
            eval_agent_on_holdout(...)
            save_checkpoint(...)
```

这就是你自己的 “mini-verl + verl-tool” 组合。

---

## 4. 建设顺序（推荐路线）

不给时间预估，只给你**步骤顺序**，你可以按这个顺序一点点填：

1. **先把 RL 核心抽象搭起来**

   * 在 `agent_rl/core/` 下实现：`policy_base.py` + `trajectory.py` + `trainer_grpo.py`。
   * 用一个 **不带工具的简单 QA env** 验证 GRPO 训练闭环是通的。

2. **加上 Agent 层（消息 + chat_template 封装）**

   * 把 minimind 的 chat_template + `<tool_call>` tag 集成到 `agent/`。
   * 做一个假工具（比如 echo_tool），训练时 reward 简单模拟，先跑通“模型输出正确 tool_call JSON”。

3. **实现代码执行 Tool + CodeExecEnv**

   * `tools/code_exec_tool.py`：本地 `subprocess` 版沙箱。
   * `envs/code_exec_env.py`：读取一道 code 任务 + 测试用例，reward 按是否通过测试给。

4. **把“code-exec agent + GRPO”打通**

   * 写 `train_code_exec_agent.py`，让模型逐渐学会：在合适回合调用 `exec_code`，并修正代码直至通过测试。

5. **再往上长：Search / SQL / Web 等任务 + 异步 rollout / tool server**

   * 这一步就已经非常接近 VerlTool 的形态了：tool-as-environment、多任务、多工具、异步 rollout。([GitHub][3])

---

## 5. 一句话总结整套蓝图

> **你在 minimind 里新建一个 `agent_rl/` 子系统：
> 下面有 RL 核心（policy/trajectory/trainer）、工具沙盒（tools + server）、环境抽象（envs）、Agent 封装（messages + tool_parser），再加上训练脚本，把你已有的 GRPO 算法运用在“多轮、带工具调用”的 trajectory 上。**

按照这个蓝图推进，最后你会得到一个：

* 内核类似 veRL（统一的 RL dataflow + GRPO 等算法）；([GitHub][2])
* 外壳类似 VerlTool（tool-as-environment、tool server、多轮 agent 训练）；([GitHub][3])
* 但完全贴合 minimind 代码风格、并保持你现在“纯 PyTorch、从 0 重构”的哲学。([GitHub][4])

如果你接下来想从哪一层开始写（比如先上 CodeExecEnv），你直接告诉我那个模块名字，我可以帮你把第一个版本的**具体代码骨架**撸出来。

[1]: https://github.com/jingyaogong/minimind "GitHub - jingyaogong/minimind:  「大模型」2小时完全从0训练26M的小参数GPT！ Train a 26M-parameter GPT from scratch in just 2h!"
[2]: https://github.com/volcengine/verl "GitHub - volcengine/verl: verl: Volcano Engine Reinforcement Learning for LLMs"
[3]: https://github.com/TIGER-AI-Lab/verl-tool "GitHub - TIGER-AI-Lab/verl-tool: A version of verl to support diverse tool use"
[4]: https://raw.githubusercontent.com/jingyaogong/minimind/master/README.md?utm_source=chatgpt.com "https://raw.githubusercontent.com/jingyaogong/mini..."
