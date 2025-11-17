[31 tools called]

## VERL 架构解析与我们可复用的设计

### 1. 框架总览（Ray 主流程 + 多角色工作流）
- `main_ppo.py` 中的 `TaskRunner` 会先根据配置注册 actor、critic、reward、reference-policy 等角色，再加载权重、tokenizer、reward 函数和数据集，最后交给 `RayPPOTrainer` 运行。代码直接说明了 reward 可以来自“规则打分/模型打分/发送代码题到沙箱”的多源组合。
```259:341:verl/trainer/main_ppo.py
        # - for code related prompt, we send to a sandbox if there are test cases
        self.add_reward_model_worker(config)
        ...
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        ...
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            ...
        )
        trainer.init_workers()
        trainer.fit()
```

### 2. HuggingFace 模型如何接入
1. **注册不同架构**：`ModelRegistry` 把 HuggingFace 的模型类名映射到 verl 自己的并行实现，actor/critic 可以按 `value` 标记获取不同类。
```22:58:verl/models/registry.py
_MODELS = {
    "LlamaForCausalLM": ("llama", (...)),
    "Qwen2ForCausalLM": ("qwen2", (...)),
    ...
}
class ModelRegistry:
    @staticmethod
    def load_model_cls(model_arch: str, value=False) -> Optional[type[nn.Module]]:
        module = importlib.import_module(f"verl.models.{module_name}.{megatron}.modeling_{module_name}_megatron")
        return getattr(module, model_cls_name, None)
```
2. **权重/Tokenizer**：`TaskRunner` 用 `copy_to_local` 拉 checkpoint，再通过 `hf_tokenizer`、`hf_processor` 实例化，确保和 HuggingFace 生态一致（见上一段代码块）。
3. **推理/采样**：`HFRollout` 把 HuggingFace `generate` 过程包装到 `DataProto` 输入/输出，支持 FSDP `summon_full_params`、Ray micro-batching、采样/greedy 模式切换。
```40:177:verl/workers/rollout/hf_rollout.py
class HFRollout(BaseRollout):
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_prompts = prompts.chunk(...)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        return DataProto.concat(output)
    @torch.no_grad()
    def _generate_minibatch(...):
        kwargs = {"do_sample": True, "top_p": top_p, ...}
        generation_config = GenerationConfig(**kwargs)
        self.module.generate(
            input_ids=idx,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=response_length,
            ...
        )
        ...
        return DataProto(batch=batch)
```
4. **自定义 kernel/并行策略**：在 `models/transformers/llama.py` 里，官方 `Attention.forward` 被改写以适配 flash-attn、Ulysses sequence parallel、RoPE cache、dtype 强制转换等，保证 HF 模型能在大规模并行训练中工作。
```42:167:verl/models/transformers/llama.py
def llama_flash_attn_forward(...):
    query_states = self.q_proj(hidden_states)
    ...
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    if ulysses_sp_size > 1:
        query_states = gather_seq_scatter_heads(...)
    ...
    query_states, key_states = apply_rotary_pos_emb(...)
    if past_key_value is not None:
        key_states, value_states = past_key_value.update(...)
    attn_output = _flash_attention_forward(...)
    return attn_output, attn_weights, past_key_value
```

### 3. 数据入口与批处理
- **Hydra 数据配置**：`create_rl_dataset` 支持自定义 Dataset、动态数据生成（DynamicGenDataset）或默认的 `RLHFDataset`，因此扩展性很好。
```344:389:verl/trainer/main_ppo.py
if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
    dataset_cls = load_extern_type(...)
elif "datagen" in data_config ...:
    from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset
    dataset_cls = DynamicGenDataset
else:
    dataset_cls = RLHFDataset
dataset = dataset_cls(
    data_files=data_paths,
    tokenizer=tokenizer,
    processor=processor,
    config=data_config,
    max_samples=max_samples,
)
```
- **标准化数据协议**：`DataProto` 用 `TensorDict` + `meta_info` 描述一个 batch，可 slice/concat/序列化，贯穿 rollouts、trainer、reward 等模块。
```328:352:verl/protocol.py
@dataclass
class DataProto:
    """
    A DataProto is a data structure that aims to provide a standard protocol for data exchange between functions.
    """
    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)
    def __len__(self):
        if self.batch is not None:
            return self.batch.batch_size[0]
        ...
```
- **Rollout 请求建模**：`AsyncRolloutRequest` 将 messages、tool schemas、多模态输入、tokenized tensors 和 reward 字典打包，让异步 rollout/工具调用拥有统一状态机。
```81:150:verl/workers/rollout/schemas.py
class AsyncRolloutRequest(BaseModel):
    request_id: str
    state: AsyncRolloutRequestStateEnum
    messages: list[Message]
    ...
    tool_schemas: Optional[list[OpenAIFunctionToolSchema]] = None
    tools_kwargs: dict[str, Any] = {}
    ...
    @model_validator(mode="before")
    def initialize_request(cls, values):
        ...
        tools = (
            [tool.model_dump() for tool in tool_schemas]
            if (tool_schemas := values.get("tool_schemas", [])) else None
        )
        ...
        tokenization_dict_with_prompt = cls._handle_apply_chat_template(...)
```

### 4. GRPO 在 VERL 中的实现
- GRPO 被当作一个 Advantage Estimator 注册在 `core_algos.py`，对每个 prompt 的多条采样结果做 group 内均值/方差归一化或 Dr.GRPO 变体，并把标量 advantage 广播到 token 维度，再返回 `(advantages, returns)`。
```264:328:verl/trainer/ppo/core_algos.py
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, ...):
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    ...
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0)
            id2std[idx] = torch.tensor(1.0)
        else:
            scores_tensor = torch.stack(id2score[idx])
            id2mean[idx] = torch.mean(scores_tensor)
            id2std[idx] = torch.std(scores_tensor)
    ...
    scores = scores.unsqueeze(-1) * response_mask
    return scores, scores
```
- `ppo_trainer.yaml`/`algorithm` config 里提供 `adv_estimator=grpo`、`norm_adv_by_std_in_grpo`、`use_kl_loss` 等开关，`RayPPOTrainer` 会按配置创建 group、控制 `actor_rollout_ref.rollout.n`（每 prompt 多少 completion），并且跳过 critic 价值网络，从而得到 GRPO 的“无价值函数”训练流程。

### 5. 外部工具与交互机制
- `BaseTool` 规范了工具的 schema、创建/执行/奖励/释放接口，并用 `rollout_trace_op` 包装执行过程，方便在 rollout 中插拔。
```24:87:verl/tools/base_tool.py
class BaseTool:
    """Base class for tools."""
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.tool_schema = tool_schema or self.get_openai_tool_schema()
        self.name = self.tool_schema.function.name
    async def create(...):
        return str(uuid4()), ToolResponse()
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        return ToolResponse(text="Updated the tool state."), 0.0, {}
    async def calc_reward(...):
        return 0.0
```
- `AsyncRolloutRequest` 将 `tool_schemas`、`tools_kwargs` 纳入序列化数据结构，rollout worker 解析后即可根据对话/工具模板生成 prompt，再将工具调用结果写回 messages，实现“工具即环境”的交互式训练。
- `interactions/`（如 `gsm8k_interaction.py`）可以看作工具层的补充，用于自定义 turn-level 反馈或复杂的“多步骤任务”。

### 6. 代码任务上的强化学习链路
1. **沙箱工具**：`SandboxFusionTool` 通过 Ray 执行池把代码同步投递到外部沙箱（Fusion 镜像），支持并发/限流、超时和语言选择，并将 stdout/stderr 作为工具响应返回。
```101:189:verl/tools/sandbox_fusion_tools.py
class SandboxFusionTool(BaseTool):
    def __init__(...):
        self.execution_pool = init_execution_pool(...)
        self.sandbox_fusion_url = config.get("sandbox_fusion_url", "")
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        code = parameters.get("code", "")
        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        return ToolResponse(text=result), None, None
    def execute_code(...):
        result_status, metadata = _process_single_case(...)
        if metadata["run_status"] == "Finished":
            actual_output = metadata["stdout"] + metadata["stderr"]
            return ToolResponse(text=actual_output)
        else:
            return ToolResponse(text="no stdout here")
```
2. **训练/奖励整合**：`TaskRunner.run` 中的注释明确指出，“若 prompt 带代码测试，就发给 sandbox”，这会在 reward 层把沙箱运行结果和其他打分源融合，随后经 `load_reward_manager` 返回给 trainer（见上面 `main_ppo.py` 片段）。
3. **工具调用路径**：rollout 请求里包含 `tool_schemas`，actor 生成 `<tool_call>`，工具执行后返回文本（可能含测试反馈），reward manager 结合标签或 ground truth 计算 outcome score，最后 GRPO 在组内做比较，从而对“能通过测试”的回答给出正优势。
4. **扩展空间**：同一机制可用于搜索工具（`search_tool.py`）、数学判题（`gsm8k_tool.py`）、Geo3k 等，只需在工具/interaction 中定义对应的 execution & reward，就能把“工具评测”注入 RL。

---

### 对我们 Miniagent 的启示
1. **抽象层级**：可借鉴 `DataProto` + `AsyncRolloutRequest` 的数据协议，统一 policy、rollout、工具的接口，这样才能平滑地插入 HuggingFace、vLLM、SGLang 等不同后端。
2. **HF 接入**：沿用“注册 → 权重拷贝 → tokenizer/processor → rollout wrapper”的套路，实现我们自己的 minimind policy wrapper；必要时对官方模型的 attention/kernel 进行 monkey patch。
3. **GRPO 配置化**：把 advantage estimator、group size、KL 正则等都放进 config，trainer 内部只做流水线控制，方便改成 PPO/RLOO 等算法。
4. **工具/沙箱**：我们可以复用 `BaseTool` 的模式（tool schema + async execute + reward hook）来封装代码执行、搜索、文件操作等扩展工具；reward 端按任务标签选择工具或评测方式。
5. **代码任务 RL**：在我们的 `agent_rl` 中，需要准备：
   - 可扩展的 code-exec 工具（或 HTTP 服务）；
   - 将工具结果写入 observation，并与 reward 计算联动；
   - 为 code prompts 生成多个回答、比较通过率，匹配 GRPO 对“组内对比”的要求。

掌握了这些设计后，就能对照 `prompt.md` 把 Miniagent 的强化学习部分落成：先对齐数据/工具管线，再实现 minimind policy wrapper 和 GRPO 更新逻辑。