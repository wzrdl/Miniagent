"""
Policy wrapper around MiniMind causal language models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import json

import torch
from transformers import AutoTokenizer

from agent_rl.agent.chat_template import ChatTemplate
from agent_rl.agent.messages import Message
from agent_rl.agent.tool_parser import ToolParser
from agent_rl.core.policy_base import BasePolicy, PolicyOutput
from model.model_minimind import MiniMindForCausalLM


@dataclass
class MiniMindPolicyConfig:
    """
    Configuration bundle used by :class:`MiniMindPolicy`.
    """

    model_path: str
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True


class MiniMindPolicy(BasePolicy):
    """
    Adapter that exposes forward/generate APIs expected by rollout workers.
    """

    def __init__(self, config: MiniMindPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            # Generation requires a pad token; fall back to EOS when unspecified.
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = MiniMindForCausalLM.from_pretrained(config.model_path)
        self.model.to(config.device, dtype=config.dtype)
        self.chat_template = ChatTemplate()
        self.tool_parser = ToolParser()

    def forward(self, messages: Sequence[Message], **_) -> PolicyOutput:
        """
        Compute token-level logits/logprobs for the current conversation prefix.
        """

        prompt = self.chat_template.encode(messages)
        tokenized = self._tokenize(prompt)
        outputs = self.model(**tokenized)
        logits = outputs.logits[:, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)
        return PolicyOutput(logprobs=logprobs, logits=logits, values=None, metadata={"prompt": prompt})

    @torch.no_grad()
    def generate_action(
        self,
        messages: Sequence[Dict[str, Any] | Message],
        max_new_tokens: int | None = None,
        **gen_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Produce an assistant response plus optional tool-call markup.
        """

        normalized_messages = self._normalize_messages(messages)
        prompt = self.chat_template.encode(normalized_messages)
        tokenized = self._tokenize(prompt)
        generation = self.model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=gen_kwargs.get("temperature", self.config.temperature),
            top_p=gen_kwargs.get("top_p", self.config.top_p),
            do_sample=gen_kwargs.get("do_sample", self.config.do_sample),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        prompt_length = tokenized["input_ids"].shape[-1]
        completion_ids = generation[0, prompt_length:]
        decoded = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        tool_call = self._extract_tool_call(decoded)
        response_logprobs = self._gather_generated_logprobs(generation.scores, completion_ids)
        metadata = {
            "prompt_text": prompt,
            "response_ids": completion_ids.detach().cpu(),
            "response_logprobs": response_logprobs.detach().cpu(),
        }
        return {"content": decoded, "tool_call": tool_call, "metadata": metadata}

    def _tokenize(self, prompt: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        return encoded

    @staticmethod
    def _normalize_messages(messages: Sequence[Dict[str, Any] | Message]) -> List[Message]:
        """
        Ensure we always operate on :class:`Message` objects.
        """

        normalized: list[Message] = []
        for msg in messages:
            if isinstance(msg, Message):
                normalized.append(msg)
            else:
                normalized.append(Message(role=msg["role"], content=msg["content"], tool_name=msg.get("tool_name")))
        return normalized

    def _extract_tool_call(self, text: str) -> str | None:
        """
        Return the first <tool_call>...</tool_call> block if one exists.
        """

        if "<tool_call>" not in text:
            return None
        try:
            # Reuse ToolParser to ensure the payload is valid JSON.
            self.tool_parser.parse(text)
            start = text.index("<tool_call>")
            end = text.index("</tool_call>") + len("</tool_call>")
            return text[start:end]
        except (ValueError, json.JSONDecodeError):  # type: ignore[attr-defined]
            return None

    def compute_sequence_logprob(self, prompt: str, response_ids: torch.Tensor) -> torch.Tensor:
        """
        Recompute log probabilities of the response under the current policy.
        """

        prompt_encoding = self._tokenize(prompt)
        prompt_ids = prompt_encoding["input_ids"]
        response_ids = response_ids.unsqueeze(0)
        concat_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.ones_like(concat_ids, device=self.config.device)
        outputs = self.model(input_ids=concat_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        logprobs = torch.log_softmax(logits, dim=-1)
        target_ids = concat_ids[:, 1:]
        token_logprobs = logprobs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        response_len = response_ids.shape[-1]
        return token_logprobs[:, -response_len:].squeeze(0)

    def _gather_generated_logprobs(self, scores: List[torch.Tensor], completion_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert generation step logits into token-level log probabilities.
        """

        logprob_values: List[torch.Tensor] = []
        for step_scores, token_id in zip(scores, completion_ids):
            step_logprob = torch.log_softmax(step_scores, dim=-1)[0, token_id]
            logprob_values.append(step_logprob)
        return torch.stack(logprob_values)


__all__ = ["MiniMindPolicy", "MiniMindPolicyConfig"]

