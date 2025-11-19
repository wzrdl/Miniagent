"""
Dataset utilities for search QA tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import json


@dataclass(slots=True)
class SearchTaskSample:
    task_id: str
    question: str
    answer: str
    documents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SearchTaskSample":
        return cls(
            task_id=payload["task_id"],
            question=payload["question"],
            answer=payload.get("answer", ""),
            documents=list(payload.get("documents", [])),
            metadata=payload.get("metadata", {}),
        )

    def to_env_task(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "answer": self.answer,
            "documents": self.documents,
            "metadata": self.metadata,
        }


class SearchTaskDataset(Sequence[SearchTaskSample]):
    def __init__(self, samples: Iterable[SearchTaskSample]) -> None:
        self._samples = list(samples)

    @classmethod
    def from_path(cls, path: str | Path) -> "SearchTaskDataset":
        records = []
        with Path(path).open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                records.append(SearchTaskSample.from_dict(json.loads(line)))
        return cls(records)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> SearchTaskSample:
        return self._samples[index]


__all__ = ["SearchTaskDataset", "SearchTaskSample"]

