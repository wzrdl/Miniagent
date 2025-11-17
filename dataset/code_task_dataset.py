"""
Utilities for loading reinforcement-learning code tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import json
import random


@dataclass(slots=True)
class CodeTestCase:
    """
    Metadata describing one executable test case.
    """

    input: str
    output: str
    description: str | None = None
    timeout: float | None = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CodeTestCase":
        return cls(
            input=payload.get("input", ""),
            output=payload.get("output", ""),
            description=payload.get("description"),
            timeout=payload.get("timeout"),
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "input": self.input,
            "output": self.output,
        }
        if self.description is not None:
            data["description"] = self.description
        if self.timeout is not None:
            data["timeout"] = self.timeout
        return data


@dataclass(slots=True)
class CodeTaskSample:
    """
    Canonical representation of a single code-generation task.
    """

    task_id: str
    prompt: str
    reference_solution: str
    tests: list[CodeTestCase] = field(default_factory=list)
    language: str = "python"
    starter_code: str | None = None
    ground_truth: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CodeTaskSample":
        tests = [CodeTestCase.from_dict(item) for item in payload.get("tests", [])]
        return cls(
            task_id=payload["task_id"],
            prompt=payload["prompt"],
            reference_solution=payload.get("reference_solution", "").strip(),
            tests=tests,
            language=payload.get("language", "python"),
            starter_code=payload.get("starter_code"),
            ground_truth=payload.get("ground_truth"),
            metadata=payload.get("metadata", {}),
        )

    def to_env_task(self) -> Dict[str, Any]:
        """
        Convert the sample into a dict understood by agent environments.
        """

        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "reference_solution": self.reference_solution,
            "tests": [test.to_dict() for test in self.tests],
            "language": self.language,
            "starter_code": self.starter_code,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }


class CodeTaskDataset(Sequence[CodeTaskSample]):
    """
    Lightweight dataset wrapper for JSON/JSONL code tasks.
    """

    def __init__(
        self,
        samples: Iterable[CodeTaskSample],
        *,
        split: str | None = None,
        seed: int | None = None,
    ) -> None:
        self._samples: List[CodeTaskSample] = list(samples)
        self.split = split
        self._rng = random.Random(seed)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        split: str | None = None,
        seed: int | None = None,
    ) -> "CodeTaskDataset":
        file_path = cls._resolve_path(path, split)
        samples = [CodeTaskSample.from_dict(record) for record in cls._load_records(file_path)]
        resolved_split = split or file_path.stem
        return cls(samples, split=resolved_split, seed=seed)

    @staticmethod
    def _resolve_path(path: str | Path, split: str | None) -> Path:
        path = Path(path)
        if path.is_dir():
            if split is None:
                raise ValueError("When `path` is a directory you must supply `split`.")
            candidate = path / f"{split}.jsonl"
            if not candidate.exists():
                raise FileNotFoundError(f"Expected file '{candidate}' for split '{split}'.")
            return candidate
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _load_records(path: Path) -> List[Dict[str, Any]]:
        if path.suffix == ".jsonl":
            records: List[Dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            return records
        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
            if isinstance(payload, list):
                return payload
            raise ValueError(f"File '{path}' must contain a list when using .json format.")
        raise ValueError(f"Unsupported file extension '{path.suffix}' (expected .jsonl or .json)")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._samples)

    def __getitem__(self, index: int) -> CodeTaskSample:  # type: ignore[override]
        return self._samples[index]

    def __iter__(self) -> Iterator[CodeTaskSample]:
        return iter(self._samples)

    def sample(self, k: int = 1, *, with_replacement: bool = False) -> List[CodeTaskSample]:
        if k <= 0:
            raise ValueError("k must be positive")
        if with_replacement:
            return [self._rng.choice(self._samples) for _ in range(k)]
        if k > len(self._samples):
            raise ValueError("Cannot sample more tasks than available without replacement")
        return random.sample(self._samples, k)

    def iter_batches(self, batch_size: int, *, shuffle: bool = True) -> Iterator[List[CodeTaskSample]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = list(range(len(self._samples)))
        if shuffle:
            self._rng.shuffle(indices)
        for idx in range(0, len(indices), batch_size):
            batch_indices = indices[idx : idx + batch_size]
            yield [self._samples[i] for i in batch_indices]


def load_code_task_splits(
    data_dir: str | Path,
    *,
    splits: Sequence[str] = ("train", "eval"),
    seed: int | None = None,
) -> Dict[str, CodeTaskDataset]:
    """
    Convenience loader that maps split name -> dataset.
    """

    datasets: Dict[str, CodeTaskDataset] = {}
    for split in splits:
        datasets[split] = CodeTaskDataset.from_path(data_dir, split=split, seed=seed)
    return datasets


__all__ = ["CodeTaskDataset", "CodeTaskSample", "CodeTestCase", "load_code_task_splits"]

