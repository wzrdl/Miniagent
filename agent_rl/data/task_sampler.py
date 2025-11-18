"""
Task sampling helpers that bridge ``dataset/code_task_dataset.py`` and rollouts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

from dataset import CodeTaskDataset, load_code_task_splits


@dataclass
class CodeTaskSampler:
    """
    Provides deterministic iteration over a :class:`CodeTaskDataset`.

    The sampler shuffles indices at the beginning of every pass (when ``shuffle``
    is enabled) and exposes ``next``/``iter_batches`` helpers that already
    convert :class:`CodeTaskSample` objects into dictionaries consumable by
    `CodeExecEnv`.
    """

    dataset: CodeTaskDataset
    shuffle: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        # Maintain a cached list of dataset indices so we can reshuffle cheaply.
        self._indices: List[int] = list(range(len(self.dataset)))
        self._rng = random.Random(self.seed)
        self._cursor = 0
        if self.shuffle:
            self._rng.shuffle(self._indices)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        """
        Yields infinite task dicts, reshuffling once all samples are consumed.
        """

        while True:
            yield from self.iter_batches(batch_size=1)

    def iter_batches(self, batch_size: int) -> Iterator[List[Dict[str, object]]]:
        """
        Yield batches of ``batch_size`` task dictionaries.

        Args:
            batch_size: Number of tasks per yielded batch.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        while True:
            batch: List[Dict[str, object]] = []
            for _ in range(batch_size):
                batch.append(self._next_task())
            yield batch

    def next(self, batch_size: int = 1) -> List[Dict[str, object]]:
        """
        Return a single batch of tasks (synchronously) without creating an iterator.
        """

        batches = next(self.iter_batches(batch_size))
        return batches

    def _next_task(self) -> Dict[str, object]:
        if self._cursor >= len(self._indices):
            self._cursor = 0
            if self.shuffle:
                self._rng.shuffle(self._indices)
        dataset_index = self._indices[self._cursor]
        self._cursor += 1
        sample = self.dataset[dataset_index]
        # The environment expects a plain dictionary, so convert from dataclass.
        return sample.to_env_task()


def load_code_task_dataset(
    data_dir: str | Path,
    *,
    train_split: str = "train",
    eval_split: str = "eval",
    seed: int | None = None,
) -> Dict[str, CodeTaskDataset]:
    """
    Convenience wrapper that loads train/eval splits from ``dataset/code_tasks``.

    Args:
        data_dir: Directory containing ``*.jsonl`` split files.
        train_split: File stem for the training split.
        eval_split: File stem for the evaluation split.
        seed: Optional deterministic seed shared by both splits.
    """

    split_names: List[str] = []
    for name in (train_split, eval_split):
        if name not in split_names:
            split_names.append(name)
    datasets = load_code_task_splits(data_dir, splits=tuple(split_names), seed=seed)
    return datasets


__all__ = ["CodeTaskSampler", "load_code_task_dataset"]

