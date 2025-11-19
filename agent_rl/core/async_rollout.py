"""
Utilities for running multiple rollout episodes concurrently.
"""

from __future__ import annotations

import itertools
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from agent_rl.core.rollout import RolloutWorker
from agent_rl.core.trajectory import Episode


WorkerFactory = Callable[[], RolloutWorker]


class AsyncRolloutExecutor:
    """
    Thread-based executor that dispatches rollout jobs to a pool of workers.
    """

    def __init__(self, worker_factory: WorkerFactory, max_workers: int = 2) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive for AsyncRolloutExecutor.")
        self.worker_factory = worker_factory
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._workers = [self.worker_factory() for _ in range(max_workers)]
        self._worker_cycle = itertools.cycle(range(max_workers))

    def shutdown(self) -> None:
        """
        Shut down the underlying executor, waiting for all outstanding jobs.
        """

        self._executor.shutdown(wait=True)

    def run_batch(self, tasks: Sequence[Dict[str, Any]]) -> Tuple[List[Episode], List[Any]]:
        """
        Schedule ``run_episode`` for each task concurrently and return episodes/stats.
        """

        futures: List[Future] = []
        for task in tasks:
            worker = self._next_worker()
            futures.append(self._executor.submit(worker.run_episode, task))
        results = [future.result() for future in futures]
        episodes, stats = zip(*results)
        return list(episodes), list(stats)

    def _next_worker(self) -> RolloutWorker:
        idx = next(self._worker_cycle)
        return self._workers[idx]


__all__ = ["AsyncRolloutExecutor"]

