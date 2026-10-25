"""Lightweight distributed-rank helpers for pre-Lightning startup
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any, TypeVar


_GLOBAL_RANK_ENV_KEYS = (
    "RANK",
    "SLURM_PROCID",
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "MV2_COMM_WORLD_RANK",
    "JSM_NAMESPACE_RANK",
)

_ReturnT = TypeVar("_ReturnT")


def global_rank(environment: Mapping[str, str] | None = None) -> int:
    """Return the launcher-provided global rank, defaulting to zero.

    ``LOCAL_RANK`` is deliberately not used on its own: every node has a local
    rank zero, so treating it as a global rank causes concurrent writes in a
    multi-node job.  Torchrun, Slurm and common MPI launchers all expose one of
    the global-rank variables checked here.
    """
    env = os.environ if environment is None else environment
    for key in _GLOBAL_RANK_ENV_KEYS:
        value = env.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except ValueError as exc:
                raise ValueError(
                    f"Environment variable {key} must be an integer, got {value!r}"
                ) from exc

    required = ("NODE_RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK")
    if all(env.get(key) not in (None, "") for key in required):
        try:
            return (
                int(env["NODE_RANK"]) * int(env["LOCAL_WORLD_SIZE"])
                + int(env["LOCAL_RANK"])
            )
        except ValueError as exc:
            values = {key: env[key] for key in required}
            raise ValueError(
                f"Distributed rank environment values must be integers: {values}"
            ) from exc

    return 0


def is_global_zero(environment: Mapping[str, str] | None = None) -> bool:
    """Return whether the current launcher process is global rank zero."""
    return global_rank(environment) == 0


def rank_zero_only(function: Callable[..., _ReturnT]) -> Callable[..., _ReturnT | None]:
    """Run an early-startup side effect only on launcher global rank zero."""
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> _ReturnT | None:
        if is_global_zero():
            return function(*args, **kwargs)
        return None

    return wrapped
