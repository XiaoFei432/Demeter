"""Jump consistent hash helper for elastic bucket repartitioning."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Mapping


def jump_consistent_hash(key: str | bytes | int, buckets: int) -> int:
    """Return a stable bucket id using the JumpHash algorithm."""

    if buckets <= 0:
        raise ValueError("buckets must be positive")
    if isinstance(key, int):
        value = key & 0xFFFFFFFFFFFFFFFF
    else:
        payload = key if isinstance(key, bytes) else str(key).encode("utf-8")
        value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")

    jump = -1
    candidate = 0
    while candidate < buckets:
        jump = candidate
        value = (value * 2862933555777941757 + 1) & 0xFFFFFFFFFFFFFFFF
        candidate = int((jump + 1) * (1 << 31) / ((value >> 33) + 1))
    return jump


class BucketPartitioner:
    """Map bucket keys to DoP groups with minimal movement when DoP changes."""

    def assign(self, keys: Iterable[str | bytes | int], parallelism: int) -> Dict[str | bytes | int, int]:
        return {key: jump_consistent_hash(key, parallelism) for key in keys}

    def migration_plan(
        self,
        previous: Mapping[str | bytes | int, int],
        new_parallelism: int,
    ) -> Dict[str | bytes | int, tuple[int, int]]:
        moved: Dict[str | bytes | int, tuple[int, int]] = {}
        for key, old_bucket in previous.items():
            new_bucket = jump_consistent_hash(key, new_parallelism)
            if new_bucket != old_bucket:
                moved[key] = (old_bucket, new_bucket)
        return moved
