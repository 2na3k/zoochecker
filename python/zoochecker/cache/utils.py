"""Shared MLX utilities for the KV cache backends."""

import mlx.core as mx

from ._base import KVPrefix


def compact_snapshot(
    prefix_len: int, layers: tuple[tuple[mx.array, mx.array], ...]
) -> KVPrefix[mx.array]:
    # A gather creates independent storage even when the prefix is a strided
    # view into a larger buffer. Evaluate it here to drop the source graph.
    positions = mx.arange(prefix_len)
    copied = tuple(
        (mx.take(keys, positions, axis=2), mx.take(values, positions, axis=2))
        for keys, values in layers
    )
    mx.eval(copied)
    return KVPrefix(prefix_len, copied)
