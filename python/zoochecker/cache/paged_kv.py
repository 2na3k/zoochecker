"""MLX KV storage backed by a shared pool of fixed-token blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

import mlx.core as mx

from zoochecker.server_args import ServerArgs

from ._base import KVCachePool, KVPrefix, RequestKVCache
from .utils import compact_snapshot


_BLOCK_TOKENS = 16
type _KV = tuple[mx.array, mx.array]


@dataclass(slots=True)
class _Block:
    # Each layer fills this token range independently during the forward pass.
    layers: dict[int, _KV] = field(default_factory=dict)


class _PagedRequestKVCache(RequestKVCache[mx.array]):
    def __init__(self, pool: MLXPagedKVCachePool) -> None:
        self._pool: MLXPagedKVCachePool | None = pool
        self._blocks: list[_Block] = []
        self._offsets = [0] * pool.num_layers
        self._lock = Lock()

    def _active_pool(self) -> MLXPagedKVCachePool:
        if self._pool is None:
            raise RuntimeError("KV cache lease has been released")
        return self._pool

    def _check_layer(self, layer: int) -> None:
        if not 0 <= layer < len(self._offsets):
            raise IndexError("KV cache layer index out of range")

    @property
    def capacity(self) -> int:
        with self._lock:
            return self._active_pool().max_tokens

    def offset(self, layer: int) -> int:
        with self._lock:
            self._active_pool()
            self._check_layer(layer)
            return self._offsets[layer]

    def fetch(self, layer: int) -> _KV:
        with self._lock, mx.stream(mx.gpu):
            pool = self._active_pool()
            self._check_layer(layer)
            return self._read(pool, layer, self._offsets[layer])

    def blocks(self, layer: int) -> tuple[_KV, ...]:
        with self._lock, mx.stream(mx.gpu):
            self._active_pool()
            self._check_layer(layer)
            return self._blocks_at(layer, self._offsets[layer])

    def append(self, layer: int, keys: mx.array, values: mx.array) -> None:
        with self._lock, mx.stream(mx.gpu):
            pool = self._active_pool()
            self._check_layer(layer)
            self._append(pool, [(layer, (keys, values))])

    def update_and_fetch(self, layer: int, keys: mx.array, values: mx.array) -> _KV:
        with self._lock, mx.stream(mx.gpu):
            pool = self._active_pool()
            self._check_layer(layer)
            self._append(pool, [(layer, (keys, values))])
            return self._read(pool, layer, self._offsets[layer])

    def snapshot(self, prefix_len: int) -> KVPrefix[mx.array]:
        with self._lock, mx.stream(mx.gpu):
            pool = self._active_pool()
            if not 0 < prefix_len <= min(self._offsets):
                raise ValueError("Snapshot prefix must be written to every KV cache layer")
            return compact_snapshot(
                prefix_len,
                tuple(self._read(pool, layer, prefix_len) for layer in range(pool.num_layers)),
            )

    def restore(self, state: KVPrefix[mx.array]) -> None:
        with self._lock, mx.stream(mx.gpu):
            pool = self._active_pool()
            if any(self._offsets):
                raise ValueError("Restore requires an empty KV cache lease")
            if state.prefix_len <= 0 or len(state.layers) != pool.num_layers:
                raise ValueError("Invalid KV snapshot length or layer count")
            if any(
                keys.ndim != 4 or keys.shape[2] != state.prefix_len
                for keys, _ in state.layers
            ):
                raise ValueError("KV snapshot layer length differs from its prefix length")
            self._append(pool, list(enumerate(state.layers)))

    def _read(self, pool: MLXPagedKVCachePool, layer: int, length: int) -> _KV:
        chunks = self._blocks_at(layer, length)
        if not chunks:
            empty = mx.zeros(pool.shape(0), dtype=pool.dtype)
            return empty, mx.zeros_like(empty)
        if len(chunks) == 1:
            return chunks[0]
        all_keys, all_values = zip(*chunks)
        return mx.concatenate(all_keys, axis=2), mx.concatenate(all_values, axis=2)

    def _blocks_at(self, layer: int, length: int) -> tuple[_KV, ...]:
        if not length:
            return ()
        chunks = [block.layers[layer] for block in self._blocks[:_blocks_for(length)]]
        tail_length = (length - 1) % _BLOCK_TOKENS + 1
        keys, values = chunks[-1]
        if keys.shape[2] != tail_length:
            chunks[-1] = keys[:, :, :tail_length, :], values[:, :, :tail_length, :]
        return tuple(chunks)

    def _append(self, pool: MLXPagedKVCachePool, writes: list[tuple[int, _KV]]) -> None:
        ends = {}
        for layer, (keys, values) in writes:
            if (
                keys.ndim != 4
                or keys.shape != values.shape
                or keys.shape != pool.shape(keys.shape[2])
            ):
                raise ValueError("keys and values must have shape [1, kv_heads, tokens, head_dim]")
            if keys.shape[2] == 0:
                raise ValueError("KV cache updates require at least one token")
            if not all(mx.issubdtype(array.dtype, mx.floating) for array in (keys, values)):
                raise TypeError("keys and values must be floating-point arrays")
            end = self._offsets[layer] + keys.shape[2]
            if end > pool.max_tokens:
                raise OverflowError(f"KV cache capacity is {pool.max_tokens} tokens, got {end}")
            ends[layer] = end

        extra_blocks = max(0, _blocks_for(max(ends.values())) - len(self._blocks))
        reserved = pool._take_blocks(extra_blocks) if extra_blocks else []
        try:
            table = self._blocks + reserved if reserved else self._blocks
            pending: list[tuple[_Block, int, _KV]] = []
            for layer, (keys, values) in writes:
                keys, values = keys.astype(pool.dtype), values.astype(pool.dtype)
                position = self._offsets[layer]
                while position < ends[layer]:
                    index, within_block = divmod(position, _BLOCK_TOKENS)
                    count = min(_BLOCK_TOKENS - within_block, ends[layer] - position)
                    start = position - self._offsets[layer]
                    block = table[index]
                    new_keys = keys[:, :, start:start + count, :]
                    new_values = values[:, :, start:start + count, :]
                    if within_block:
                        old_keys, old_values = block.layers[layer]
                        new_keys = mx.concatenate((old_keys, new_keys), axis=2)
                        new_values = mx.concatenate((old_values, new_values), axis=2)
                    pending.append((block, layer, (new_keys, new_values)))
                    position += count
        except BaseException:
            if reserved:
                pool._return_blocks(reserved)
            raise

        for block, layer, pair in pending:
            block.layers[layer] = pair
        self._blocks = table
        for layer, end in ends.items():
            self._offsets[layer] = end

    def _release(self) -> None:
        with self._lock:
            pool = self._active_pool()
            pool._return_blocks(self._blocks)
            self._blocks = []
            self._pool = None


def _blocks_for(tokens: int) -> int:
    return (tokens + _BLOCK_TOKENS - 1) // _BLOCK_TOKENS


class MLXPagedKVCachePool(KVCachePool[mx.array]):
    """Lease slots and allocate their GPU KV in 16-token blocks as they grow.

    The shared free list can accommodate every slot reaching its token limit.
    Blocks store only written data. Callers can read the blocks directly or
    use fetch to gather them for contiguous attention.
    """

    def __init__(
        self, args: ServerArgs, *, num_layers: int, num_kv_heads: int, head_dim: int
    ) -> None:
        dimensions = {
            "cache_slots": args.cache_slots,
            "cache_max_tokens": args.cache_max_tokens,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }
        for name, value in dimensions.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        dtypes = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
        if args.cache_dtype not in dtypes:
            raise ValueError(f"Unsupported KV cache dtype: {args.cache_dtype!r}")
        if not mx.metal.is_available():
            raise RuntimeError("The mlx_paged_kv cache pool requires an MLX Metal GPU")

        self.dtype = dtypes[args.cache_dtype]
        self.max_tokens = args.cache_max_tokens
        self.num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._capacity = args.cache_slots
        self._leased: dict[int, _PagedRequestKVCache] = {}
        self._free_blocks = [
            _Block() for _ in range(self.capacity * _blocks_for(self.max_tokens))
        ]
        self._slot_lock = Lock()
        self._block_lock = Lock()

    def shape(self, tokens: int) -> tuple[int, int, int, int]:
        return (1, self._num_kv_heads, tokens, self._head_dim)

    @property
    def capacity(self) -> int:
        return self._capacity

    def lease(self, slot: int) -> RequestKVCache[mx.array]:
        with self._slot_lock:
            if not 0 <= slot < self.capacity:
                raise IndexError("KV cache slot index out of range")
            if slot in self._leased:
                raise BufferError("KV cache slot is already leased")
            cache = _PagedRequestKVCache(self)
            self._leased[slot] = cache
            return cache

    def release(self, slot: int, lease: RequestKVCache[mx.array]) -> None:
        with self._slot_lock:
            active = self._leased.get(slot)
            if active is None or active is not lease:
                raise ValueError("KV cache lease is not active for this slot")
            active._release()
            del self._leased[slot]

    def _take_blocks(self, count: int) -> list[_Block]:
        with self._block_lock:
            if count > len(self._free_blocks):
                raise BufferError("No free KV cache blocks")
            blocks = self._free_blocks[-count:]
            del self._free_blocks[-count:]
            return blocks

    def _return_blocks(self, blocks: list[_Block]) -> None:
        for block in blocks:
            block.layers.clear()
        with self._block_lock:
            self._free_blocks.extend(blocks)
