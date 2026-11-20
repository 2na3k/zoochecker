"""Cache pool creation and CPU-side request slot management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx
    from zoochecker.server_args import ServerArgs

from ._base import KVCachePool, RequestKVCache


class CacheFactory:
    @staticmethod
    def create_pool(
        args: ServerArgs, *, num_layers: int, num_kv_heads: int, head_dim: int
    ) -> KVCachePool[mx.array]:
        if args.cache_pool_type == "mlx_kv":
            from .naive_kv import MLXKVCachePool

            return MLXKVCachePool(
                args, num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim
            )
        if args.cache_pool_type == "mlx_paged_kv":
            from .paged_kv import MLXPagedKVCachePool

            return MLXPagedKVCachePool(
                args, num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim
            )
        raise ValueError(f"Unsupported cache pool type: {args.cache_pool_type!r}")


class CacheManager[ArrayT]:
    """Own request admission and release. Forward passes use the lease directly."""

    def __init__(self, pool: KVCachePool[ArrayT]) -> None:
        self._pool: KVCachePool[ArrayT] | None = pool
        self._free_slots = list(reversed(range(pool.capacity)))
        self._active: dict[RequestKVCache[ArrayT], int] = {}
        self._lock = Lock()

    def allocate(self) -> RequestKVCache[ArrayT]:
        with self._lock:
            if self._pool is None:
                raise RuntimeError("KV cache manager is closed")
            if not self._free_slots:
                raise BufferError("No free KV cache slots")
            slot = self._free_slots[-1]
            cache = self._pool.lease(slot)
            self._active[cache] = slot
            self._free_slots.pop()
            return cache

    def deallocate(self, cache: RequestKVCache[ArrayT]) -> None:
        with self._lock:
            if cache not in self._active:
                raise ValueError("KV cache lease is not active in this manager")
            self._return_slot(cache)

    def _return_slot(self, cache: RequestKVCache[ArrayT]) -> None:
        assert self._pool is not None
        self._pool.release(self._active[cache], cache)
        self._free_slots.append(self._active.pop(cache))

    @contextmanager
    def request_cache(self) -> Iterator[RequestKVCache[ArrayT]]:
        cache = self.allocate()
        try:
            yield cache
        finally:
            with self._lock:
                if cache in self._active:
                    self._return_slot(cache)

    def close(self) -> None:
        """Invalidate open leases and release the pool at application shutdown."""
        with self._lock:
            for cache in tuple(self._active):
                self._return_slot(cache)
            self._free_slots.clear()
            self._pool = None


class UnifiedCacheManager(CacheManager["mx.array"]):
    """Build the configured MLX pool and manage its request slots."""

    def __init__(
        self, args: ServerArgs, *, num_layers: int, num_kv_heads: int, head_dim: int
    ) -> None:
        super().__init__(CacheFactory.create_pool(
            args, num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim
        ))
