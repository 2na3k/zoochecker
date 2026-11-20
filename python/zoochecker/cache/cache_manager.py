from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock

from zoochecker.server_args import ServerArgs

from ._base import KVCachePool, KVCacheStats, RequestKVCache


class CacheCapacityError(ValueError):
    pass


class CacheFactory:
    @staticmethod
    def create_pool(
        args: ServerArgs, *, num_layers: int, num_kv_heads: int, head_dim: int
    ) -> KVCachePool:
        if args.cache_pool_type == "mlx_kv":
            from .naive_kv import MLXKVCachePool

            return MLXKVCachePool(
                args,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
        if args.cache_pool_type == "mlx_paged_kv":
            from .paged_kv import MLXPagedKVCachePool

            return MLXPagedKVCachePool(
                args,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
        raise ValueError(f"Unsupported cache pool type: {args.cache_pool_type!r}")


class CacheManager:
    def __init__(self, pool: KVCachePool) -> None:
        self._pool: KVCachePool | None = pool
        self._free_slots = list(reversed(range(pool.capacity)))
        self._active: dict[RequestKVCache, int] = {}
        self._lock = Lock()

    def stats(self) -> KVCacheStats:
        with self._lock:
            if self._pool is None:
                raise RuntimeError("KV cache manager is closed")
            return self._pool.stats()

    def allocate(self) -> RequestKVCache:
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

    def deallocate(self, cache: RequestKVCache) -> None:
        with self._lock:
            if cache not in self._active:
                raise ValueError("KV cache lease is not active in this manager")
            self._return_slot(cache)

    def _return_slot(self, cache: RequestKVCache) -> None:
        assert self._pool is not None
        self._pool.release(self._active[cache], cache)
        self._free_slots.append(self._active.pop(cache))

    @contextmanager
    def request_cache(self, *, prompt_tokens: int = 0) -> Iterator[RequestKVCache]:
        cache = self.allocate()
        try:
            if prompt_tokens > cache.capacity:
                raise CacheCapacityError("Prompt exceeds KV cache capacity")
            yield cache
        finally:
            with self._lock:
                if cache in self._active:
                    self._return_slot(cache)

    def close(self) -> None:
        with self._lock:
            for cache in tuple(self._active):
                self._return_slot(cache)
            self._free_slots.clear()
            self._pool = None


class UnifiedCacheManager(CacheManager):
    def __init__(
        self, args: ServerArgs, *, num_layers: int, num_kv_heads: int, head_dim: int
    ) -> None:
        super().__init__(
            CacheFactory.create_pool(
                args,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
        )
