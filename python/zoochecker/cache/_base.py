from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True, slots=True)
class KVPrefix:
    prefix_len: int
    layers: tuple[tuple[mx.array, mx.array], ...]


@dataclass(frozen=True, slots=True)
class KVCacheStats:
    slots_total: int
    slots_used: int
    blocks_total: int
    blocks_used: int
    block_tokens: int
    bytes_total: int
    bytes_used: int


class RequestKVCache(ABC):
    @property
    @abstractmethod
    def capacity(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def offset(self, layer: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def fetch(self, layer: int) -> tuple[mx.array, mx.array]:
        raise NotImplementedError

    @abstractmethod
    def append(self, layer: int, keys: mx.array, values: mx.array) -> None:
        raise NotImplementedError

    @abstractmethod
    def blocks(self, layer: int) -> tuple[tuple[mx.array, mx.array], ...]:
        raise NotImplementedError

    @abstractmethod
    def update_and_fetch(
        self, layer: int, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self, prefix_len: int) -> KVPrefix:
        raise NotImplementedError

    @abstractmethod
    def restore(self, state: KVPrefix) -> None:
        raise NotImplementedError


class KVCachePool(ABC):
    @property
    @abstractmethod
    def capacity(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> KVCacheStats:
        raise NotImplementedError

    @abstractmethod
    def lease(self, slot: int) -> RequestKVCache:
        raise NotImplementedError

    @abstractmethod
    def release(self, slot: int, lease: RequestKVCache) -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    prefix_len: int
    state: KVPrefix


class PrefixCache(ABC):
    @abstractmethod
    def match(
        self, token_ids: Sequence[int]
    ) -> AbstractContextManager[PrefixMatch | None]:
        raise NotImplementedError

    @abstractmethod
    def insert(self, token_ids: Sequence[int], state: KVPrefix) -> bool:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError
