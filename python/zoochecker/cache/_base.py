"""Contracts for request KV storage and cross-request prefix reuse."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KVPrefix[ArrayT]:
    """Read-only decoder KV state owning just the captured tokens."""

    prefix_len: int
    layers: tuple[tuple[ArrayT, ArrayT], ...]


class RequestKVCache[ArrayT](ABC):
    """Forward-pass access to one active request's per-layer KV state."""

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Maximum tokens per layer."""
        raise NotImplementedError

    @abstractmethod
    def offset(self, layer: int) -> int:
        """Number of tokens written to the layer."""
        raise NotImplementedError

    @abstractmethod
    def fetch(self, layer: int) -> tuple[ArrayT, ArrayT]:
        """Read the layer's written keys and values as a single array pair."""
        raise NotImplementedError

    @abstractmethod
    def append(self, layer: int, keys: ArrayT, values: ArrayT) -> None:
        """Append keys and values without gathering the existing prefix."""
        raise NotImplementedError

    @abstractmethod
    def blocks(self, layer: int) -> tuple[tuple[ArrayT, ArrayT], ...]:
        """Read ordered KV blocks without gathering, or return () for an empty layer.

        Each block exposes only its written tokens. Dense storage exposes one
        block. The arrays are read-only views and may keep backing storage alive.
        """
        raise NotImplementedError

    @abstractmethod
    def update_and_fetch(
        self, layer: int, keys: ArrayT, values: ArrayT
    ) -> tuple[ArrayT, ArrayT]:
        """Append keys and values and return the layer's new written prefix."""
        raise NotImplementedError

    @abstractmethod
    def snapshot(self, prefix_len: int) -> KVPrefix[ArrayT]:
        """Capture a positive-length prefix written to every layer.

        The snapshot stays valid across subsequent appends and slot release.
        """
        raise NotImplementedError

    @abstractmethod
    def restore(self, state: KVPrefix[ArrayT]) -> None:
        """Load a compatible snapshot into a fresh lease at offset zero.

        The restored state remains valid after the source snapshot is released.
        """
        raise NotImplementedError


class KVCachePool[ArrayT](ABC):
    """Backend storage for slots selected by the CPU-side cache manager."""

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Number of addressable request slots."""
        raise NotImplementedError

    @abstractmethod
    def lease(self, slot: int) -> RequestKVCache[ArrayT]:
        """Open a fresh request handle for a free slot."""
        raise NotImplementedError

    @abstractmethod
    def release(self, slot: int, lease: RequestKVCache[ArrayT]) -> None:
        """Invalidate the active handle for this slot and reset the slot."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PrefixMatch[StateT]:
    """A positive-length prefix and its read-only model-state snapshot."""

    prefix_len: int
    state: StateT


class PrefixCache[StateT](ABC):
    """Index reusable model state by token prefix.

    Each instance is scoped to one model execution context. Requests that can
    produce different model state from the same token IDs, such as requests
    using different adapters or media embeddings, require separate scopes.

    StateT is a backend-defined snapshot or handle containing the state
    required to resume at one token boundary. The worker restores it into
    request-owned state before appending tokens. For a KV-only model, use
    KVPrefix as StateT and RequestKVCache.snapshot and restore at the boundary.
    """

    @abstractmethod
    def match(
        self, token_ids: Sequence[int]
    ) -> AbstractContextManager[PrefixMatch[StateT] | None]:
        """Acquire the longest prefix with usable state, or yield None on a miss.

        All state required by the model must be valid at the returned boundary.
        The cache keeps the snapshot usable until the context exits. Before
        exiting, the caller must finish reading it or take independent ownership
        of any storage used by later work.

        Pass only the tokens eligible for reuse. For example, pass
        token_ids[:-1] when the final token must run to produce logits.
        """
        raise NotImplementedError

    @abstractmethod
    def insert(self, token_ids: Sequence[int], state: StateT) -> bool:
        """Publish state at the boundary len(token_ids), which must be positive.

        State must represent every supplied token. On success, the cache
        retains a snapshot that remains valid across request mutation and slot
        release. Return True if the boundary is reusable, including an existing
        entry. Return False if the cache cannot admit it, such as when capacity
        is pinned by active matches.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from lookup immediately.

        Release unpinned storage immediately and pinned storage after its last
        match exits.
        """
        raise NotImplementedError
