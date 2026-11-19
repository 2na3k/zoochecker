"""Interfaces for model-state components used by the prefix cache."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Any
from dataclasses import dataclass



class BlockPointer(int):
    """
    Kind of lightweight pointer to the block in the memory
    with the hash something something
    """
    pass


@dataclass
class Block:
    """
    Actual block interface for that thing

    One question: using block hash then are we going to match the hash
    That's not the spirit right?
    """
    block_hash: int
    pointer: Optional[BlockPointer] = None



class Cache(ABC):
    """Contract for one reusable cache-state component.

    A radix-cache coordinator can combine components (for example, full
    attention K/V and sliding-window K/V) while asking each one whether a
    candidate prefix is safe to reuse. ``state`` is intentionally opaque:
    model/backend-specific cache objects stay out of this interface.
    """

    @property
    @abstractmethod
    def component_id(self) -> str:
        """Stable name for this state component, e.g. ``"full_kv"``."""
        raise NotImplementedError

    @abstractmethod
    def can_reuse(self, tokens: Sequence[int], prefix_len: int) -> bool:
        """Return whether this component is valid at the candidate prefix."""
        raise NotImplementedError

    @abstractmethod
    def load_state(self, tokens: Sequence[int], prefix_len: int) -> Any:
        """Return cached state for a prefix accepted by :meth:`can_reuse`."""
        raise NotImplementedError

    @abstractmethod
    def store_state(
        self, tokens: Sequence[int], prefix_len: int, state: Any
    ) -> None:
        """Store state representing the first ``prefix_len`` tokens."""
        raise NotImplementedError

    @abstractmethod
    def fork_state(self, state: Any) -> Any:
        """Make state safe for request-local mutation (copy-on-write is fine)."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Release all state owned by this component."""
        raise NotImplementedError
