from abc import ABC, abstractmethod
from typing import Any

import mlx.core as mx

from zoochecker.cache import RequestKVCache
from zoochecker.types import InvalidPrompt, TokenizedRequest


class BaseModel(ABC):
    """
    Separate from the interface of the engine, mostly
    going to be the process being put in the things
    """

    @property
    @abstractmethod
    def cache_geometry(self) -> tuple[int, int, int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def context_length(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def make_cache(self, lease: RequestKVCache) -> Any:
        raise NotImplementedError

    async def prepare_prompt(
        self, request: TokenizedRequest, cache: Any
    ) -> tuple[int, ...]:
        if request.vision is not None:
            raise InvalidPrompt("This model does not support images")
        return request.token_ids

    def image_size(self, width: int, height: int) -> tuple[int, int]:
        raise InvalidPrompt("This model does not support images")

    @abstractmethod
    def forward(self, tokens: mx.array, cache: Any, *, logits: bool = True) -> mx.array:
        """
        Technically going to modify those things right in this to modify the
        tensor x to the output

        input x: tensor, output tensor output

        this would be a GPU bounded process so the thing must be separate from
        the engine, the engine would handle the non-GPU works (tokenizer, remap output, etc)
        """
        raise NotImplementedError("Forward method must be implemented!!")
