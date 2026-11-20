import asyncio
from collections.abc import Awaitable, Callable, Collection
from time import monotonic
from typing import TYPE_CHECKING, Self

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from zoochecker.cache import CacheManager, RequestKVCache
from zoochecker.cache.cache_manager import CacheCapacityError
from zoochecker.models._base import BaseModel
from zoochecker.models.factory import ModelFactory
from zoochecker.server_args import ServerArgs
from zoochecker.types import (
    EmitDelta,
    GeneratedTokens,
    InvalidPrompt,
    OutputDelta,
    TokenizedRequest,
)

if TYPE_CHECKING:
    from zoochecker.metrics import Metrics

WorkerRun = Callable[
    [TokenizedRequest, RequestKVCache, EmitDelta | None],
    Awaitable[GeneratedTokens | None],
]
Prefill = Callable[
    [TokenizedRequest, RequestKVCache], Awaitable[GeneratedTokens | None]
]


class WorkerService:
    def __init__(self, cache_manager: CacheManager, worker: WorkerRun) -> None:
        self._cache_manager = cache_manager
        self._worker = worker

    async def run(
        self, request: TokenizedRequest, emit: EmitDelta | None = None
    ) -> GeneratedTokens | None:
        try:
            with self._cache_manager.request_cache(
                prompt_tokens=len(request.token_ids)
            ) as lease:
                result = await self._worker(request, lease, emit)
        except CacheCapacityError as exc:
            raise InvalidPrompt(str(exc)) from exc
        if emit is not None and result is not None:
            await emit(
                OutputDelta(
                    request_id=request.request_id,
                    token_text="",
                    token_id=-1,
                    finish_reason=result.finish_reason,
                    prompt_tokens=result.prompt_tokens
                    if result.prompt_tokens is not None
                    else len(request.token_ids),
                    completion_tokens=len(result.token_ids),
                )
            )
        return result


class Worker:
    def __init__(
        self,
        args: ServerArgs,
        model: BaseModel,
        eos_tokens: Collection[int],
        metrics: "Metrics | None" = None,
    ) -> None:
        if args.prefill_step_size < 1:
            raise ValueError("prefill_step_size must be positive")
        self._model = model
        self._prefill_step_size = args.prefill_step_size
        self._eos_tokens = frozenset(eos_tokens)
        self._metrics = metrics

    @classmethod
    def load(
        cls, args: ServerArgs, metrics: "Metrics | None" = None
    ) -> tuple[Self, TokenizerWrapper]:
        model, codec = ModelFactory.load(args)
        return cls(args, model, codec.eos_token_ids, metrics), codec

    @property
    def cache_geometry(self) -> tuple[int, int, int]:
        return self._model.cache_geometry

    def image_size(self, width: int, height: int) -> tuple[int, int]:
        return self._model.image_size(width, height)

    async def run(
        self,
        request: TokenizedRequest,
        lease: RequestKVCache,
        emit: EmitDelta | None = None,
    ) -> GeneratedTokens | None:
        cache = self._model.make_cache(lease)
        prompt = await self._model.prepare_prompt(request, cache)
        if not prompt:
            raise InvalidPrompt("Prompt must contain at least one token")
        capacity = min(lease.capacity, self._model.context_length)
        if len(prompt) > capacity:
            raise InvalidPrompt("Prompt exceeds the model context length")

        sampling = request.sampling
        prefill_started = monotonic()
        if self._metrics is not None:
            self._metrics.prompt_prepared(request.request_id, len(prompt))
            self._metrics.record(
                "prefill_start",
                request_id=request.request_id,
                prompt_tokens=len(prompt),
            )
        for start in range(0, len(prompt), self._prefill_step_size):
            end = min(start + self._prefill_step_size, len(prompt))
            scores = self._model.forward(
                mx.array([prompt[start:end]], dtype=mx.int32),
                cache,
                logits=sampling is not None and end == len(prompt),
            )
            await asyncio.sleep(0)

        if self._metrics is not None:
            self._metrics.prefill_finished(
                request.request_id, monotonic() - prefill_started
            )
        if sampling is None:
            return None

        limit = min(sampling.max_tokens, capacity - len(prompt) + 1)
        sampler = make_sampler(temp=sampling.temperature, top_p=sampling.top_p)
        generated = []
        for index in range(limit):
            token = sampler(scores.astype(mx.float32)).item()
            if token in self._eos_tokens:
                return GeneratedTokens(tuple(generated), "stop", len(prompt))
            generated.append(token)
            if self._metrics is not None:
                self._metrics.token_generated(request.request_id)
            if emit is not None:
                await emit(
                    OutputDelta(
                        request_id=request.request_id,
                        token_text="",
                        token_id=token,
                        prompt_tokens=len(prompt),
                        completion_tokens=len(generated),
                    )
                )
            if index + 1 < limit:
                decode_started = monotonic()
                scores = self._model.forward(mx.array([[token]], dtype=mx.int32), cache)
                if self._metrics is not None:
                    self._metrics.record(
                        "decode_step",
                        request_id=request.request_id,
                        duration_seconds=monotonic() - decode_started,
                    )
                await asyncio.sleep(0)
        return GeneratedTokens(tuple(generated), "length", len(prompt))
