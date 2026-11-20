from collections.abc import Awaitable, Callable

from zoochecker.cache import CacheManager, RequestKVCache
from zoochecker.types import TokenizedRequest

Prefill = Callable[[TokenizedRequest, RequestKVCache], Awaitable[None]]


class Scheduler:
    def __init__(self, cache_manager: CacheManager, prefill: Prefill) -> None:
        self._cache_manager = cache_manager
        self._prefill = prefill

    async def prefill(self, request: TokenizedRequest) -> None:
        with self._cache_manager.request_cache(
            prompt_tokens=len(request.token_ids)
        ) as cache:
            await self._prefill(request, cache)
