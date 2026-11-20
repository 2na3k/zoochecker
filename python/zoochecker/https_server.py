"""Application initialization."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from zoochecker.cache import UnifiedCacheManager
from zoochecker.server_args import ServerArgs


def create_app(
    args: ServerArgs, *, num_layers: int, num_kv_heads: int, head_dim: int
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        manager = UnifiedCacheManager(
            args, num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        app.state.cache_manager = manager
        try:
            yield
        finally:
            manager.close()
            del app.state.cache_manager

    return FastAPI(lifespan=lifespan)
