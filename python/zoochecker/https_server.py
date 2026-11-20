"""Application initialization."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from zoochecker.cache import UnifiedCacheManager
from zoochecker.scheduler import Prefill, Scheduler
from zoochecker.server_args import ServerArgs
from zoochecker.tokenizer import Encode, PrefillFailed, Tokenizer, TokenizerClient


class PrefillRequest(BaseModel):
    prompt: str


def create_app(
    args: ServerArgs,
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    encode: Encode | None = None,
    prefill: Prefill | None = None,
) -> FastAPI:
    endpoint = args.tokenizer_ipc_endpoint
    if not (endpoint is None and encode is None and prefill is None) and (
        endpoint is None or encode is None or prefill is None
    ):
        raise ValueError(
            "tokenizer_ipc_endpoint, encode and prefill must be provided together"
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            manager = UnifiedCacheManager(
                args,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
            )
            stack.callback(manager.close)
            app.state.cache_manager = manager
            stack.callback(delattr, app.state, "cache_manager")
            if endpoint is not None and encode is not None and prefill is not None:
                tokenizer = Tokenizer(encode, Scheduler(manager, prefill))
                await stack.enter_async_context(tokenizer.running(endpoint))
                client = TokenizerClient(endpoint)
                stack.callback(client.close)
                app.state.tokenizer_client = client
                stack.callback(delattr, app.state, "tokenizer_client")
            yield

    app = FastAPI(lifespan=lifespan)
    if endpoint is not None:

        @app.post("/v1/prefill")
        async def prefill_prompt(
            body: PrefillRequest, request: Request
        ) -> dict[str, str]:
            client: TokenizerClient = request.app.state.tokenizer_client
            try:
                request_id = await client.submit(body.prompt)
            except PrefillFailed as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return {"request_id": request_id}

    return app
