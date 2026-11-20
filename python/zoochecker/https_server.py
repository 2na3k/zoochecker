"""Application initialization."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager, closing
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from zoochecker.api.v1.service import create_router
from zoochecker.api.v1.status import record_http, router as status_router
from zoochecker.cache import UnifiedCacheManager
from zoochecker.ipc_connector import (
    TokenizerClient,
    WorkerClient,
    ipc_endpoints,
    serve_tokenizer,
    serve_worker,
)
from zoochecker.metrics import Metrics
from zoochecker.scheduler import Scheduler
from zoochecker.server_args import ServerArgs
from zoochecker.tokenizer import Encode, Tokenizer
from zoochecker.types import InferenceFailed, InvalidPrompt

if TYPE_CHECKING:
    from zoochecker.worker.runner import Prefill


class PrefillRequest(BaseModel):
    prompt: str


def create_app(
    args: ServerArgs,
    *,
    num_layers: int | None = None,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    encode: Encode | None = None,
    prefill: "Prefill | None" = None,
) -> FastAPI:
    geometry = (num_layers, num_kv_heads, head_dim)
    use_model = geometry == (None, None, None)
    serve_requests = use_model or encode is not None
    if use_model:
        if encode is not None or prefill is not None:
            raise ValueError("Custom encode and prefill require cache geometry")
        if (
            args.max_new_tokens < 1
            or not 0 <= args.temperature <= 2
            or not 0 < args.top_p <= 1
        ):
            raise ValueError("Invalid default generation settings")
    else:
        if None in geometry:
            raise ValueError("The cache requires all three geometry values")
        if (encode is None) != (prefill is None):
            raise ValueError("encode and prefill must be provided together")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            metrics = Metrics(args) if args.telemetry_enabled else None
            if metrics is not None:
                await stack.enter_async_context(metrics.running())
            layers, heads, width = geometry
            worker = codec = None
            if use_model:
                from zoochecker.worker.runner import Worker

                worker, codec = Worker.load(args, metrics)
                layers, heads, width = worker.cache_geometry
                app.state.image_size = worker.image_size
                stack.callback(delattr, app.state, "image_size")
            assert layers is not None and heads is not None and width is not None
            manager = UnifiedCacheManager(
                args,
                num_layers=layers,
                num_kv_heads=heads,
                head_dim=width,
            )
            stack.callback(manager.close)
            app.state.cache_manager = manager
            stack.callback(delattr, app.state, "cache_manager")
            if serve_requests:
                from zoochecker.worker.runner import WorkerService

                async def run_prefill(request, lease, emit):
                    assert prefill is not None
                    return await prefill(request, lease)

                run_worker = worker.run if worker is not None else run_prefill
                tokenizer_endpoint, worker_endpoint = stack.enter_context(
                    ipc_endpoints()
                )
                service = WorkerService(manager, run_worker)
                worker_task = await stack.enter_async_context(
                    serve_worker(worker_endpoint, service.run)
                )
                worker_client = stack.enter_context(
                    closing(WorkerClient(worker_endpoint))
                )
                scheduler = Scheduler(worker_client.run, metrics)
                await stack.enter_async_context(scheduler.running())
                if worker is not None and codec is not None:
                    tokenizer = Tokenizer.from_codec(args, codec, scheduler, metrics)
                else:
                    assert encode is not None
                    tokenizer = Tokenizer(encode, scheduler, metrics=metrics)
                tokenizer_task = await stack.enter_async_context(
                    serve_tokenizer(tokenizer_endpoint, tokenizer.process)
                )
                client = TokenizerClient(tokenizer_endpoint)
                stack.callback(client.close)
                app.state.tokenizer_client = client
                stack.callback(delattr, app.state, "tokenizer_client")
                app.state.scheduler = scheduler
                app.state.ipc_tasks = (tokenizer_task, worker_task)
                stack.callback(delattr, app.state, "scheduler")
                stack.callback(delattr, app.state, "ipc_tasks")
            app.state.metrics = metrics
            stack.callback(delattr, app.state, "metrics")
            yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(status_router)
    app.middleware("http")(record_http)
    if use_model:
        app.include_router(create_router(args))
    if serve_requests:

        @app.post("/v1/prefill")
        async def prefill_prompt(
            body: PrefillRequest, request: Request
        ) -> dict[str, str]:
            client: TokenizerClient = request.app.state.tokenizer_client
            try:
                request_id = await client.submit(body.prompt)
            except InvalidPrompt as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except InferenceFailed as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return {"request_id": request_id}

    return app
