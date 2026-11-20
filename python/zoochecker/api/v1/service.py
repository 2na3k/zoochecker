# would be the single file to register all of the inference class for this thing
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial
from time import monotonic, time
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.background import BackgroundTask

from zoochecker.cache import UnifiedCacheManager
from zoochecker.images import prepare_chat_images
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
from zoochecker.types import (
    AssistantMessage,
    ChatChunkChoice,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatDelta,
    ChatMessage,
    CompletionChoice,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionRequest,
    CompletionResponse,
    ErrorInfo,
    ErrorResponse,
    FinishReason,
    Generation,
    HealthResponse,
    InferenceFailed,
    InvalidPrompt,
    ModelInfo,
    ModelList,
    OutputDelta,
    PrefillRequest,
    PrefillResponse,
    Sampling,
    StatsResponse,
    TelemetryDisabledResponse,
    TelemetryEvent,
    UnavailableResponse,
    UsageInfo,
)

if TYPE_CHECKING:
    from zoochecker.worker.runner import Prefill


def _error(
    status: int, message: str, error_type: str, *, param: str | None = None
) -> JSONResponse:
    response = ErrorResponse(
        error=ErrorInfo(message=message, type=error_type, param=param)
    )
    return JSONResponse(
        status_code=status,
        content=response.model_dump(),
    )


async def _validation_error(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    issue = error.errors()[0]
    location = issue["loc"]
    if location and location[0] in ("body", "query", "path"):
        location = location[1:]
    param = ".".join(str(part) for part in location) or None
    return _error(422, issue["msg"], "invalid_request_error", param=param)


def _usage(result: Generation | OutputDelta) -> UsageInfo:
    return UsageInfo(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.prompt_tokens + result.completion_tokens,
    )


def _sse(data: ChatCompletionChunk | CompletionChunk | ErrorResponse) -> str:
    return f"data: {data.model_dump_json(exclude_unset=True)}\n\n"


StreamItem = OutputDelta | InvalidPrompt | InferenceFailed


async def _wait_for_first_item(
    request: Request, queue: asyncio.Queue[StreamItem]
) -> StreamItem | None:
    async def wait_for_disconnect() -> None:
        while (await request.receive())["type"] != "http.disconnect":
            pass

    disconnected = asyncio.create_task(wait_for_disconnect())
    first_frame = asyncio.create_task(queue.get())
    try:
        done, _ = await asyncio.wait(
            (disconnected, first_frame), return_when=asyncio.FIRST_COMPLETED
        )
        return None if disconnected in done else first_frame.result()
    finally:
        disconnected.cancel()
        first_frame.cancel()
        await asyncio.gather(disconnected, first_frame, return_exceptions=True)


async def _stream_events(
    first: OutputDelta,
    queue: asyncio.Queue[StreamItem],
    close: Callable[[], Awaitable[None]],
    *,
    response_id: str,
    created: int,
    model: str,
    chat: bool,
    include_usage: bool,
) -> AsyncIterator[str]:
    def chunk(
        content: str = "",
        reason: FinishReason | None = None,
        *,
        role: bool = False,
        usage: UsageInfo | None = None,
        output_token_ids: tuple[int, ...] | None = None,
    ) -> str:
        if chat:
            delta = ChatDelta(role="assistant") if role else ChatDelta()
            if content:
                delta.content = content
            frame = ChatCompletionChunk(
                id=response_id,
                object="chat.completion.chunk",
                created=created,
                model=model,
                choices=[]
                if usage is not None
                else [ChatChunkChoice(index=0, delta=delta, finish_reason=reason)],
            )
        else:
            frame = CompletionChunk(
                id=response_id,
                object="text_completion",
                created=created,
                model=model,
                choices=[]
                if usage is not None
                else [
                    CompletionChunkChoice(
                        index=0, text=content, logprobs=None, finish_reason=reason
                    )
                ],
            )
        if include_usage:
            frame.usage = usage
        if output_token_ids is not None:
            frame.output_token_ids = output_token_ids
        return _sse(frame)

    try:
        if chat:
            yield chunk(role=True)
        item: StreamItem = first
        while isinstance(item, OutputDelta) and item.finish_reason is None:
            yield chunk(item.token_text)
            item = await queue.get()
        if isinstance(item, OutputDelta):
            if item.token_text:
                yield chunk(item.token_text)
            yield chunk(
                reason=item.finish_reason, output_token_ids=item.output_token_ids
            )
            if include_usage:
                yield chunk(usage=_usage(item))
        else:
            error_type = (
                "invalid_request_error"
                if isinstance(item, InvalidPrompt)
                else "server_error"
            )
            yield _sse(
                ErrorResponse(error=ErrorInfo(message=str(item), type=error_type))
            )
        yield "data: [DONE]\n\n"
    finally:
        await close()


async def _stream_completion(
    request: Request,
    sampling: Sampling,
    model: str,
    *,
    include_usage: bool,
    prompt: str | None = None,
    messages: list[ChatMessage] | None = None,
) -> Response:
    request_id = uuid4().hex
    created = int(time())
    chat = messages is not None
    queue: asyncio.Queue[StreamItem] = asyncio.Queue(maxsize=1)
    client: TokenizerClient = request.app.state.tokenizer_client

    async def produce() -> None:
        try:
            await client.generate(
                sampling,
                prompt=prompt,
                messages=messages,
                request_id=request_id,
                emit=queue.put,
            )
        except (InvalidPrompt, InferenceFailed) as exc:
            await queue.put(exc)
        except Exception:
            await queue.put(InferenceFailed("Inference failed"))

    task = asyncio.create_task(produce())

    async def close() -> None:
        if not task.done() and not task.cancelling():
            task.cancel()
        await asyncio.shield(asyncio.gather(task, return_exceptions=True))

    try:
        first = await _wait_for_first_item(request, queue)
    except BaseException:
        await close()
        raise
    if first is None:
        await close()
        return Response(status_code=499)
    if isinstance(first, (InvalidPrompt, InferenceFailed)):
        await close()
        return _inference_error(first)

    return StreamingResponse(
        _stream_events(
            first,
            queue,
            close,
            response_id=f"{'chatcmpl' if chat else 'cmpl'}-{request_id}",
            created=created,
            model=model,
            chat=chat,
            include_usage=include_usage,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        background=BackgroundTask(close),
    )


def _inference_error(error: InvalidPrompt | InferenceFailed) -> JSONResponse:
    if isinstance(error, InvalidPrompt):
        return _error(400, str(error), "invalid_request_error")
    return _error(500, str(error), "server_error")


def _sampling(
    body: CompletionRequest | ChatCompletionRequest,
    args: ServerArgs,
    max_tokens: int | None,
) -> Sampling:
    return Sampling(
        args.max_new_tokens if max_tokens is None else max_tokens,
        args.temperature if body.temperature is None else body.temperature,
        args.top_p if body.top_p is None else body.top_p,
    )


def _model_error(model: str | None, args: ServerArgs) -> JSONResponse | None:
    if model is not None and model != args.model_name:
        return _error(404, f"Model {model!r} is unavailable", "invalid_request_error")
    return None


async def list_models(request: Request) -> ModelList:
    args: ServerArgs = request.app.state.server_args
    return ModelList(data=[ModelInfo(id=args.model_name)])


async def completion(
    body: CompletionRequest, request: Request
) -> CompletionResponse | Response:
    args: ServerArgs = request.app.state.server_args
    if error := _model_error(body.model, args):
        return error
    sampling = _sampling(body, args, body.max_tokens)
    if body.stream:
        return await _stream_completion(
            request,
            sampling,
            args.model_name,
            prompt=body.prompt,
            include_usage=body.stream_options is not None
            and body.stream_options.include_usage,
        )

    client: TokenizerClient = request.app.state.tokenizer_client
    try:
        result = await client.generate(sampling, prompt=body.prompt)
    except (InvalidPrompt, InferenceFailed) as exc:
        return _inference_error(exc)
    return CompletionResponse(
        id=f"cmpl-{result.request_id}",
        object="text_completion",
        created=int(time()),
        model=args.model_name,
        choices=[
            CompletionChoice(
                index=0,
                text=result.text,
                finish_reason=result.finish_reason,
                logprobs=None,
            )
        ],
        usage=_usage(result),
        output_token_ids=result.output_token_ids,
    )


async def chat_completion(
    body: ChatCompletionRequest, request: Request
) -> ChatCompletionResponse | Response:
    args: ServerArgs = request.app.state.server_args
    if error := _model_error(body.model, args):
        return error
    max_tokens = (
        body.max_completion_tokens
        if body.max_completion_tokens is not None
        else body.max_tokens
    )
    sampling = _sampling(body, args, max_tokens)
    try:
        messages = await prepare_chat_images(
            body.messages,
            request.app.state.image_size,
            args,
        )
    except InvalidPrompt as exc:
        return _inference_error(exc)
    if body.stream:
        return await _stream_completion(
            request,
            sampling,
            args.model_name,
            messages=messages,
            include_usage=body.stream_options is not None
            and body.stream_options.include_usage,
        )

    client: TokenizerClient = request.app.state.tokenizer_client
    try:
        result = await client.generate(sampling, messages=messages)
    except (InvalidPrompt, InferenceFailed) as exc:
        return _inference_error(exc)
    return ChatCompletionResponse(
        id=f"chatcmpl-{result.request_id}",
        object="chat.completion",
        created=int(time()),
        model=args.model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessage(role="assistant", content=result.text),
                finish_reason=result.finish_reason,
                logprobs=None,
            )
        ],
        usage=_usage(result),
        output_token_ids=result.output_token_ids,
    )


async def prefill_prompt(
    body: PrefillRequest, request: Request
) -> PrefillResponse | Response:
    client: TokenizerClient = request.app.state.tokenizer_client
    try:
        request_id = await client.submit(body.prompt)
    except (InvalidPrompt, InferenceFailed) as exc:
        return _inference_error(exc)
    return PrefillResponse(request_id=request_id)


def _ready(request: Request) -> bool:
    scheduler: Scheduler | None = getattr(request.app.state, "scheduler", None)
    tokenizer_client = getattr(request.app.state, "tokenizer_client", None)
    tasks = getattr(request.app.state, "ipc_tasks", ())
    return (
        scheduler is not None
        and scheduler.ready
        and tokenizer_client is not None
        and tokenizer_client.ready
        and bool(tasks)
        and all(not task.done() for task in tasks)
    )


async def health(request: Request) -> HealthResponse | Response:
    if _ready(request):
        return HealthResponse()
    return JSONResponse(
        UnavailableResponse(detail="worker is not running").model_dump(),
        status_code=503,
    )


def _telemetry_disabled() -> JSONResponse:
    return JSONResponse(TelemetryDisabledResponse().model_dump(), status_code=503)


async def stats(request: Request) -> StatsResponse | Response:
    metrics: Metrics | None = getattr(request.app.state, "metrics", None)
    if metrics is None:
        return _telemetry_disabled()
    await metrics.flush()
    return _snapshot(request, metrics)


async def events(request: Request, limit: int = 200) -> list[TelemetryEvent] | Response:
    metrics: Metrics | None = getattr(request.app.state, "metrics", None)
    if metrics is None:
        return _telemetry_disabled()
    await metrics.flush()
    return [
        TelemetryEvent.model_validate(event)
        for event in metrics.recent_events(max(0, limit))
    ]


async def prometheus_metrics(request: Request) -> Response:
    metrics: Metrics | None = getattr(request.app.state, "metrics", None)
    if metrics is None:
        return _telemetry_disabled()
    await metrics.flush()
    return Response(
        metrics.prometheus(_snapshot(request, metrics)),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


def _snapshot(request: Request, metrics: Metrics) -> StatsResponse:
    scheduler: Scheduler | None = getattr(request.app.state, "scheduler", None)
    waiting, running = scheduler.counts if scheduler is not None else (0, 0)
    cache = request.app.state.cache_manager.stats()
    return metrics.snapshot(waiting, running, cache, ready=_ready(request))


async def record_http(request: Request, call_next) -> Response:
    metrics: Metrics | None = getattr(request.app.state, "metrics", None)
    if metrics is None:
        return await call_next(request)
    started = monotonic()

    def record(status: int) -> None:
        route = request.scope.get("route")
        metrics.record(
            "http_request",
            method=request.method,
            endpoint=getattr(route, "path", "unmatched"),
            status_code=status,
            duration_seconds=monotonic() - started,
        )

    try:
        response = await call_next(request)
    except Exception:
        record(500)
        raise
    body = response.body_iterator

    async def measured_body():
        try:
            async for data in body:
                yield data
        finally:
            record(response.status_code)

    response.body_iterator = measured_body()
    return response


def create_router(
    *, include_completions: bool = True, include_prefill: bool = True
) -> APIRouter:
    router = APIRouter(prefix="/v1", responses={422: {"model": ErrorResponse}})
    disabled = {503: {"model": TelemetryDisabledResponse}}
    router.add_api_route(
        "/health",
        health,
        methods=["GET"],
        response_model=HealthResponse,
        responses={503: {"model": UnavailableResponse}},
    )
    router.add_api_route(
        "/stats",
        stats,
        methods=["GET"],
        response_model=StatsResponse,
        responses=disabled,
    )
    router.add_api_route(
        "/events",
        events,
        methods=["GET"],
        response_model=list[TelemetryEvent],
        responses=disabled,
    )
    router.add_api_route(
        "/metrics",
        prometheus_metrics,
        methods=["GET"],
        response_class=PlainTextResponse,
        responses=disabled,
    )
    if include_prefill:
        router.add_api_route(
            "/prefill",
            prefill_prompt,
            methods=["POST"],
            response_model=PrefillResponse,
            responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
        )
    if include_completions:
        responses = {
            200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}},
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        }
        router.add_api_route(
            "/models", list_models, methods=["GET"], response_model=ModelList
        )
        router.add_api_route(
            "/completions",
            completion,
            methods=["POST"],
            response_model=CompletionResponse,
            response_model_exclude_unset=True,
            responses=responses,
        )
        router.add_api_route(
            "/chat/completions",
            chat_completion,
            methods=["POST"],
            response_model=ChatCompletionResponse,
            response_model_exclude_unset=True,
            responses=responses,
        )
    return router


def _probe_aliases() -> APIRouter:
    router = APIRouter(include_in_schema=False)
    router.add_api_route(
        "/health", health, methods=["GET"], response_model=HealthResponse
    )
    router.add_api_route(
        "/metrics",
        prometheus_metrics,
        methods=["GET"],
        response_class=PlainTextResponse,
    )
    return router


def _set_state(stack: AsyncExitStack, app: FastAPI, **values: object) -> None:
    for name, value in values.items():
        setattr(app.state, name, value)
        stack.callback(delattr, app.state, name)


@asynccontextmanager
async def _lifespan(
    app: FastAPI,
    *,
    geometry: tuple[int | None, int | None, int | None],
    encode: Encode | None,
    prefill: "Prefill | None",
) -> AsyncIterator[None]:
    args: ServerArgs = app.state.server_args
    async with AsyncExitStack() as stack:
        metrics = Metrics(args) if args.telemetry_enabled else None
        if metrics is not None:
            await stack.enter_async_context(metrics.running())

        layers, heads, width = geometry
        worker = codec = None
        if geometry == (None, None, None):
            from zoochecker.worker.runner import Worker

            worker, codec = Worker.load(args, metrics)
            layers, heads, width = worker.cache_geometry
            _set_state(stack, app, image_size=worker.image_size)

        assert layers is not None and heads is not None and width is not None
        manager = UnifiedCacheManager(
            args, num_layers=layers, num_kv_heads=heads, head_dim=width
        )
        stack.callback(manager.close)
        _set_state(stack, app, cache_manager=manager)

        if worker is not None or encode is not None:
            from zoochecker.worker.runner import WorkerService

            async def run_prefill(request, lease, emit):
                assert prefill is not None
                return await prefill(request, lease)

            service = WorkerService(
                manager, worker.run if worker is not None else run_prefill
            )
            tokenizer_endpoint, worker_endpoint = stack.enter_context(ipc_endpoints())
            worker_task = await stack.enter_async_context(
                serve_worker(worker_endpoint, service.run)
            )
            worker_client = await stack.enter_async_context(
                WorkerClient(worker_endpoint)
            )
            scheduler = Scheduler(
                worker_client,
                capacity=args.cache_slots,
                batch_wait_seconds=args.scheduler_batch_wait_seconds,
                metrics=metrics,
            )
            await stack.enter_async_context(scheduler.running())
            if codec is not None:
                tokenizer = Tokenizer.from_codec(args, codec, scheduler, metrics)
            else:
                assert encode is not None
                tokenizer = Tokenizer(encode, scheduler, metrics=metrics)
            tokenizer_task = await stack.enter_async_context(
                serve_tokenizer(tokenizer_endpoint, tokenizer.process)
            )
            client = await stack.enter_async_context(
                TokenizerClient(tokenizer_endpoint)
            )
            _set_state(
                stack,
                app,
                tokenizer_client=client,
                scheduler=scheduler,
                ipc_tasks=(tokenizer_task, worker_task),
            )

        _set_state(stack, app, metrics=metrics)
        yield


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

    app = FastAPI(
        lifespan=partial(_lifespan, geometry=geometry, encode=encode, prefill=prefill)
    )
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.state.server_args = args
    app.include_router(
        create_router(
            include_completions=use_model,
            include_prefill=use_model or encode is not None,
        )
    )
    app.include_router(_probe_aliases())
    app.middleware("http")(record_http)
    return app


def run_server(args: ServerArgs) -> None:
    import uvicorn

    uvicorn.run(create_app(args), host=args.host, port=args.port)
