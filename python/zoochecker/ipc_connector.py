"""Small ZeroMQ helpers for local, bidirectional IPC, i guess that's gonna be funny"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,
    closing,
    contextmanager,
    suppress,
)
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import zmq.asyncio

from zoochecker.types import (
    EmitDelta,
    GeneratedTokens,
    Generation,
    InferenceFailed,
    InvalidPrompt,
    OutputDelta,
    PromptRequest,
    Sampling,
    TokenizedRequest,
    VisionPrompt,
)

logger = logging.getLogger(__name__)


@contextmanager
def ipc_endpoints() -> Iterator[tuple[Path, Path]]:
    with TemporaryDirectory(prefix="zc-") as directory:
        root = Path(directory)
        yield root / "tokenizer.sock", root / "worker.sock"


def _ipc_address(endpoint: str | Path) -> str:
    if isinstance(endpoint, Path) or "://" not in str(endpoint):
        return f"ipc://{Path(endpoint).expanduser().resolve()}"
    if not endpoint.startswith("ipc://"):
        raise ValueError("endpoint must be a filesystem path or ipc:// URL")
    name = endpoint.removeprefix("ipc://")
    if not name or name == "*" or name.startswith("@"):
        return endpoint
    return f"ipc://{Path(name).expanduser().resolve()}"


def create_async_ipc_pair_socket(
    endpoint: str | Path, *, bind: bool
) -> zmq.asyncio.Socket:
    address = _ipc_address(endpoint)
    name = address.removeprefix("ipc://")
    if name.startswith("/"):
        Path(name).parent.mkdir(parents=True, exist_ok=True)
    socket = zmq.asyncio.Context.instance().socket(zmq.PAIR)
    try:
        socket.setsockopt(zmq.LINGER, 0)
        if bind:
            socket.bind(address)
        else:
            socket.connect(address)
    except Exception:
        socket.close(linger=0)
        raise
    return socket


class _IpcClient:
    def __init__(self, endpoint: str | Path) -> None:
        self._socket = create_async_ipc_pair_socket(endpoint, bind=False)
        self._lock = asyncio.Lock()

    async def _request(
        self, request_id: str, data: dict, emit: EmitDelta | None = None
    ) -> dict | None:
        async with self._lock:
            try:
                await self._socket.send_json({**data, "request_id": request_id})
                while True:
                    response = await self._socket.recv_json()
                    if response.get("request_id") != request_id:
                        continue
                    if "event" in response:
                        if emit is not None:
                            delta = OutputDelta(**response["event"])
                            if delta.request_id != request_id:
                                raise InferenceFailed("Mismatched stream request ID")
                            await emit(delta)
                        continue
                    break
            except asyncio.CancelledError:
                with suppress(zmq.ZMQError):
                    await self._socket.send_json(
                        {"request_id": request_id, "cancel": True}, flags=zmq.DONTWAIT
                    )
                raise

        if response.get("invalid_prompt"):
            raise InvalidPrompt(response["error"])
        if "error" in response:
            raise InferenceFailed(response["error"])
        return response["result"]

    def close(self) -> None:
        self._socket.close()


@asynccontextmanager
async def _serve_ipc(
    endpoint: str | Path,
    handle: Callable[[str, dict, EmitDelta], Awaitable[dict | None]],
) -> AsyncIterator[asyncio.Task[None]]:
    async def process(socket: zmq.asyncio.Socket, request_id: str, data: dict) -> None:
        async def emit(delta: OutputDelta) -> None:
            await socket.send_json({"request_id": request_id, "event": asdict(delta)})

        try:
            response = {"result": await handle(request_id, data, emit)}
        except InvalidPrompt as exc:
            response = {"error": str(exc), "invalid_prompt": True}
        except InferenceFailed as exc:
            response = {"error": str(exc)}
        except Exception:
            logger.exception("Failed to process request %s on %s", request_id, endpoint)
            response = {"error": "Inference failed"}
        await socket.send_json({"request_id": request_id, **response})

    async def serve(socket: zmq.asyncio.Socket) -> None:
        active = None
        active_id = None
        async with asyncio.TaskGroup() as requests:
            try:
                while True:
                    message = await socket.recv()
                    try:
                        data = json.loads(message)
                    except ValueError:
                        data = None
                    request_id = (
                        data.get("request_id") if isinstance(data, dict) else None
                    )
                    if not isinstance(request_id, str) or not request_id:
                        await socket.send_json(
                            {
                                "request_id": None,
                                "error": "Expected a request ID",
                                "invalid_prompt": True,
                            }
                        )
                        continue
                    if data.get("cancel") is True:
                        if active is not None and request_id == active_id:
                            active.cancel()
                            await asyncio.gather(active, return_exceptions=True)
                        continue
                    if active is not None and not active.done():
                        await asyncio.gather(active, return_exceptions=True)
                    active_id = request_id
                    active = requests.create_task(process(socket, request_id, data))
            finally:
                if active is not None:
                    active.cancel()

    with closing(create_async_ipc_pair_socket(endpoint, bind=True)) as socket:
        async with asyncio.TaskGroup() as tasks:
            task = tasks.create_task(serve(socket))
            try:
                yield task
            finally:
                task.cancel()


def serve_tokenizer(
    endpoint: str | Path,
    process: Callable[[PromptRequest, EmitDelta | None], Awaitable[Generation | None]],
) -> AbstractAsyncContextManager[asyncio.Task[None]]:
    async def handle(request_id: str, data: dict, emit: EmitDelta) -> dict | None:
        prompt = data.get("messages", data.get("prompt"))
        if not isinstance(prompt, (str, list)):
            raise InvalidPrompt("Expected a text prompt or chat messages")
        sampling = Sampling(**data["sampling"]) if "sampling" in data else None

        result = await process(
            PromptRequest(request_id, prompt, sampling),
            emit if data.get("stream") else None,
        )
        return asdict(result) if result is not None else None

    return _serve_ipc(endpoint, handle)


def serve_worker(
    endpoint: str | Path,
    run: Callable[
        [TokenizedRequest, EmitDelta | None], Awaitable[GeneratedTokens | None]
    ],
) -> AbstractAsyncContextManager[asyncio.Task[None]]:
    async def handle(request_id: str, data: dict, emit: EmitDelta) -> dict | None:
        token_ids = data.get("token_ids")
        if not isinstance(token_ids, list) or any(
            type(token) is not int or token < 0 for token in token_ids
        ):
            raise InvalidPrompt("Expected nonnegative token IDs")
        sampling = data.get("sampling")
        vision = data.get("vision")
        request = TokenizedRequest(
            request_id,
            tuple(token_ids),
            Sampling(**sampling) if sampling is not None else None,
            VisionPrompt(vision["text"], tuple(vision["images"]))
            if vision is not None
            else None,
        )

        result = await run(request, emit if data.get("stream") else None)
        return asdict(result) if result is not None else None

    return _serve_ipc(endpoint, handle)


class TokenizerClient(_IpcClient):
    async def submit(self, prompt: str) -> str:
        request_id = uuid4().hex
        await self._request(request_id, {"prompt": prompt})
        return request_id

    async def generate(
        self,
        sampling: Sampling,
        *,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        request_id: str | None = None,
        emit: EmitDelta | None = None,
    ) -> Generation:
        data = {"prompt": prompt} if messages is None else {"messages": messages}

        terminal = None

        async def event(delta: OutputDelta) -> None:
            nonlocal terminal
            if delta.finish_reason is not None:
                terminal = delta
            elif emit is not None:
                await emit(delta)

        result = await self._request(
            request_id or uuid4().hex,
            {**data, "sampling": asdict(sampling), "stream": emit is not None},
            event if emit is not None else None,
        )
        assert result is not None
        if emit is not None:
            if terminal is None:
                raise InferenceFailed("Stream ended without a terminal delta")
            await emit(terminal)
        return Generation(**result)


class WorkerClient(_IpcClient):
    async def run(
        self, request: TokenizedRequest, emit: EmitDelta | None = None
    ) -> GeneratedTokens | None:
        result = await self._request(
            request.request_id,
            {
                "token_ids": request.token_ids,
                "sampling": asdict(request.sampling)
                if request.sampling is not None
                else None,
                "vision": asdict(request.vision)
                if request.vision is not None
                else None,
                "stream": emit is not None,
            },
            emit,
        )
        if result is None:
            return None
        return GeneratedTokens(
            tuple(result["token_ids"]),
            result["finish_reason"],
            result.get("prompt_tokens"),
        )
