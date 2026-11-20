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
)
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Self
from uuid import uuid4

import zmq.asyncio
from pydantic import TypeAdapter, ValidationError

from zoochecker.types import (
    ChatMessage,
    EmitDelta,
    GeneratedTokens,
    Generation,
    InferenceFailed,
    InvalidPrompt,
    OutputDelta,
    PromptRequest,
    Sampling,
    TokenizedRequest,
)

logger = logging.getLogger(__name__)

_prompt_request = TypeAdapter(PromptRequest)
_tokenized_request = TypeAdapter(TokenizedRequest)
_generation = TypeAdapter(Generation)
_generated_tokens = TypeAdapter(GeneratedTokens)
_output_delta = TypeAdapter(OutputDelta)


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


class _Reply:
    def __init__(self, client: "_IpcClient", request_id: str) -> None:
        self._client = client
        self._request_id = request_id
        self._frames: asyncio.Queue[dict] = asyncio.Queue()
        self._finished = asyncio.Event()

    def push(self, frame: dict) -> None:
        if not self._finished.is_set():
            self._frames.put_nowait(frame)
            if "event" not in frame:
                self._finished.set()

    async def result(self, emit: EmitDelta | None = None) -> object:
        while True:
            response = await self._frames.get()
            if "event" not in response:
                break
            if emit is not None:
                try:
                    delta = _output_delta.validate_python(response["event"])
                except ValidationError as exc:
                    raise InferenceFailed("Invalid IPC stream delta") from exc
                if delta.request_id != self._request_id:
                    raise InferenceFailed("Mismatched stream request ID")
                await emit(delta)
        if response.get("cancelled") is True:
            raise asyncio.CancelledError
        if "error" in response:
            error = response["error"]
            if not isinstance(error, str):
                raise InferenceFailed("Invalid IPC error")
            if response.get("invalid_prompt") is True:
                raise InvalidPrompt(error)
            raise InferenceFailed(error)
        if "result" not in response:
            raise InferenceFailed("Missing IPC result")
        return response["result"]

    async def close(self) -> None:
        try:
            if not self._finished.is_set():
                await self._client.send(
                    {"request_id": self._request_id, "cancel": True}
                )
                await self._finished.wait()
        finally:
            self._client._replies.pop(self._request_id, None)


class _IpcClient:
    def __init__(self, endpoint: str | Path) -> None:
        self._socket = create_async_ipc_pair_socket(endpoint, bind=False)
        self._send_lock = asyncio.Lock()
        self._replies: dict[str, _Reply] = {}
        self._reader: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        self._reader = asyncio.create_task(self._receive())
        return self

    async def __aexit__(self, *exc) -> None:
        try:
            if self._reader is not None:
                self._reader.cancel()
                await asyncio.gather(self._reader, return_exceptions=True)
        finally:
            self._socket.close()

    @property
    def ready(self) -> bool:
        return self._reader is not None and not self._reader.done()

    async def send(self, frame: dict) -> None:
        async with self._send_lock:
            await self._socket.send_json(frame)

    async def _receive(self) -> None:
        try:
            while True:
                frame = await self._socket.recv_json()
                reply = self._replies.get(frame.get("request_id"))
                if reply is not None:
                    reply.push(frame)
        except Exception:
            logger.exception("IPC response reader stopped")
        finally:
            for reply in self._replies.values():
                reply.push({"error": "IPC connection closed"})

    async def _begin(self, requests: list[dict]) -> list[_Reply]:
        if not self.ready:
            raise InferenceFailed("IPC response reader is unavailable")
        ids = [request["request_id"] for request in requests]
        if len(set(ids)) != len(ids) or any(
            request_id in self._replies for request_id in ids
        ):
            raise InferenceFailed("Duplicate IPC request ID")
        replies = [_Reply(self, request_id) for request_id in ids]
        self._replies.update(zip(ids, replies, strict=True))
        try:
            await self.send(requests[0] if len(requests) == 1 else {"batch": requests})
        except asyncio.CancelledError:
            await asyncio.gather(
                *(reply.close() for reply in replies), return_exceptions=True
            )
            raise
        except Exception:
            for request_id in ids:
                self._replies.pop(request_id, None)
            raise
        return replies

    async def _request(
        self, request_id: str, data: dict, emit: EmitDelta | None = None
    ) -> object:
        (reply,) = await self._begin([{**data, "request_id": request_id}])
        try:
            return await reply.result(emit)
        finally:
            await reply.close()


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
        active: dict[str, asyncio.Task[None]] = {}
        async with asyncio.TaskGroup() as requests:
            try:
                while True:
                    message = await socket.recv()
                    try:
                        frame = json.loads(message)
                    except ValueError:
                        frame = None
                    batch = frame.get("batch") if isinstance(frame, dict) else None
                    for data in batch if isinstance(batch, list) else (frame,):
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
                            task = active.get(request_id)
                            if task is not None:
                                task.cancel()
                                await asyncio.gather(task, return_exceptions=True)
                            await socket.send_json(
                                {"request_id": request_id, "cancelled": True}
                            )
                            continue
                        if request_id in active:
                            continue
                        task = requests.create_task(process(socket, request_id, data))
                        active[request_id] = task
                        task.add_done_callback(
                            lambda _, key=request_id: active.pop(key, None)
                        )
            finally:
                for task in active.values():
                    task.cancel()

    with closing(create_async_ipc_pair_socket(endpoint, bind=True)) as socket:
        async with asyncio.TaskGroup() as tasks:
            task = tasks.create_task(serve(socket))
            try:
                yield task
            finally:
                task.cancel()


def _stream_emitter(
    data: dict, sampling: Sampling | None, emit: EmitDelta
) -> EmitDelta | None:
    stream = data.get("stream", False)
    if not isinstance(stream, bool):
        raise InvalidPrompt("Expected a boolean stream flag")
    if stream and sampling is None:
        raise InvalidPrompt("Streaming requires generation settings")
    return emit if stream else None


def serve_tokenizer(
    endpoint: str | Path,
    process: Callable[[PromptRequest, EmitDelta | None], Awaitable[Generation | None]],
) -> AbstractAsyncContextManager[asyncio.Task[None]]:
    async def handle(request_id: str, data: dict, emit: EmitDelta) -> dict | None:
        if ("messages" in data) == ("prompt" in data):
            raise InvalidPrompt("Expected either a text prompt or chat messages")
        if "messages" in data and not isinstance(data["messages"], list):
            raise InvalidPrompt("Expected chat messages")
        if "prompt" in data and not isinstance(data["prompt"], str):
            raise InvalidPrompt("Expected a text prompt")
        try:
            request = _prompt_request.validate_python(
                {
                    "request_id": request_id,
                    "prompt": data.get("messages", data.get("prompt")),
                    "sampling": data.get("sampling"),
                }
            )
        except ValidationError as exc:
            raise InvalidPrompt("Invalid tokenizer request") from exc
        result = await process(request, _stream_emitter(data, request.sampling, emit))
        return asdict(result) if result is not None else None

    return _serve_ipc(endpoint, handle)


def serve_worker(
    endpoint: str | Path,
    run: Callable[
        [TokenizedRequest, EmitDelta | None], Awaitable[GeneratedTokens | None]
    ],
) -> AbstractAsyncContextManager[asyncio.Task[None]]:
    async def handle(request_id: str, data: dict, emit: EmitDelta) -> dict | None:
        try:
            request = _tokenized_request.validate_python(data)
        except ValidationError as exc:
            raise InvalidPrompt("Invalid worker request") from exc
        result = await run(request, _stream_emitter(data, request.sampling, emit))
        return asdict(result) if result is not None else None

    return _serve_ipc(endpoint, handle)


class TokenizerClient(_IpcClient):
    async def submit(self, prompt: str) -> str:
        request_id = uuid4().hex
        if await self._request(request_id, {"prompt": prompt}) is not None:
            raise InferenceFailed("Unexpected IPC prefill result")
        return request_id

    async def generate(
        self,
        sampling: Sampling,
        *,
        prompt: str | None = None,
        messages: list[ChatMessage] | None = None,
        request_id: str | None = None,
        emit: EmitDelta | None = None,
    ) -> Generation:
        if (prompt is None) == (messages is None):
            raise InvalidPrompt("Expected either a text prompt or chat messages")
        data = (
            {"prompt": prompt}
            if messages is None
            else {"messages": [message.model_dump(mode="json") for message in messages]}
        )
        request_id = request_id or uuid4().hex

        terminal: OutputDelta | None = None

        async def event(delta: OutputDelta) -> None:
            nonlocal terminal
            if terminal is not None:
                raise InferenceFailed("Received a delta after the terminal delta")
            if delta.finish_reason is not None:
                terminal = delta
            elif emit is not None:
                await emit(delta)

        result = await self._request(
            request_id,
            {**data, "sampling": asdict(sampling), "stream": emit is not None},
            event if emit is not None else None,
        )
        try:
            generation = _generation.validate_python(result)
        except ValidationError as exc:
            raise InferenceFailed("Invalid IPC generation result") from exc
        if generation.request_id != request_id:
            raise InferenceFailed("Mismatched generation request ID")
        if emit is not None:
            if terminal is None:
                raise InferenceFailed("Stream ended without a terminal delta")
            if (
                terminal.finish_reason != generation.finish_reason
                or terminal.output_token_ids != generation.output_token_ids
                or terminal.prompt_tokens != generation.prompt_tokens
                or terminal.completion_tokens != generation.completion_tokens
            ):
                raise InferenceFailed(
                    "Stream terminal delta differs from generation result"
                )
            await emit(terminal)
        return generation


class WorkerReply:
    def __init__(self, reply: _Reply) -> None:
        self._reply = reply

    async def result(self, emit: EmitDelta | None) -> GeneratedTokens | None:
        result = await self._reply.result(emit)
        if result is None:
            return None
        try:
            return _generated_tokens.validate_python(result)
        except ValidationError as exc:
            raise InferenceFailed("Invalid IPC worker result") from exc

    async def close(self) -> None:
        await self._reply.close()


class WorkerClient(_IpcClient):
    async def submit(
        self, requests: list[tuple[TokenizedRequest, bool]]
    ) -> list[WorkerReply]:
        replies = await self._begin(
            [
                {
                    "request_id": request.request_id,
                    "token_ids": request.token_ids,
                    "sampling": asdict(request.sampling)
                    if request.sampling is not None
                    else None,
                    "vision": asdict(request.vision)
                    if request.vision is not None
                    else None,
                    "stream": streaming,
                }
                for request, streaming in requests
            ]
        )
        return [WorkerReply(reply) for reply in replies]
