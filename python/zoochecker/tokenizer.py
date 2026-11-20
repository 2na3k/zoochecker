"""handle the tokenizer and stuffs, tokenizer then push to scheduler and take from worker to push the result back to the http"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager, closing
from uuid import uuid4

import zmq.asyncio

from zoochecker.ipc_connector import create_async_ipc_pair_socket
from zoochecker.scheduler import Scheduler
from zoochecker.types import TokenizedRequest

Encode = Callable[[str], Sequence[int]]
logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self, encode: Encode, scheduler: Scheduler) -> None:
        self._encode = encode
        self._scheduler = scheduler

    @asynccontextmanager
    async def running(self, endpoint: str) -> AsyncIterator[None]:
        with closing(create_async_ipc_pair_socket(endpoint, bind=True)) as socket:
            async with asyncio.TaskGroup() as tasks:
                task = tasks.create_task(self._serve(socket))
                try:
                    yield
                finally:
                    task.cancel()

    async def _serve(self, socket: zmq.asyncio.Socket) -> None:
        while True:
            message = await socket.recv()
            request_id = None
            try:
                data = json.loads(message)
                request_id = data["request_id"]
                prompt = data["prompt"]
                if not isinstance(request_id, str) or not isinstance(prompt, str):
                    raise ValueError("Expected a request ID and prompt")
                token_ids = await asyncio.to_thread(self._encode, prompt)
                await self._scheduler.prefill(
                    TokenizedRequest(request_id, tuple(token_ids))
                )
            except Exception:
                logger.exception("Failed to prefill request %s", request_id)
                await socket.send_json({"request_id": request_id, "ok": False})
            else:
                await socket.send_json({"request_id": request_id, "ok": True})


class PrefillFailed(RuntimeError):
    pass


class TokenizerClient:
    def __init__(self, endpoint: str) -> None:
        self._socket = create_async_ipc_pair_socket(endpoint, bind=False)
        self._lock = asyncio.Lock()

    async def submit(self, prompt: str) -> str:
        request_id = uuid4().hex
        async with self._lock:
            await self._socket.send_json({"request_id": request_id, "prompt": prompt})
            while True:
                response = await self._socket.recv_json()
                if response["request_id"] == request_id:
                    break

        if not response["ok"]:
            raise PrefillFailed("Prefill failed")
        return request_id

    def close(self) -> None:
        self._socket.close()
