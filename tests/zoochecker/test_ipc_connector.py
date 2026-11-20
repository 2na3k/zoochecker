import asyncio
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from zoochecker.ipc_connector import (
    TokenizerClient,
    WorkerClient,
    create_async_ipc_pair_socket,
    ipc_endpoints,
    serve_tokenizer,
    serve_worker,
)
from zoochecker.types import (
    GeneratedTokens,
    Generation,
    InferenceFailed,
    OutputDelta,
    Sampling,
    TokenizedRequest,
)


class IpcPairSocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_stream_does_not_block_other_replies(self) -> None:
        receiving = asyncio.Event()
        cancelled = asyncio.Event()

        async def process(request, emit):
            if emit is not None:
                try:
                    await emit(OutputDelta(request.request_id, "slow", 1))
                    await asyncio.Future()
                finally:
                    cancelled.set()
            return Generation(request.request_id, "fast", 1, 1, "length", (1,))

        async def consume(delta):
            self.assertEqual(delta.token_text, "slow")
            receiving.set()
            await asyncio.Future()

        with ipc_endpoints() as (endpoint, _):
            async with (
                serve_tokenizer(endpoint, process),
                TokenizerClient(endpoint) as client,
                asyncio.timeout(5),
            ):
                sampling = Sampling(1, 0, 1)
                stream = asyncio.create_task(
                    client.generate(sampling, prompt="slow", emit=consume)
                )
                try:
                    await receiving.wait()
                    self.assertEqual(
                        (await client.generate(sampling, prompt="fast")).text, "fast"
                    )
                    self.assertFalse(stream.done())
                finally:
                    stream.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await stream
                self.assertTrue(cancelled.is_set())

    async def test_invalid_tokenizer_input_is_rejected_before_dispatch(self) -> None:
        async def process(request, emit):
            self.fail("Invalid request reached the tokenizer")

        with ipc_endpoints() as (endpoint, _):
            async with serve_tokenizer(endpoint, process), asyncio.timeout(2):
                with closing(
                    create_async_ipc_pair_socket(endpoint, bind=False)
                ) as socket:
                    for payload in (
                        {"messages": [42]},
                        {
                            "prompt": "text",
                            "sampling": {
                                "max_tokens": -1,
                                "temperature": "bad",
                                "top_p": 3,
                            },
                        },
                        {"prompt": "text", "stream": "true"},
                    ):
                        await socket.send_json({"request_id": "test", **payload})
                        response = await socket.recv_json()
                        self.assertIs(response["invalid_prompt"], True)

    async def test_invalid_worker_finish_reason_is_rejected(self) -> None:
        async def run(request, emit):
            return GeneratedTokens((1,), "unexpected", 1)

        with ipc_endpoints() as (_, endpoint):
            async with (
                serve_worker(endpoint, run),
                WorkerClient(endpoint) as client,
                asyncio.timeout(2),
            ):
                (reply,) = await client.submit(
                    [(TokenizedRequest("test", (1,), Sampling(1, 0, 1)), False)]
                )
                try:
                    with self.assertRaisesRegex(
                        InferenceFailed, "Invalid IPC worker result"
                    ):
                        await reply.result(None)
                finally:
                    await reply.close()

    async def test_terminal_delta_without_output_token_ids_is_rejected(self) -> None:
        async def emit(delta):
            self.fail("Invalid delta reached the caller")

        with ipc_endpoints() as (endpoint, _):
            with closing(create_async_ipc_pair_socket(endpoint, bind=True)) as socket:
                async with TokenizerClient(endpoint) as client, asyncio.timeout(2):
                    task = asyncio.create_task(
                        client.generate(Sampling(1, 0, 1), prompt="text", emit=emit)
                    )
                    request_id = (await socket.recv_json())["request_id"]
                    await socket.send_json(
                        {
                            "request_id": request_id,
                            "event": {
                                "request_id": request_id,
                                "token_text": "",
                                "token_id": -1,
                                "finish_reason": "stop",
                            },
                        }
                    )
                    await socket.send_json({"request_id": request_id, "result": None})
                    with self.assertRaisesRegex(
                        InferenceFailed, "Invalid IPC stream delta"
                    ):
                        await task

    async def test_allocated_endpoints_are_distinct_and_cleaned_up(self) -> None:
        with ipc_endpoints() as (tokenizer, worker), ipc_endpoints() as other:
            self.assertNotEqual(tokenizer, worker)
            self.assertNotEqual(tokenizer.parent, other[0].parent)
            self.assertEqual(tokenizer.parent.stat().st_mode & 0o777, 0o700)
            await self._assert_round_trip(tokenizer)
            await self._assert_round_trip(worker)
        self.assertFalse(tokenizer.parent.exists())
        self.assertFalse(other[0].parent.exists())

    async def _assert_round_trip(self, endpoint: str | Path) -> None:
        with (
            closing(create_async_ipc_pair_socket(endpoint, bind=True)) as server,
            closing(create_async_ipc_pair_socket(endpoint, bind=False)) as client,
        ):
            async with asyncio.timeout(2):
                await client.send_json({"type": "request", "value": 42})
                self.assertEqual(
                    await server.recv_json(), {"type": "request", "value": 42}
                )
                await server.send_json({"type": "response", "value": 42})
                self.assertEqual(
                    await client.recv_json(), {"type": "response", "value": 42}
                )

    async def test_path_endpoint_is_full_duplex_and_creates_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            endpoint = Path(temp_dir) / "nested" / "zoochecker.sock"
            await self._assert_round_trip(endpoint)
            self.assertTrue(endpoint.parent.is_dir())

    async def test_ipc_url_endpoint_is_full_duplex(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            endpoint = f"ipc://{Path(temp_dir) / 'zoochecker.sock'}"
            await self._assert_round_trip(endpoint)

    def test_rejects_non_ipc_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "ipc://"):
            create_async_ipc_pair_socket("tcp://127.0.0.1:5555", bind=True)


if __name__ == "__main__":
    unittest.main()
