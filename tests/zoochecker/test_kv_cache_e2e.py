import asyncio
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import httpx
import mlx.core as mx

from draft.vanilla_mlx_kv_cache import VanillaAttention
from zoochecker.cache import UnifiedCacheManager
from zoochecker.https_server import create_app
from zoochecker.scheduler import Scheduler
from zoochecker.server_args import ServerArgs
from zoochecker.types import TokenizedRequest


@unittest.skipUnless(mx.metal.is_available(), "requires MLX Metal")
class KVCacheE2ETests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        mx.random.seed(0)
        self.model = VanillaAttention(width=64, heads=4, layers=2, max_tokens=9)

    def app(self, dtype="float32"):
        args = ServerArgs("vanilla", None, cache_dtype=dtype, cache_slots=2, cache_max_tokens=9)
        return create_app(args, num_layers=2, num_kv_heads=4, head_dim=16)

    def assert_close(self, actual, expected, tolerance=1e-5):
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(mx.allclose(actual, expected, atol=tolerance, rtol=tolerance).item())

    async def test_scheduler_releases_cache_when_prefill_is_cancelled(self):
        admitted = asyncio.Queue()
        block_handler = asyncio.Event()

        async def handle(request, cache):
            admitted.put_nowait((request, cache))
            await block_handler.wait()

        args = ServerArgs("vanilla", None, cache_max_tokens=3)
        manager = UnifiedCacheManager(args, num_layers=2, num_kv_heads=4, head_dim=16)
        with closing(manager):
            scheduler = Scheduler(manager, handle)
            async with asyncio.TaskGroup() as tasks:
                task = tasks.create_task(scheduler.prefill(TokenizedRequest("request", (1, 2, 3))))
                try:
                    _, lease = await asyncio.wait_for(admitted.get(), timeout=2)
                    with self.assertRaises(BufferError):
                        manager.allocate()
                finally:
                    task.cancel()

            with manager.request_cache() as cache:
                self.assertEqual(cache.offset(0), 0)
            with self.assertRaises(RuntimeError):
                lease.fetch(0)

    async def test_http_request_reaches_mlx_prefill_through_tokenizer_ipc(self):
        seen = []

        async def prefill(request, cache):
            self.assertEqual(cache.offset(0), 0)
            mx.eval(self.model(mx.array([request.token_ids]), cache))
            seen.append((request.request_id, request.token_ids, cache.offset(0)))

        with tempfile.TemporaryDirectory() as temp_dir:
            args = ServerArgs(
                "vanilla",
                None,
                cache_dtype="float32",
                cache_max_tokens=3,
                tokenizer_ipc_endpoint=str(Path(temp_dir) / "ipc.sock"),
            )
            app = create_app(
                args,
                num_layers=2,
                num_kv_heads=4,
                head_dim=16,
                encode=lambda prompt: list(map(int, prompt)),
                prefill=prefill,
            )
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    response = await asyncio.wait_for(
                        client.post("/v1/prefill", json={"prompt": "123"}), timeout=2
                    )
                with app.state.cache_manager.request_cache() as cache:
                    self.assertEqual(cache.offset(0), 0)

            self.assertEqual(response.status_code, 200)
            self.assertEqual(seen, [(response.json()["request_id"], (1, 2, 3), 3)])

    async def test_prefill_then_decode_at_exact_capacity(self):
        app = self.app()
        ids = mx.arange(9)[None, :]
        async with app.router.lifespan_context(app):
            manager = app.state.cache_manager
            with manager.request_cache() as cache:
                outputs = [self.model(ids[:, :4], cache)]
                outputs += [self.model(ids[:, i:i + 1], cache) for i in range(4, 9)]
                self.assert_close(mx.concatenate(outputs, axis=1), self.model(ids))
                self.assertEqual([cache.offset(i) for i in range(2)], [9, 9])
            outstanding = manager.allocate()

        self.assertFalse(hasattr(app.state, "cache_manager"))
        with self.assertRaises(RuntimeError):
            outstanding.fetch(0)

    async def test_bfloat16_chunked_prefill_then_decode(self):
        app = self.app("bfloat16")
        self.model.set_dtype(mx.bfloat16)
        ids = mx.array([[12, 13, 9, 1, 8, 7, 4]])
        async with app.router.lifespan_context(app):
            with app.state.cache_manager.request_cache() as cache:
                chunks = (ids[:, :2], ids[:, 2:6], ids[:, 6:])
                outputs = [self.model(chunk, cache) for chunk in chunks]
                self.assert_close(mx.concatenate(outputs, axis=1), self.model(ids), 0.01)
                self.assertEqual(cache.fetch(1)[0].dtype, mx.bfloat16)

    async def test_interleaved_requests_and_slot_reuse(self):
        app = self.app()
        a = mx.array([[5, 4, 3, 2]])
        b = mx.array([[6, 7, 8, 9, 10]])
        c = mx.array([[55, 61]])
        async with app.router.lifespan_context(app):
            manager = app.state.cache_manager
            with manager.request_cache() as second:
                with manager.request_cache() as first:
                    a_prefix = self.model(a[:, :2], first)
                    b_prefix = self.model(b[:, :3], second)
                    a_tail = self.model(a[:, 2:], first)
                with manager.request_cache() as reused:
                    self.assert_close(self.model(c, reused), self.model(c))
                    b_tail = self.model(b[:, 3:], second)
                self.assert_close(mx.concatenate([a_prefix, a_tail], axis=1), self.model(a))
                self.assert_close(mx.concatenate([b_prefix, b_tail], axis=1), self.model(b))

    async def test_resume_another_request_from_a_released_slots_prefix(self):
        app = self.app()
        original = mx.array([[5, 4, 3, 2]])
        follow_up = mx.array([[5, 4, 15, 16]])
        async with app.router.lifespan_context(app):
            manager = app.state.cache_manager
            with manager.request_cache() as source:
                mx.eval(self.model(original, source))
                snapshot = source.snapshot(2)
            with manager.request_cache() as reused:
                mx.eval(self.model(mx.array([[1, 2]]), reused))
            with manager.request_cache() as target:
                target.restore(snapshot)
                del snapshot
                self.assertEqual([target.offset(i) for i in range(2)], [2, 2])
                resumed = self.model(follow_up[:, 2:], target)
                self.assert_close(resumed, self.model(follow_up)[:, 2:])
