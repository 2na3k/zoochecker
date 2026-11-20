import asyncio
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import httpx
import mlx.core as mx

from draft.vanilla_mlx_kv_cache import VanillaAttention
from zoochecker.cache import UnifiedCacheManager
from zoochecker.api.v1.service import create_app
from zoochecker.ipc_connector import WorkerClient, serve_worker
from zoochecker.scheduler import Scheduler
from zoochecker.server_args import ServerArgs
from zoochecker.types import GeneratedTokens, OutputDelta, Sampling, TokenizedRequest
from zoochecker.worker.runner import WorkerService


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

    async def test_worker_ipc_shutdown_releases_cache_and_cancels_scheduler_requests(self):
        admitted = asyncio.Queue()
        block_handler = asyncio.Event()

        async def handle(request, cache, emit):
            admitted.put_nowait((request, cache))
            await block_handler.wait()

        args = ServerArgs("vanilla", None, cache_max_tokens=3, cache_slots=1)
        manager = UnifiedCacheManager(args, num_layers=2, num_kv_heads=4, head_dim=16)
        with closing(manager), tempfile.TemporaryDirectory() as temp_dir:
            endpoint = Path(temp_dir) / "worker.sock"
            service = WorkerService(manager, handle)
            async with serve_worker(endpoint, service.run):
                async with WorkerClient(endpoint) as client:
                    scheduler = Scheduler(client, capacity=args.cache_slots)
                    async with scheduler.running():
                        active = asyncio.create_task(
                            scheduler.run(TokenizedRequest("active", (1, 2, 3)))
                        )
                        request, lease = await asyncio.wait_for(admitted.get(), timeout=2)
                        self.assertEqual(request.token_ids, (1, 2, 3))
                        queued = asyncio.create_task(
                            scheduler.run(TokenizedRequest("queued", (4,)))
                        )
                        await asyncio.sleep(0)
                        self.assertEqual(scheduler.counts, (1, 1))
                        with self.assertRaises(BufferError):
                            manager.allocate()
                    for task in (active, queued):
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    self.assertEqual(scheduler.counts, (0, 0))

            with manager.request_cache() as cache:
                self.assertEqual(cache.offset(0), 0)
            with self.assertRaises(RuntimeError):
                lease.fetch(0)

    async def test_stream_cancellation_reaches_worker_and_releases_lease(self):
        async def handle(request, cache, emit):
            self.assertEqual(cache.offset(0), 0)
            if emit is not None:
                token = 42 if request.request_id == "stream" else 43
                await emit(OutputDelta(request.request_id, "", token, prompt_tokens=1, completion_tokens=1))
                if request.request_id == "stream":
                    await asyncio.Future()
            return GeneratedTokens((43,), "length")

        args = ServerArgs("vanilla", None, cache_max_tokens=3, cache_slots=1)
        manager = UnifiedCacheManager(args, num_layers=2, num_kv_heads=4, head_dim=16)
        with closing(manager), tempfile.TemporaryDirectory() as temp_dir:
            endpoint = Path(temp_dir) / "worker.sock"
            service = WorkerService(manager, handle)
            async with serve_worker(endpoint, service.run), asyncio.timeout(5):
                async with WorkerClient(endpoint) as client:
                    scheduler = Scheduler(client, capacity=args.cache_slots)
                    async with scheduler.running():
                        output = asyncio.Queue()
                        active = asyncio.create_task(
                            scheduler.run(TokenizedRequest("stream", (1,), Sampling(1, 0, 1)), output.put)
                        )
                        self.assertEqual(
                            await output.get(), OutputDelta("stream", "", 42, prompt_tokens=1, completion_tokens=1)
                        )
                        active.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await active
                        result = await scheduler.run(TokenizedRequest("next", (2,), Sampling(1, 0, 1)), output.put)
                        self.assertEqual(result, GeneratedTokens((43,), "length"))
                        self.assertEqual(
                            await output.get(), OutputDelta("next", "", 43, prompt_tokens=1, completion_tokens=1)
                        )
                        self.assertEqual(
                            await output.get(), OutputDelta("next", "", -1, "length", 1, 1, (43,))
                        )
                        self.assertEqual(manager.stats().slots_used, 0)

    async def test_http_request_reaches_mlx_prefill_through_both_ipc_hops(self):
        seen = []
        prefilled = asyncio.Event()
        finish = asyncio.Event()

        async def prefill(request, cache):
            self.assertEqual(cache.offset(0), 0)
            mx.eval(self.model(mx.array([request.token_ids]), cache))
            seen.append((request.request_id, request.token_ids, cache.offset(0)))
            prefilled.set()
            await finish.wait()

        args = ServerArgs(
            "vanilla",
            None,
            cache_dtype="float32",
            cache_max_tokens=3,
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
                async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
                    pending = tasks.create_task(
                        client.post("/v1/prefill", json={"prompt": "123"})
                    )
                    await prefilled.wait()
                    try:
                        self.assertEqual((await client.get("/v1/health")).json(), {"status": "ok"})
                        stats = (await client.get("/v1/stats")).json()
                        self.assertEqual(stats["active_requests"], 1)
                        self.assertEqual(stats["scheduler"], {"waiting_seqs": 0, "running_seqs": 1})
                        self.assertEqual(stats["kvcache_text"]["slots_used"], 1)
                        self.assertGreater(stats["kvcache_text"]["bytes_used"], 0)
                    finally:
                        finish.set()
                    response = await pending
                stats = (await client.get("/v1/stats")).json()
                self.assertEqual(stats["active_requests"], 0)
                self.assertEqual(stats["scheduler"]["running_seqs"], 0)
                self.assertEqual(stats["kvcache_text"]["slots_used"], 0)
            with app.state.cache_manager.request_cache() as cache:
                self.assertEqual(cache.offset(0), 0)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(seen, [(response.json()["request_id"], (1, 2, 3), 3)])

    async def test_concurrent_requests_batch_then_reuse_a_free_slot(self):
        started = asyncio.Queue()
        release = {token: asyncio.Event() for token in (1, 2, 3)}

        async def prefill(request, cache):
            token, = request.token_ids
            self.assertEqual(cache.offset(0), 0)
            mx.eval(self.model(mx.array([[token]]), cache))
            started.put_nowait((token, request.request_id))
            await release[token].wait()

        args = ServerArgs(
            "vanilla", None, cache_slots=2, cache_max_tokens=3,
            scheduler_batch_wait_seconds=0.05,
        )
        app = create_app(
            args, num_layers=2, num_kv_heads=4, head_dim=16,
            encode=lambda prompt: [int(prompt)], prefill=prefill,
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
            asyncio.timeout(5),
        ):
            requests = {
                token: asyncio.create_task(client.post("/v1/prefill", json={"prompt": str(token)}))
                for token in (1, 2, 3)
            }
            try:
                first, first_id = await started.get()
                second, second_id = await started.get()
                stats = (await client.get("/v1/stats")).json()
                self.assertEqual(stats["scheduler"], {"waiting_seqs": 1, "running_seqs": 2})
                self.assertEqual(stats["kvcache_text"]["slots_used"], 2)

                release[first].set()
                self.assertEqual((await requests[first]).json()["request_id"], first_id)
                third, third_id = await started.get()
                self.assertFalse(requests[second].done())
                release[second].set()
                release[third].set()
                self.assertEqual((await requests[second]).json()["request_id"], second_id)
                self.assertEqual((await requests[third]).json()["request_id"], third_id)

                events = (await client.get("/v1/events")).json()
                sizes = {
                    event["request_id"]: event["batch_size"]
                    for event in events if event["event_type"] == "request_dispatched"
                }
                self.assertEqual(sizes, {first_id: 2, second_id: 2, third_id: 1})
            finally:
                for event in release.values():
                    event.set()
                await asyncio.gather(*requests.values(), return_exceptions=True)

    async def test_health_stays_available_with_telemetry_disabled(self):
        async def prefill(request, cache):
            return None

        args = ServerArgs(
            "vanilla", None, telemetry_enabled=False,
        )
        app = create_app(
            args, num_layers=2, num_kv_heads=4, head_dim=16,
            encode=lambda prompt: [1], prefill=prefill,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            self.assertEqual((await client.get("/v1/health")).status_code, 503)
            async with app.router.lifespan_context(app):
                self.assertEqual((await client.get("/v1/health")).status_code, 200)
                self.assertEqual((await client.get("/health")).status_code, 200)
                for route in ("/v1/metrics", "/v1/stats", "/v1/events"):
                    self.assertEqual((await client.get(route)).status_code, 503)
            self.assertEqual((await client.get("/v1/health")).status_code, 503)

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
