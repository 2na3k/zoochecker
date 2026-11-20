import unittest

import mlx.core as mx

from draft.vanilla_mlx_kv_cache import VanillaAttention
from zoochecker.https_server import create_app
from zoochecker.server_args import ServerArgs


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
