from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
import unittest

import mlx.core as mx

from zoochecker.cache import CacheFactory, CacheManager, KVCachePool, KVPrefix, UnifiedCacheManager
from zoochecker.cache.naive_kv import MLXKVCachePool
from zoochecker.server_args import ServerArgs


GEOMETRY = {"num_layers": 2, "num_kv_heads": 2, "head_dim": 4}


class CacheFactoryTests(unittest.TestCase):
    def test_rejects_invalid_settings(self):
        args = ServerArgs("test", None)
        for field, value in (
            ("cache_pool_type", "unknown"),
            ("cache_dtype", "int8"),
            ("cache_slots", 0),
            ("cache_max_tokens", -1),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                CacheFactory.create_pool(replace(args, **{field: value}), **GEOMETRY)
        with self.assertRaisesRegex(ValueError, "head_dim"):
            CacheFactory.create_pool(args, **dict(GEOMETRY, head_dim=0))

    @unittest.skipUnless(mx.metal.is_available(), "requires MLX Metal")
    def test_selects_pool_using_server_args(self):
        args = ServerArgs("test", None, cache_slots=2, cache_max_tokens=4)
        pool = CacheFactory.create_pool(args, **GEOMETRY)
        self.assertIsInstance(pool, KVCachePool)
        self.assertIsInstance(pool, MLXKVCachePool)
        self.assertEqual(pool.capacity, 2)

    @unittest.skipUnless(mx.metal.is_available(), "requires MLX Metal")
    def test_pool_rejects_duplicate_leases_and_stale_release(self):
        pool = CacheFactory.create_pool(ServerArgs("test", None, cache_max_tokens=4), **GEOMETRY)
        with self.assertRaises(IndexError):
            pool.lease(-1)
        first = pool.lease(0)
        self.assertEqual(first.capacity, 4)
        with self.assertRaises(BufferError):
            pool.lease(0)
        pool.release(0, first)
        second = pool.lease(0)
        try:
            with self.assertRaises(ValueError):
                pool.release(0, first)
            self.assertEqual(second.offset(0), 0)
            with self.assertRaises(RuntimeError):
                first.offset(0)
        finally:
            pool.release(0, second)


@unittest.skipUnless(mx.metal.is_available(), "requires MLX Metal")
class KVCacheTests(unittest.TestCase):
    def setUp(self):
        args = ServerArgs("test", None, cache_slots=2, cache_max_tokens=4)
        self.manager = UnifiedCacheManager(args, **GEOMETRY)
        self.addCleanup(self.manager.close)
        self.zeros = mx.zeros((1, 2, 2, 4))
        self.ones = mx.ones((1, 2, 2, 4))

    def assert_arrays_equal(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(mx.array_equal(actual, expected).item())

    def test_append_fetch_and_layer_offsets(self):
        with self.manager.request_cache() as cache:
            self.assertEqual(cache.fetch(0)[0].shape, (1, 2, 0, 4))
            original, _ = cache.update_and_fetch(0, self.ones, self.zeros)
            cache.update_and_fetch(0, self.zeros, self.ones)
            keys, values = cache.fetch(0)
            self.assertEqual(keys.dtype, mx.bfloat16)
            self.assert_arrays_equal(keys, mx.concatenate([self.ones, self.zeros], axis=2))
            self.assert_arrays_equal(values, mx.concatenate([self.zeros, self.ones], axis=2))
            self.assert_arrays_equal(original, self.ones)
            self.assertEqual(cache.offset(0), 4)
            self.assertEqual(cache.offset(1), 0)

    def test_invalid_writes_preserve_both_arrays_and_offset(self):
        with self.manager.request_cache() as cache:
            cache.update_and_fetch(0, self.ones, self.zeros)
            with self.assertRaises(OverflowError):
                too_long = mx.ones((1, 2, 3, 4))
                cache.update_and_fetch(0, too_long, too_long)
            with self.assertRaises(ValueError):
                cache.update_and_fetch(0, self.ones, self.zeros[:, :, :1, :])
            with self.assertRaises(ValueError):
                empty = self.zeros[:, :, :0, :]
                cache.update_and_fetch(0, empty, empty)
            with self.assertRaises(TypeError):
                cache.update_and_fetch(0, self.ones, self.zeros.astype(mx.int32))
            with self.assertRaises(IndexError):
                cache.fetch(-1)
            self.assertEqual(cache.offset(0), 2)
            keys, values = cache.fetch(0)
            self.assert_arrays_equal(keys, self.ones)
            self.assert_arrays_equal(values, self.zeros)

    def test_snapshot_requires_all_layers_and_failed_restore_preserves_empty_lease(self):
        with self.manager.request_cache() as cache:
            cache.update_and_fetch(0, self.ones, self.zeros)
            with self.assertRaises(ValueError):
                cache.snapshot(1)
        malformed = KVPrefix(2, (
            (self.ones, self.zeros),
            (self.ones, self.zeros[:, :, :, :1]),
        ))
        with self.manager.request_cache() as cache:
            with self.assertRaises(ValueError):
                cache.restore(malformed)
            self.assertEqual([cache.offset(i) for i in range(2)], [0, 0])

    def test_exhaustion_release_and_clean_reuse(self):
        first = self.manager.allocate()
        snapshot, _ = first.update_and_fetch(0, self.ones, self.zeros)
        with self.manager.request_cache() as second:
            second.update_and_fetch(0, self.ones * 2, self.zeros)
            with self.assertRaises(BufferError):
                self.manager.allocate()
            self.manager.deallocate(first)
            with self.assertRaises(RuntimeError):
                first.update_and_fetch(0, self.ones, self.zeros)
            with self.assertRaises(ValueError):
                self.manager.deallocate(first)
            with self.manager.request_cache() as replacement:
                self.assertEqual(replacement.offset(0), 0)
                self.assertEqual(replacement.fetch(0)[0].shape[2], 0)
                keys, _ = replacement.update_and_fetch(0, self.zeros, self.zeros)
                self.assert_arrays_equal(keys, self.zeros)
                self.assert_arrays_equal(second.fetch(0)[0], self.ones * 2)
        self.assert_arrays_equal(snapshot, self.ones)

    def test_context_returns_slot_on_error_and_shutdown_invalidates_leases(self):
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            with self.manager.request_cache() as cache:
                cache.update_and_fetch(0, self.ones, self.zeros)
                raise RuntimeError("forward failed")
        with self.manager.request_cache() as cache, self.manager.request_cache():
            self.assertEqual(cache.offset(0), 0)
            self.manager.close()
            with self.assertRaisesRegex(RuntimeError, "released"):
                cache.fetch(0)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                self.manager.allocate()

    def test_concurrent_admission_respects_slot_limit(self):
        barrier = Barrier(3, timeout=5)

        def allocate():
            barrier.wait()
            try:
                return self.manager.allocate()
            except BufferError:
                return None

        with ThreadPoolExecutor(max_workers=3) as executor:
            attempts = [executor.submit(allocate) for _ in range(3)]
            admitted = [cache for attempt in attempts if (cache := attempt.result()) is not None]
        self.assertEqual(len(admitted), 2)
        self.assertIsNot(admitted[0], admitted[1])
        for cache in admitted:
            self.manager.deallocate(cache)


@unittest.skipUnless(mx.metal.is_available(), "requires MLX Metal")
class CacheBackendTests(unittest.TestCase):
    def test_block_reads_expose_only_written_tokens(self):
        for pool_type, widths in (("mlx_kv", [35]), ("mlx_paged_kv", [16, 16, 3])):
            with self.subTest(pool_type=pool_type):
                args = ServerArgs("test", None, cache_pool_type=pool_type, cache_max_tokens=48)
                manager = CacheManager(CacheFactory.create_pool(args, **GEOMETRY))
                try:
                    with manager.request_cache() as cache:
                        self.assertEqual(cache.blocks(0), ())
                        keys = mx.arange(2 * 35 * 4).reshape(1, 2, 35, 4).astype(mx.float32)
                        cache.append(0, keys, -keys)
                        chunks = cache.blocks(0)
                        self.assertEqual([k.shape[2] for k, _ in chunks], widths)
                        start = 0
                        for k, v in chunks:
                            end = start + k.shape[2]
                            expected = keys[:, :, start:end, :].astype(k.dtype)
                            self.assertTrue(mx.array_equal(k, expected).item())
                            self.assertTrue(mx.array_equal(v, -expected).item())
                            start = end
                finally:
                    manager.close()

    def test_released_writes_and_small_snapshots_do_not_retain_backing_buffers(self):
        for pool_type in ("mlx_kv", "mlx_paged_kv"):
            with self.subTest(pool_type=pool_type):
                mx.synchronize()
                before_pool = mx.get_active_memory()
                args = ServerArgs("test", None, cache_pool_type=pool_type, cache_max_tokens=512)
                manager = UnifiedCacheManager(args, num_layers=1, num_kv_heads=4, head_dim=64)
                snapshots = []
                try:
                    for i in range(64):
                        with manager.request_cache() as cache:
                            keys = mx.full((1, 4, 64, 64), i, dtype=mx.bfloat16)
                            mx.eval(keys)
                            cache.append(0, keys, keys)
                            del keys
                    mx.synchronize()
                    self.assertLessEqual(mx.get_active_memory() - before_pool, 64 * 1024)

                    for i in range(32):
                        with manager.request_cache() as cache:
                            keys = mx.full((1, 4, 64, 64), i, dtype=mx.bfloat16)
                            cache.append(0, keys, keys)
                            snapshots.append(cache.snapshot(1))
                            del keys
                    mx.synchronize()
                    payload_bytes = len(snapshots) * 2 * 4 * 64 * 2
                    self.assertLessEqual(
                        mx.get_active_memory() - before_pool, payload_bytes + 64 * 1024
                    )
                    self.assertEqual(snapshots[-1].layers[0][0][0, 0, 0, 0].item(), 31)
                finally:
                    manager.close()
