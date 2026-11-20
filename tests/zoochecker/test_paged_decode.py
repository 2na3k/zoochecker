import unittest

import mlx.core as mx


@unittest.skipUnless(mx.metal.is_available(), "requires MLX Metal")
class PagedDecodeTests(unittest.TestCase):
    def test_matches_contiguous_grouped_attention_for_permuted_pages(self):
        from zoochecker.kernels.paged_decode import paged_decode

        heads, kv_heads, dim, page_tokens = 8, 2, 256, 16
        table = mx.array([7, 2, 5, 0, 6], dtype=mx.int32)
        shape = (kv_heads, 8, page_tokens, dim)
        query = mx.random.normal((1, heads, 1, dim), key=mx.random.key(0)).astype(
            mx.bfloat16
        )
        keys = mx.random.normal(shape, key=mx.random.key(1)).astype(mx.bfloat16)
        values = mx.random.normal(shape, key=mx.random.key(2)).astype(mx.bfloat16)
        contiguous_keys = mx.take(keys, table, axis=1).reshape(1, kv_heads, -1, dim)
        contiguous_values = mx.take(values, table, axis=1).reshape(1, kv_heads, -1, dim)
        scale = dim**-0.5

        for length, partition in (
            (1, 64),
            (16, 64),
            (17, 64),
            (64, 64),
            (65, 31),
            (79, 64),
        ):
            with self.subTest(length=length, partition=partition):
                expected = mx.fast.scaled_dot_product_attention(
                    query,
                    contiguous_keys[:, :, :length, :],
                    contiguous_values[:, :, :length, :],
                    scale=scale,
                )
                actual = paged_decode(
                    query,
                    keys,
                    values,
                    table,
                    length=length,
                    scale=scale,
                    partition=partition,
                )
                self.assertTrue(
                    mx.allclose(actual, expected, atol=0.004, rtol=0.01).item()
                )
