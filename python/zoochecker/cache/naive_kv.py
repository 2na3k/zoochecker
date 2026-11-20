from dataclasses import dataclass
from threading import Lock

import mlx.core as mx

from zoochecker.server_args import ServerArgs

from ._base import KVCachePool, KVPrefix, RequestKVCache
from .utils import compact_snapshot


@dataclass
class _Layer:
    keys: mx.array
    values: mx.array
    length: int = 0


class MLXRequestKVCache(RequestKVCache):
    def __init__(self, layers: tuple[_Layer, ...]) -> None:
        self._layers: tuple[_Layer, ...] | None = layers

    def _require_active(self) -> tuple[_Layer, ...]:
        if self._layers is None:
            raise RuntimeError("KV cache lease has been released")
        return self._layers

    def _layer(self, index: int) -> _Layer:
        layers = self._require_active()
        if not 0 <= index < len(layers):
            raise IndexError("KV cache layer index out of range")
        return layers[index]

    def _release(self) -> None:
        if self._layers is not None:
            for layer in self._layers:
                if layer.length:
                    layer.keys = mx.zeros(
                        layer.keys.shape, dtype=layer.keys.dtype, stream=mx.gpu
                    )
                    layer.values = mx.zeros(
                        layer.values.shape, dtype=layer.values.dtype, stream=mx.gpu
                    )
                    layer.length = 0
        self._layers = None

    def offset(self, layer: int) -> int:
        return self._layer(layer).length

    @property
    def capacity(self) -> int:
        return self._layer(0).keys.shape[2]

    def fetch(self, layer: int) -> tuple[mx.array, mx.array]:
        state = self._layer(layer)
        return self._prefix(state)

    def blocks(self, layer: int) -> tuple[tuple[mx.array, mx.array], ...]:
        state = self._layer(layer)
        return (self._prefix(state),) if state.length else ()

    def append(self, layer: int, keys: mx.array, values: mx.array) -> None:
        state = self._layer(layer)
        state.keys, state.values, state.length = self._prepare_append(
            state, keys, values
        )

    def update_and_fetch(
        self, layer: int, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        state = self._layer(layer)
        state.keys, state.values, state.length = self._prepare_append(
            state, keys, values
        )
        return self._prefix(state)

    def snapshot(self, prefix_len: int) -> KVPrefix:
        layers = self._require_active()
        if not 0 < prefix_len <= min(layer.length for layer in layers):
            raise ValueError("Snapshot prefix must be written to every KV cache layer")
        return compact_snapshot(
            prefix_len, tuple(self._prefix(layer, prefix_len) for layer in layers)
        )

    def restore(self, state: KVPrefix) -> None:
        layers = self._require_active()
        if any(layer.length for layer in layers):
            raise ValueError("Restore requires an empty KV cache lease")
        if state.prefix_len <= 0 or len(state.layers) != len(layers):
            raise ValueError("Invalid KV snapshot length or layer count")
        if any(
            keys.ndim != 4 or keys.shape[2] != state.prefix_len
            for keys, _ in state.layers
        ):
            raise ValueError("KV snapshot layer length differs from its prefix length")
        updates = [
            self._prepare_append(layer, keys, values)
            for layer, (keys, values) in zip(layers, state.layers, strict=True)
        ]
        for layer, (keys, values, length) in zip(layers, updates, strict=True):
            layer.keys, layer.values, layer.length = keys, values, length

    @staticmethod
    def _prepare_append(
        state: _Layer, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array, int]:
        batch, heads, capacity, head_dim = state.keys.shape
        if (
            keys.ndim != 4
            or keys.shape != values.shape
            or (keys.shape[0], keys.shape[1], keys.shape[3]) != (batch, heads, head_dim)
        ):
            raise ValueError(
                "keys and values must have shape [1, kv_heads, tokens, head_dim]"
            )
        if keys.shape[2] == 0:
            raise ValueError("KV cache updates require at least one token")
        if not all(mx.issubdtype(array.dtype, mx.floating) for array in (keys, values)):
            raise TypeError("keys and values must be floating-point arrays")
        end = state.length + keys.shape[2]
        if end > capacity:
            raise OverflowError(f"KV cache capacity is {capacity} tokens, got {end}")

        start = mx.array([state.length], dtype=mx.int32)
        updated_keys = mx.slice_update(
            state.keys, keys.astype(state.keys.dtype), start, (2,)
        )
        updated_values = mx.slice_update(
            state.values, values.astype(state.values.dtype), start, (2,)
        )
        return updated_keys, updated_values, end

    @staticmethod
    def _prefix(layer: _Layer, length: int | None = None) -> tuple[mx.array, mx.array]:
        end = layer.length if length is None else length
        return layer.keys[:, :, :end, :], layer.values[:, :, :end, :]


class MLXKVCachePool(KVCachePool):
    def __init__(
        self, args: ServerArgs, *, num_layers: int, num_kv_heads: int, head_dim: int
    ) -> None:
        dimensions = {
            "cache_slots": args.cache_slots,
            "cache_max_tokens": args.cache_max_tokens,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }
        for name, value in dimensions.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        dtypes = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
        if args.cache_dtype not in dtypes:
            raise ValueError(f"Unsupported KV cache dtype: {args.cache_dtype!r}")
        if not mx.metal.is_available():
            raise RuntimeError("The mlx_kv cache pool requires an MLX Metal GPU")

        dtype = dtypes[args.cache_dtype]
        shape = (1, num_kv_heads, args.cache_max_tokens, head_dim)
        self._slots = tuple(
            tuple(
                _Layer(
                    mx.zeros(shape, dtype=dtype, stream=mx.gpu),
                    mx.zeros(shape, dtype=dtype, stream=mx.gpu),
                )
                for _ in range(num_layers)
            )
            for _ in range(args.cache_slots)
        )
        mx.eval(
            [
                array
                for slot in self._slots
                for layer in slot
                for array in (layer.keys, layer.values)
            ]
        )
        self._leased: dict[int, MLXRequestKVCache] = {}
        self._lock = Lock()

    @property
    def capacity(self) -> int:
        return len(self._slots)

    def lease(self, slot: int) -> MLXRequestKVCache:
        with self._lock:
            if not 0 <= slot < self.capacity:
                raise IndexError("KV cache slot index out of range")
            if slot in self._leased:
                raise BufferError("KV cache slot is already leased")
            cache = MLXRequestKVCache(self._slots[slot])
            self._leased[slot] = cache
            return cache

    def release(self, slot: int, lease: RequestKVCache) -> None:
        with self._lock:
            active = self._leased.get(slot)
            if active is None or active is not lease:
                raise ValueError("KV cache lease is not active for this slot")
            active._release()
            del self._leased[slot]
