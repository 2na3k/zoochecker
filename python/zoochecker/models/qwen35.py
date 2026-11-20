import asyncio
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import create_attention_mask

from zoochecker.cache import RequestKVCache
from zoochecker.images import decode_prepared_images
from zoochecker.types import InvalidPrompt, TokenizedRequest

from ._base import BaseModel


class _AttentionCache:
    def __init__(self, lease: RequestKVCache, layer: int) -> None:
        self._lease = lease
        self._layer = layer

    @property
    def offset(self) -> int:
        return self._lease.offset(self._layer)

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        cached_keys, cached_values = self._lease.update_and_fetch(
            self._layer, keys, values
        )
        return cached_keys.astype(keys.dtype), cached_values.astype(values.dtype)

    def make_mask(
        self, length: int, *, return_array: bool = False, window_size: int | None = None
    ) -> mx.array | str | None:
        return create_attention_mask(length, self.offset, return_array, window_size)


@dataclass(slots=True)
class _QwenCache:
    layers: list
    inputs: dict | None = None
    embeddings: mx.array | None = None
    positions: mx.array | None = None
    rope_delta: mx.array | None = None


class Qwen35(BaseModel):
    def __init__(self, model: Any, *, processor: Any = None) -> None:
        self._model = model
        self._text = model.language_model
        self._processor = processor
        self._full_layers = tuple(
            index
            for index, layer in enumerate(self._text.layers)
            if not layer.is_linear
        )
        if not self._full_layers:
            raise ValueError("Qwen 3.5 must have at least one full-attention layer")

    @property
    def cache_geometry(self) -> tuple[int, int, int]:
        args = self._text.args
        return len(self._full_layers), args.num_key_value_heads, args.head_dim

    @property
    def context_length(self) -> int:
        return self._text.args.max_position_embeddings

    def make_cache(self, lease: RequestKVCache) -> _QwenCache:
        cache = self._text.make_cache()
        for slot, layer in enumerate(self._full_layers):
            cache[layer] = _AttentionCache(lease, slot)
        return _QwenCache(cache)

    async def prepare_prompt(
        self, request: TokenizedRequest, cache: _QwenCache
    ) -> tuple[int, ...]:
        if request.vision is None or self._processor is None:
            return await super().prepare_prompt(request, cache)
        from mlx_vlm.utils import prepare_inputs

        images = await asyncio.to_thread(decode_prepared_images, request.vision.images)
        try:
            cache.inputs = prepare_inputs(
                self._processor,
                images=images,
                prompts=request.vision.text,
                image_token_index=self._model.config.image_token_index,
            )
        except ValueError as exc:
            raise InvalidPrompt("Could not prepare images for Qwen") from exc
        return tuple(cache.inputs["input_ids"][0].tolist())

    def image_size(self, width: int, height: int) -> tuple[int, int]:
        if self._processor is None:
            return super().image_size(width, height)
        resized_height, resized_width = self._processor.image_processor._resolved_size(
            height, width
        )
        return resized_width, resized_height

    def forward(
        self,
        tokens: mx.array,
        cache: _QwenCache,
        *,
        logits: bool = True,
    ) -> mx.array:
        if self._processor is None:
            hidden = self._text.model(tokens, cache=cache.layers)[:, -1, :]
        else:
            hidden = self._vision_forward(tokens, cache)[:, -1, :]
        if not logits:
            output = hidden
        elif self._text.args.tie_word_embeddings:
            output = self._text.model.embed_tokens.as_linear(hidden)
        else:
            output = self._text.lm_head(hidden)
        mx.eval(
            output,
            [
                cache.layers[i].state
                for i, layer in enumerate(self._text.layers)
                if layer.is_linear
            ],
        )
        return output

    def _vision_forward(self, tokens: mx.array, cache: _QwenCache) -> mx.array:
        start = cache.layers[self._full_layers[0]].offset
        end = start + tokens.shape[-1]
        if cache.inputs is not None:
            features = self._model.get_input_embeddings(
                cache.inputs["input_ids"],
                cache.inputs["pixel_values"],
                image_grid_thw=cache.inputs["image_grid_thw"],
                mask=cache.inputs.get("attention_mask"),
            )
            cache.embeddings = features.inputs_embeds
            cache.positions = features.position_ids
            cache.rope_delta = features.rope_deltas
            mx.eval(cache.embeddings, cache.positions, cache.rope_delta)
            cache.inputs = None
        embeddings = None
        if cache.embeddings is not None:
            embeddings = cache.embeddings[:, start:end]
            positions = cache.positions[..., start:end]
            if end == cache.embeddings.shape[1]:
                cache.embeddings = cache.positions = None
        else:
            positions = mx.arange(start, end)[None, :]
            if cache.rope_delta is not None:
                positions = mx.broadcast_to(
                    (positions + cache.rope_delta)[None, ...], (3, 1, tokens.shape[-1])
                )
        return self._text.model(
            tokens, inputs_embeds=embeddings, cache=cache.layers, position_ids=positions
        )
