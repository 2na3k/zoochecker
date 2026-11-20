"""handle the tokenizer and stuffs, tokenizer then push to scheduler and take from worker to push the result back to the http"""

import asyncio
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING

from zoochecker.scheduler import Scheduler
from zoochecker.server_args import ServerArgs
from zoochecker.types import (
    EmitDelta,
    Generation,
    InvalidPrompt,
    OutputDelta,
    PromptRequest,
    TokenizedRequest,
    VisionPrompt,
)

if TYPE_CHECKING:
    from mlx_lm.tokenizer_utils import StreamingDetokenizer, TokenizerWrapper

    from zoochecker.metrics import Metrics

Encode = Callable[[str], Sequence[int]]
EncodeChat = Callable[[list[dict]], Sequence[int] | VisionPrompt]
Decode = Callable[[Sequence[int]], str]


class Tokenizer:
    def __init__(
        self,
        encode: Encode,
        scheduler: Scheduler,
        *,
        encode_chat: EncodeChat | None = None,
        decode: Decode | None = None,
        detokenizer: "Callable[[], StreamingDetokenizer] | None" = None,
        special_tokens: frozenset[int] = frozenset(),
        metrics: "Metrics | None" = None,
    ) -> None:
        self._encode = encode
        self._scheduler = scheduler
        self._encode_chat = encode_chat
        self._decode = decode
        self._detokenizer = detokenizer
        self._special_tokens = special_tokens
        self._metrics = metrics

    @classmethod
    def from_codec(
        cls,
        args: ServerArgs,
        codec: "TokenizerWrapper",
        scheduler: Scheduler,
        metrics: "Metrics | None" = None,
    ) -> "Tokenizer":
        from jinja2 import TemplateError
        from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

        template = NaiveStreamingDetokenizer(codec)

        def detokenizer() -> "StreamingDetokenizer":
            instance = copy(template)
            instance.reset()
            return instance

        def encode_chat(messages: list[dict]) -> list[int] | VisionPrompt:
            images = []
            formatted = []
            for message in messages:
                content = message["content"]
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if part["type"] == "image_url":
                            images.append(part["image_url"]["url"])
                            parts.append({"type": "image"})
                        else:
                            parts.append(part)
                    content = parts
                formatted.append({"role": message["role"], "content": content})
            try:
                prompt = codec.apply_chat_template(
                    formatted,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=args.enable_thinking,
                )
            except TemplateError as exc:
                raise InvalidPrompt(str(exc)) from exc
            if images:
                return VisionPrompt(prompt, tuple(images))
            return codec.encode(prompt, add_special_tokens=False)

        return cls(
            codec.encode,
            scheduler,
            encode_chat=encode_chat,
            decode=lambda tokens: codec.decode(list(tokens), skip_special_tokens=True),
            detokenizer=detokenizer,
            special_tokens=frozenset(codec.all_special_ids),
            metrics=metrics,
        )

    async def process(
        self, request: PromptRequest, emit: EmitDelta | None = None
    ) -> Generation | None:
        operation = "completion" if isinstance(request.prompt, str) else "chat"
        if request.sampling is None:
            operation = "prefill"
        metrics = self._metrics
        span_context = (
            metrics.request(request.request_id, operation)
            if metrics is not None
            else nullcontext()
        )
        with span_context as span:
            vision = None
            if isinstance(request.prompt, str):
                tokens = await asyncio.to_thread(self._encode, request.prompt)
            else:
                if self._encode_chat is None:
                    raise InvalidPrompt("Chat is unavailable")
                encoded = await asyncio.to_thread(self._encode_chat, request.prompt)
                if isinstance(encoded, VisionPrompt):
                    vision, tokens = encoded, ()
                else:
                    tokens = encoded
            if span is not None:
                span.prompt_tokens = len(tokens)
                span.path = "vlm" if vision is not None else "text"
                metrics.record(
                    "request_admitted",
                    request_id=request.request_id,
                    operation=operation,
                    path=span.path,
                    prompt_tokens=len(tokens),
                )
            detokenizer = None
            terminal = None
            if emit is not None:
                if self._detokenizer is None:
                    raise InvalidPrompt("Streaming is unavailable")
                detokenizer = self._detokenizer()

            async def emit_delta(delta: OutputDelta) -> None:
                nonlocal terminal
                if delta.finish_reason is not None:
                    terminal = delta
                elif delta.token_id not in self._special_tokens:
                    assert detokenizer is not None and emit is not None
                    detokenizer.add_token(delta.token_id)
                    if segment := detokenizer.last_segment:
                        await emit(replace(delta, token_text=segment))

            output = await self._scheduler.run(
                TokenizedRequest(
                    request.request_id, tuple(tokens), request.sampling, vision
                ),
                emit_delta if detokenizer is not None else None,
            )
            if request.sampling is None:
                return None
            if output is None or self._decode is None:
                raise RuntimeError("Generation is unavailable")
            if detokenizer is not None:
                if terminal is None:
                    raise RuntimeError("Worker stream ended without a terminal delta")
                assert emit is not None
                detokenizer.finalize()
                await emit(replace(terminal, token_text=detokenizer.last_segment))
                text = detokenizer.text
            else:
                text = await asyncio.to_thread(self._decode, output.token_ids)
            if span is not None:
                span.finish_reason = output.finish_reason
                span.completion_tokens = len(output.token_ids)
            return Generation(
                request.request_id,
                text,
                output.prompt_tokens
                if output.prompt_tokens is not None
                else len(tokens),
                len(output.token_ids),
                output.finish_reason,
            )
