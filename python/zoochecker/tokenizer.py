"""handle the tokenizer and stuffs, tokenizer then push to scheduler and take from worker to push the result back to the http"""

import asyncio
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from copy import copy
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

from zoochecker.scheduler import Scheduler
from zoochecker.server_args import ServerArgs
from zoochecker.types import (
    ChatMessage,
    EmitDelta,
    GeneratedTokens,
    Generation,
    ImagePart,
    InferenceFailed,
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
EncodeChat = Callable[[list[ChatMessage]], Sequence[int] | VisionPrompt]
Decode = Callable[[Sequence[int]], str]


def _encode_chat(
    codec: "TokenizerWrapper", enable_thinking: bool, messages: list[ChatMessage]
) -> list[int] | VisionPrompt:
    from jinja2 import TemplateError

    images = []
    formatted = []
    for message in messages:
        content = message.content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, ImagePart):
                    images.append(part.image_url.url)
                    parts.append({"type": "image"})
                else:
                    parts.append(part.model_dump())
            content = parts
        formatted.append({"role": message.role, "content": content})
    try:
        prompt = codec.apply_chat_template(
            formatted,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TemplateError as exc:
        raise InvalidPrompt(str(exc)) from exc
    if images:
        return VisionPrompt(prompt, tuple(images))
    return codec.encode(prompt, add_special_tokens=False)


class Tokenizer:
    def __init__(
        self,
        encode: Encode,
        scheduler: Scheduler,
        *,
        encode_chat: EncodeChat | None = None,
        decode: Decode | None = None,
        detokenizer: "StreamingDetokenizer | None" = None,
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
        from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

        return cls(
            codec.encode,
            scheduler,
            encode_chat=partial(_encode_chat, codec, args.enable_thinking),
            decode=lambda tokens: codec.decode(list(tokens), skip_special_tokens=True),
            detokenizer=NaiveStreamingDetokenizer(codec),
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
            encoded = await self._encode_request(request)
            if span is not None:
                span.prompt_tokens = len(encoded.token_ids)
                span.path = "vlm" if encoded.vision is not None else "text"
                metrics.record(
                    "request_admitted",
                    request_id=request.request_id,
                    operation=operation,
                    path=span.path,
                    prompt_tokens=span.prompt_tokens,
                )
            if emit is not None and encoded.sampling is not None:
                result = await self._stream(encoded, emit)
            else:
                result = await self._complete(encoded)
            if span is not None and result is not None:
                span.finish_reason = result.finish_reason
                span.completion_tokens = result.completion_tokens
            return result

    async def _encode_request(self, request: PromptRequest) -> TokenizedRequest:
        if isinstance(request.prompt, str):
            encoded = await asyncio.to_thread(self._encode, request.prompt)
        elif self._encode_chat is not None:
            encoded = await asyncio.to_thread(self._encode_chat, request.prompt)
        else:
            raise InvalidPrompt("Chat is unavailable")

        if isinstance(encoded, VisionPrompt):
            return TokenizedRequest(request.request_id, (), request.sampling, encoded)
        return TokenizedRequest(request.request_id, tuple(encoded), request.sampling)

    async def _complete(self, request: TokenizedRequest) -> Generation | None:
        output = await self._scheduler.run(request)
        if request.sampling is None:
            return None
        if output is None or self._decode is None:
            raise RuntimeError("Generation is unavailable")
        text = await asyncio.to_thread(self._decode, output.token_ids)
        return self._generation(request, output, text)

    async def _stream(self, request: TokenizedRequest, emit: EmitDelta) -> Generation:
        if self._detokenizer is None:
            raise InvalidPrompt("Streaming is unavailable")
        decoder = copy(self._detokenizer)
        decoder.reset()
        terminal: OutputDelta | None = None

        async def decode_delta(delta: OutputDelta) -> None:
            nonlocal terminal
            if terminal is not None:
                raise InferenceFailed(
                    "Received a worker delta after the terminal delta"
                )
            if delta.finish_reason is not None:
                terminal = delta
                return
            if delta.token_id in self._special_tokens:
                return
            decoder.add_token(delta.token_id)
            if text := decoder.last_segment:
                await emit(replace(delta, token_text=text))

        output = await self._scheduler.run(request, decode_delta)
        if output is None or terminal is None:
            raise InferenceFailed("Worker stream ended without a terminal delta")
        if (
            terminal.output_token_ids != output.token_ids
            or terminal.finish_reason != output.finish_reason
        ):
            raise InferenceFailed(
                "Worker stream returned an inconsistent terminal delta"
            )
        decoder.finalize()
        await emit(replace(terminal, token_text=decoder.last_segment))
        return self._generation(request, output, decoder.text)

    @staticmethod
    def _generation(
        request: TokenizedRequest, output: GeneratedTokens, text: str
    ) -> Generation:
        return Generation(
            request.request_id,
            text,
            output.prompt_tokens
            if output.prompt_tokens is not None
            else len(request.token_ids),
            len(output.token_ids),
            output.finish_reason,
            output.token_ids,
        )
