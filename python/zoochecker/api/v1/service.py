# would be the single file to register all of the inference class for this thing
import asyncio
import json
from time import time
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from zoochecker.images import ImagePart, prepare_chat_images
from zoochecker.server_args import ServerArgs
from zoochecker.ipc_connector import TokenizerClient
from zoochecker.types import (
    Generation,
    InferenceFailed,
    InvalidPrompt,
    OutputDelta,
    Sampling,
)


class StreamOptions(BaseModel):
    include_usage: bool = False


class _CompletionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: Literal[1] = 1


class CompletionRequest(_CompletionOptions):
    prompt: str


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[Annotated[TextPart | ImagePart, Field(discriminator="type")]]


class ChatCompletionRequest(_CompletionOptions):
    messages: list[ChatMessage] = Field(min_length=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)


def _error(status: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


def _usage(result: Generation | OutputDelta) -> dict[str, int]:
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_completion(
    request: Request,
    client: TokenizerClient,
    sampling: Sampling,
    inputs: dict,
    model: str,
    *,
    chat: bool,
    include_usage: bool,
) -> Response:
    request_id = uuid4().hex
    queue: asyncio.Queue[OutputDelta | InvalidPrompt | InferenceFailed] = asyncio.Queue(
        maxsize=1
    )
    base = {
        "id": f"{'chatcmpl' if chat else 'cmpl'}-{request_id}",
        "object": "chat.completion.chunk" if chat else "text_completion",
        "created": int(time()),
        "model": model,
    }
    if include_usage:
        base["usage"] = None

    async def produce() -> None:
        try:
            await client.generate(
                sampling, **inputs, request_id=request_id, emit=queue.put
            )
        except (InvalidPrompt, InferenceFailed) as exc:
            await queue.put(exc)
        except Exception:
            await queue.put(InferenceFailed("Inference failed"))

    task = asyncio.create_task(produce())

    async def close() -> None:
        if not task.done() and not task.cancelling():
            task.cancel()
        await asyncio.shield(asyncio.gather(task, return_exceptions=True))

    async def wait_for_disconnect() -> None:
        while (await request.receive())["type"] != "http.disconnect":
            pass

    disconnected = asyncio.create_task(wait_for_disconnect())
    first_frame = asyncio.create_task(queue.get())
    try:
        done, _ = await asyncio.wait(
            (disconnected, first_frame), return_when=asyncio.FIRST_COMPLETED
        )
        if disconnected in done:
            await close()
            return Response(status_code=499)
        first = first_frame.result()
    except BaseException:
        await close()
        raise
    finally:
        disconnected.cancel()
        first_frame.cancel()
        await asyncio.gather(disconnected, first_frame, return_exceptions=True)
    if isinstance(first, (InvalidPrompt, InferenceFailed)):
        await close()
        if isinstance(first, InvalidPrompt):
            return _error(400, str(first), "invalid_request_error")
        return _error(500, str(first), "server_error")

    def chunk(content: str, reason: str | None = None) -> str:
        if chat:
            choice = {"delta": {"content": content} if content else {}}
        else:
            choice = {"text": content, "logprobs": None}
        return _sse(
            {**base, "choices": [{"index": 0, **choice, "finish_reason": reason}]}
        )

    async def events():
        try:
            if chat:
                yield _sse(
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            item = first
            while isinstance(item, OutputDelta) and item.finish_reason is None:
                yield chunk(item.token_text)
                item = await queue.get()
            if isinstance(item, OutputDelta):
                if item.token_text:
                    yield chunk(item.token_text)
                yield chunk("", item.finish_reason)
                if include_usage:
                    yield _sse({**base, "choices": [], "usage": _usage(item)})
            else:
                error_type = (
                    "invalid_request_error"
                    if isinstance(item, InvalidPrompt)
                    else "server_error"
                )
                yield _sse({"error": {"message": str(item), "type": error_type}})
            yield "data: [DONE]\n\n"
        finally:
            await close()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        background=BackgroundTask(close),
    )


def create_router(args: ServerArgs) -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.get("/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": args.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    async def complete(
        body: CompletionRequest | ChatCompletionRequest, request: Request
    ) -> Response:
        if body.model is not None and body.model != args.model_name:
            return _error(
                404, f"Model {body.model!r} is unavailable", "invalid_request_error"
            )
        chat = isinstance(body, ChatCompletionRequest)
        max_tokens = body.max_tokens
        if chat and body.max_completion_tokens is not None:
            max_tokens = body.max_completion_tokens
        sampling = Sampling(
            args.max_new_tokens if max_tokens is None else max_tokens,
            args.temperature if body.temperature is None else body.temperature,
            args.top_p if body.top_p is None else body.top_p,
        )
        client: TokenizerClient = request.app.state.tokenizer_client
        if chat:
            try:
                messages = await prepare_chat_images(
                    [message.model_dump() for message in body.messages],
                    request.app.state.image_size,
                    args,
                )
            except InvalidPrompt as exc:
                return _error(400, str(exc), "invalid_request_error")
            inputs = {"messages": messages}
        else:
            inputs = {"prompt": body.prompt}
        if body.stream:
            return await _stream_completion(
                request,
                client,
                sampling,
                inputs,
                args.model_name,
                chat=chat,
                include_usage=body.stream_options is not None
                and body.stream_options.include_usage,
            )
        try:
            result = await client.generate(sampling, **inputs)
        except InvalidPrompt as exc:
            return _error(400, str(exc), "invalid_request_error")
        except InferenceFailed as exc:
            return _error(500, str(exc), "server_error")

        choice = {"index": 0, "logprobs": None, "finish_reason": result.finish_reason}
        if chat:
            choice["message"] = {"role": "assistant", "content": result.text}
        else:
            choice["text"] = result.text

        prefix = "chatcmpl" if chat else "cmpl"
        return JSONResponse(
            {
                "id": f"{prefix}-{result.request_id}",
                "object": "chat.completion" if chat else "text_completion",
                "created": int(time()),
                "model": args.model_name,
                "choices": [choice],
                "usage": _usage(result),
            }
        )

    @router.post("/completions")
    async def completion(body: CompletionRequest, request: Request) -> Response:
        return await complete(body, request)

    @router.post("/chat/completions")
    async def chat_completion(
        body: ChatCompletionRequest, request: Request
    ) -> Response:
        return await complete(body, request)

    return router
