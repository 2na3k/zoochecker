import unittest

import httpx
from fastapi import FastAPI
from openai import AsyncOpenAI

from zoochecker.api.v1.service import create_app, create_router
from zoochecker.server_args import ServerArgs
from zoochecker.types import Generation, InvalidPrompt, OutputDelta


class TokenizerClientStub:
    async def submit(self, prompt):
        raise InvalidPrompt("Empty prompt")

    async def generate(
        self, sampling, *, request_id="test", emit=None, prompt=None, messages=None
    ):
        token_ids = () if prompt == "empty" else (41, 42)
        text = " answer" if token_ids else ""
        if emit is not None:
            if token_ids:
                await emit(OutputDelta(request_id, text, token_ids[-1]))
            await emit(
                OutputDelta(request_id, "", -1, "stop", 3, len(token_ids), token_ids)
            )
        return Generation(request_id, text, 3, len(token_ids), "stop", token_ids)


class CompletionSchemaTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.state.tokenizer_client = TokenizerClientStub()
        self.app.state.image_size = None
        self.app.state.server_args = ServerArgs("test-model", None)
        self.app.include_router(create_router())

    def test_public_routes_are_registered_under_v1(self):
        paths = create_app(ServerArgs("test-model", None)).openapi()["paths"]
        self.assertEqual(
            set(paths),
            {
                "/v1/models",
                "/v1/completions",
                "/v1/chat/completions",
                "/v1/prefill",
                "/v1/health",
                "/v1/stats",
                "/v1/events",
                "/v1/metrics",
            },
        )

    async def test_validation_and_prefill_errors_share_the_error_schema(self):
        app = create_app(ServerArgs("test-model", None))
        app.state.tokenizer_client = TokenizerClientStub()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            invalid = await client.post(
                "/v1/completions", json={"prompt": "text", "max_tokens": 0}
            )
            prefill = await client.post("/v1/prefill", json={"prompt": ""})
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(prefill.status_code, 400)
        for response in (invalid, prefill):
            self.assertEqual(set(response.json()), {"error"})
            self.assertEqual(response.json()["error"]["type"], "invalid_request_error")
        self.assertEqual(invalid.json()["error"]["param"], "max_tokens")

    def sdk(self):
        return AsyncOpenAI(
            api_key="test",
            base_url="http://test/v1",
            http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app)),
        )

    async def test_sdk_reads_output_token_ids_from_both_completion_routes(self):
        async with self.sdk() as sdk:
            text = await sdk.completions.create(model="test-model", prompt="test")
            chat = await sdk.chat.completions.create(
                model="test-model", messages=[{"role": "user", "content": "test"}]
            )
            empty = await sdk.completions.create(model="test-model", prompt="empty")

        self.assertEqual(text.choices[0].text, " answer")
        self.assertEqual(chat.choices[0].message.content, " answer")
        for result in (text, chat):
            self.assertEqual(result.model_extra["output_token_ids"], [41, 42])
            self.assertEqual(result.usage.completion_tokens, 2)
        self.assertEqual(empty.model_extra["output_token_ids"], [])
        self.assertEqual(empty.usage.completion_tokens, 0)

    async def test_sdk_reads_stream_ids_only_on_the_finish_chunk(self):
        async with self.sdk() as sdk:
            chat = await sdk.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                stream=True,
                stream_options={"include_usage": True},
            )
            chat_frames = [frame async for frame in chat]
            text = await sdk.completions.create(
                model="test-model", prompt="test", stream=True
            )
            text_frames = [frame async for frame in text]
            empty = await sdk.completions.create(
                model="test-model", prompt="empty", stream=True
            )
            empty_frames = [frame async for frame in empty]

        self.assertEqual(chat_frames[0].choices[0].delta.role, "assistant")
        self.assertEqual(chat_frames[1].choices[0].delta.content, " answer")
        self.assertEqual(text_frames[0].choices[0].text, " answer")
        for frames in (chat_frames, text_frames):
            with_ids = [
                frame for frame in frames if "output_token_ids" in frame.model_extra
            ]
            self.assertEqual(len(with_ids), 1)
            self.assertEqual(with_ids[0].choices[0].finish_reason, "stop")
            self.assertEqual(with_ids[0].model_extra["output_token_ids"], [41, 42])
        self.assertEqual(chat_frames[-1].choices, [])
        self.assertEqual(chat_frames[-1].usage.completion_tokens, 2)
        self.assertIsNone(text_frames[-1].usage)
        self.assertEqual(len(empty_frames), 1)
        self.assertEqual(empty_frames[0].choices[0].finish_reason, "stop")
        self.assertEqual(empty_frames[0].model_extra["output_token_ids"], [])
