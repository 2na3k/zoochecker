import asyncio
import base64
import json
import os
import tempfile
import unittest
from contextlib import closing
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import httpx
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from PIL import Image

from zoochecker.cache import UnifiedCacheManager
from zoochecker.api.v1.service import create_app
from zoochecker.models.qwen35 import Qwen35
from zoochecker.server_args import ServerArgs
from zoochecker.tokenizer import Tokenizer


DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2] / "draft/models/Qwen3.5-0.8B-MLX-4bit"
)
MODEL = Path(os.environ.get("ZOOCHECKER_QWEN35_TEST_MODEL", DEFAULT_MODEL))
POOLS = ("mlx_kv", "mlx_paged_kv")


@unittest.skipUnless(mx.metal.is_available(), "requires MLX Metal")
@unittest.skipUnless(
    (MODEL / "config.json").is_file(), "requires a local Qwen 3.5 checkpoint"
)
class Qwen35E2ETests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream, cls.codec = load(str(MODEL))

    @classmethod
    def tearDownClass(cls):
        del cls.upstream, cls.codec

    async def test_image_requests_and_streamed_text_match_nonstreaming(self):
        config = json.loads((MODEL / "config.json").read_text())
        if not config.get("vision_config"):
            self.skipTest("requires a Qwen vision checkpoint")

        received_sizes = []
        process = Tokenizer.process

        async def inspect_image(tokenizer, request, emit=None):
            if isinstance(request.prompt, list):
                for message in request.prompt:
                    if isinstance(message.content, str):
                        continue
                    for part in message.content:
                        if part.type == "image_url":
                            data = part.image_url.url.split(",", 1)[1]
                            with Image.open(BytesIO(base64.b64decode(data))) as image:
                                received_sizes.append(image.size)
            return await process(tokenizer, request, emit)

        args = ServerArgs(
            "qwen-test",
            str(MODEL),
            cache_pool_type="mlx_paged_kv",
            prefill_step_size=16,
        )
        app = create_app(args)
        self.enterContext(patch.object(Tokenizer, "process", inspect_image))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
            asyncio.timeout(30),
        ):
            for color in ("red", "blue"):
                with self.subTest(color=color):
                    image = BytesIO()
                    size = (71, 93)
                    expected_size = app.state.image_size(*size)
                    self.assertNotEqual(size, expected_size)
                    Image.new("RGB", size, color).save(image, format="PNG")
                    source = (
                        "data:image/png;base64,"
                        + base64.b64encode(image.getvalue()).decode()
                    )
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": source}},
                                {
                                    "type": "text",
                                    "text": "What color is this image? Answer in one word.",
                                },
                            ],
                        }
                    ]
                    body = {"messages": messages, "max_completion_tokens": 10}
                    result = (
                        (await client.post("/v1/chat/completions", json=body))
                        .raise_for_status()
                        .json()
                    )
                    self.assertIn(
                        color, result["choices"][0]["message"]["content"].lower()
                    )
                    self.assertGreater(
                        result["usage"]["prompt_tokens"], args.prefill_step_size
                    )
                    await self._assert_stream(
                        client, "/v1/chat/completions", body, result
                    )
                    self.assertEqual(
                        received_sizes[-2:], [expected_size, expected_size]
                    )

            invalid = await client.post(
                "/v1/chat/completions",
                json={
                    "stream": True,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "http://127.0.0.1/image.png"},
                                },
                                {"type": "text", "text": "Describe this image."},
                            ],
                        }
                    ],
                },
            )
            self.assertEqual(invalid.status_code, 400)
            body = {"prompt": "The capital of France is", "max_tokens": 8}
            result = (
                (await client.post("/v1/completions", json=body))
                .raise_for_status()
                .json()
            )
            await self._assert_stream(client, "/v1/completions", body, result)

    async def _assert_stream(self, client, route, body, result):
        response = await client.post(
            route,
            json={**body, "stream": True, "stream_options": {"include_usage": True}},
        )
        response.raise_for_status()
        self.assertIn("text/event-stream", response.headers["content-type"])
        frames = [
            line.removeprefix("data: ")
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(frames.pop(), "[DONE]")
        chunks = [json.loads(frame) for frame in frames]
        self.assertEqual(chunks[-1]["usage"], result["usage"])
        terminal = [chunk for chunk in chunks if "output_token_ids" in chunk]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["output_token_ids"], result["output_token_ids"])
        self.assertEqual(len({chunk["id"] for chunk in chunks}), 1)
        choices = [chunk["choices"][0] for chunk in chunks if chunk["choices"]]
        final = result["choices"][0]
        self.assertEqual(choices[-1]["finish_reason"], final["finish_reason"])
        if "message" in final:
            self.assertEqual(choices[0]["delta"]["role"], "assistant")
            text = "".join(choice["delta"].get("content", "") for choice in choices)
            self.assertEqual(text, final["message"]["content"])
        else:
            self.assertEqual(
                "".join(choice["text"] for choice in choices), final["text"]
            )

    def test_chunked_prefill_and_decode_match_native_attention(self):
        model = Qwen35(self.upstream)
        num_layers, num_kv_heads, head_dim = model.cache_geometry
        tokens = self.codec.encode(
            "The capital of France is Paris, and the capital of Italy is Rome."
        )
        native = self.upstream.make_cache()
        expected = []
        chunks = (tokens[:5], tokens[5:-1], tokens[-1:])
        for chunk in chunks:
            scores = self.upstream(mx.array([chunk]), cache=native)[:, -1, :]
            mx.eval(scores)
            expected.append(scores)

        for pool in POOLS:
            with self.subTest(cache_pool_type=pool):
                args = ServerArgs(
                    "qwen-test",
                    str(MODEL),
                    cache_pool_type=pool,
                    cache_max_tokens=len(tokens),
                    cache_block_tokens=4,
                )
                manager = UnifiedCacheManager(
                    args,
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )
                with closing(manager), manager.request_cache() as lease:
                    cache = model.make_cache(lease)
                    offset = 0
                    for chunk, reference in zip(chunks, expected, strict=True):
                        actual = model.forward(mx.array([chunk]), cache)
                        self.assertTrue(
                            mx.allclose(actual, reference, atol=0.08, rtol=0.02).item()
                        )
                        self.assertEqual(
                            mx.argmax(actual).item(), mx.argmax(reference).item()
                        )
                        offset += len(chunk)
                        self.assertEqual(
                            [lease.offset(i) for i in range(num_layers)],
                            [offset] * num_layers,
                        )

    async def test_openai_requests_use_cache_and_return_detokenized_text(self):
        prompt = "The capital of France is"
        messages = [
            {
                "role": "user",
                "content": "Name the capital of France. Answer in one word.",
            }
        ]
        chat_prompt = self.codec.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        text_ids = self.codec.encode(prompt)
        chat_ids = self.codec.encode(chat_prompt, add_special_tokens=False)
        routes = (
            ("/v1/completions", {"prompt": prompt, "max_tokens": 8}, text_ids, 8),
            (
                "/v1/chat/completions",
                {"messages": messages, "max_completion_tokens": 12},
                chat_ids,
                12,
            ),
        )
        baseline = []
        for _, _, prompt_ids, limit in routes:
            reference = []
            for token, _ in generate_step(
                mx.array(prompt_ids), self.upstream, max_tokens=limit
            ):
                if token in self.codec.eos_token_ids:
                    break
                reference.append(token)
            baseline.append(
                (self.codec.decode(reference, skip_special_tokens=True), reference)
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            for pool in POOLS:
                with self.subTest(cache_pool_type=pool):
                    log = Path(temp_dir) / f"{pool}.jsonl"
                    args = ServerArgs(
                        "qwen-test",
                        str(MODEL),
                        cache_pool_type=pool,
                        cache_block_tokens=4,
                        cache_max_tokens=max(len(text_ids), len(chat_ids)) + 12,
                        prefill_step_size=4,
                        telemetry_json_log=str(log),
                    )
                    app = create_app(args)
                    async with (
                        app.router.lifespan_context(app),
                        httpx.AsyncClient(
                            transport=httpx.ASGITransport(app=app),
                            base_url="http://test",
                        ) as client,
                        asyncio.timeout(30),
                    ):
                        models = (await client.get("/v1/models")).json()
                        self.assertEqual(models["data"][0]["id"], args.model_name)
                        oversized = await client.post(
                            "/v1/completions",
                            json={"prompt": "word " * args.cache_max_tokens},
                        )
                        self.assertEqual(oversized.status_code, 400)
                        terminal_status = {}
                        for (route, body, prompt_ids, _), (expected, output_ids) in zip(
                            routes, baseline, strict=True
                        ):
                            response = await client.post(
                                route, json={"model": args.model_name, **body}
                            )
                            response.raise_for_status()
                            completion = response.json()
                            choice = completion["choices"][0]
                            request_id = completion["id"].split("-", 1)[1]
                            terminal_status[request_id] = (
                                "finished_eos"
                                if choice["finish_reason"] == "stop"
                                else "finished_length"
                            )
                            text = (
                                choice["message"]["content"]
                                if "message" in choice
                                else choice["text"]
                            )
                            self.assertTrue(text.strip())
                            self.assertEqual(text, expected)
                            self.assertEqual(completion["output_token_ids"], output_ids)
                            self.assertEqual(
                                completion["usage"],
                                {
                                    "prompt_tokens": len(prompt_ids),
                                    "completion_tokens": len(output_ids),
                                    "total_tokens": len(prompt_ids) + len(output_ids),
                                },
                            )
                        stats = (await client.get("/v1/stats")).json()
                        throughput = stats["throughput"]
                        self.assertEqual(throughput["requests_total"], 3)
                        self.assertEqual(
                            throughput["prompt_tokens_total"],
                            len(text_ids) + len(chat_ids),
                        )
                        self.assertEqual(
                            throughput["completion_tokens_total"],
                            sum(len(output_ids) for _, output_ids in baseline),
                        )
                        self.assertGreater(throughput["step_loop_iterations_total"], 0)
                        self.assertGreater(stats["latency_p50_ttft_seconds"], 0)
                        self.assertGreater(stats["latency_p50_tpot_seconds"], 0)
                        self.assertGreater(stats["latency_p50_prefill_seconds"], 0)
                        self.assertGreater(stats["mlx_memory"]["active_bytes"], 0)
                        events = (await client.get("/v1/events?limit=200")).json()
                        statuses = {}
                        for event in events:
                            if event["event_type"] == "request_status":
                                statuses.setdefault(event["request_id"], []).append(
                                    event["status"]
                                )
                        for request_id, status in terminal_status.items():
                            self.assertEqual(
                                statuses[request_id], ["waiting", "running", status]
                            )
                        self.assertIn(
                            ["waiting", "running", "finished_rejected"],
                            statuses.values(),
                        )
                        event_types = {event["event_type"] for event in events}
                        self.assertTrue(
                            {
                                "first_token",
                                "prefill_end",
                                "request_finished",
                                "http_request",
                            }
                            <= event_types
                        )
                        metrics = await client.get("/v1/metrics")
                        metrics.raise_for_status()
                        self.assertIn(
                            'zoochecker_ttft_seconds_count{model="qwen-test"} 2.0',
                            metrics.text,
                        )
                        self.assertIn(
                            'zoochecker_kvcache_slots_used{model="qwen-test"} 0.0',
                            metrics.text,
                        )

                    records = [
                        json.loads(line) for line in log.read_text().splitlines()
                    ]
                    self.assertEqual(
                        sum(
                            event["event_type"] == "request_finished"
                            for event in records
                        ),
                        3,
                    )
