import asyncio
import json
import logging
import sys
from collections import Counter, deque
from collections.abc import AsyncIterator, Iterator
from contextlib import ExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass
from math import isfinite
from time import monotonic, time
from typing import TextIO

import mlx.core as mx
from prometheus_client import CollectorRegistry, Gauge, Histogram, generate_latest
from prometheus_client import Counter as PrometheusCounter

from zoochecker.cache import KVCacheStats
from zoochecker.server_args import ServerArgs
from zoochecker.types import (
    CacheStats,
    InvalidPrompt,
    MLXMemoryStats,
    SchedulerStats,
    StatsResponse,
    ThroughputStats,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RequestSpan:
    started_at: float
    path: str = "text"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "prefill"
    first_token_at: float | None = None
    last_token_at: float | None = None
    prefill_duration_seconds: float | None = None


class Metrics:
    def __init__(self, args: ServerArgs) -> None:
        if (
            min(
                args.telemetry_event_queue_size,
                args.telemetry_recent_events,
                args.telemetry_latency_samples,
            )
            < 1
        ):
            raise ValueError(
                "Telemetry queue, history, and sample sizes must be positive"
            )
        if (
            not isfinite(args.telemetry_memory_poll_interval)
            or args.telemetry_memory_poll_interval <= 0
        ):
            raise ValueError("Telemetry memory poll interval must be positive")
        self._args = args
        self._started = monotonic()
        self._queue: asyncio.Queue[dict] = asyncio.Queue(
            args.telemetry_event_queue_size
        )
        self._events: deque[dict] = deque(maxlen=args.telemetry_recent_events)
        self._spans: dict[str, RequestSpan] = {}
        self._totals: Counter[str] = Counter()
        self._memory = dict.fromkeys(("active_bytes", "cache_bytes", "peak_bytes"), 0)
        self._samples = {
            name: deque(maxlen=args.telemetry_latency_samples)
            for name in (
                "total",
                "ttft",
                "tpot",
                "prefill",
                "decode_step",
                "queue_wait",
            )
        }

        self._registry = CollectorRegistry()
        self._counters = {
            name: PrometheusCounter(
                f"zoochecker_{name}", description, labels, registry=self._registry
            )
            for name, description, labels in (
                (
                    "requests_total",
                    "Completed pipeline requests",
                    ("model", "finish_reason"),
                ),
                (
                    "prompt_tokens_total",
                    "Prompt tokens processed by completed requests",
                    ("model",),
                ),
                (
                    "completion_tokens_total",
                    "Output tokens from completed requests",
                    ("model",),
                ),
                ("errors_total", "Internal pipeline failures", ("model",)),
                (
                    "step_loop_iterations_total",
                    "Completed model decode steps",
                    ("model",),
                ),
                (
                    "telemetry_dropped_events_total",
                    "Events dropped when the telemetry queue was full",
                    (),
                ),
                (
                    "http_requests_total",
                    "HTTP requests handled",
                    ("method", "endpoint", "status_code"),
                ),
            )
        }
        self._histograms = {
            key: Histogram(
                f"zoochecker_{name}_seconds",
                description,
                ("model",),
                registry=self._registry,
            )
            for key, name, description in (
                ("total", "request_latency", "Pipeline request latency"),
                (
                    "ttft",
                    "ttft",
                    "Time from tokenizer receipt to the first generated token",
                ),
                ("tpot", "tpot", "Time between generated output tokens"),
                ("prefill", "prefill_duration", "Model prompt prefill latency"),
                ("decode_step", "decode_step_duration", "Model decode forward latency"),
                ("queue_wait", "queue_wait", "Time waiting in the scheduler queue"),
            )
        }
        self._http_latency = Histogram(
            "zoochecker_http_request_duration_seconds",
            "HTTP request duration",
            ("method", "endpoint", "status_code"),
            registry=self._registry,
        )
        self._gauges = {
            name: Gauge(
                f"zoochecker_{name}", description, ("model",), registry=self._registry
            )
            for name, description in (
                ("active_requests", "Active tokenizer requests"),
                ("scheduler_waiting_seqs", "Requests waiting in the scheduler"),
                ("scheduler_running_seqs", "Requests dispatched by the scheduler"),
                ("kvcache_slots_total", "KV cache lease capacity"),
                ("kvcache_slots_used", "Active KV cache leases"),
                ("kvcache_blocks_total", "KV cache block capacity"),
                ("kvcache_blocks_used", "Allocated KV cache blocks"),
                ("kvcache_blocks_free", "Free KV cache blocks"),
                ("kvcache_utilisation_ratio", "Fraction of KV cache blocks allocated"),
                ("kvcache_bytes_total", "KV cache byte capacity"),
                ("kvcache_bytes_used", "Bytes of valid KV tensors"),
                ("mlx_active_bytes", "Active MLX allocator bytes"),
                ("mlx_cache_bytes", "Cached MLX allocator bytes"),
                ("mlx_peak_bytes", "Peak MLX allocator bytes"),
            )
        }
        for name in (
            "prompt_tokens_total",
            "completion_tokens_total",
            "errors_total",
            "step_loop_iterations_total",
        ):
            self._counters[name].labels(model=args.model_name)

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        with ExitStack() as stack:
            sink = self._args.telemetry_json_log
            log = None
            if sink in ("stdout", "stderr"):
                log = sys.stdout if sink == "stdout" else sys.stderr
            elif sink:
                try:
                    log = stack.enter_context(
                        open(sink, "a", encoding="utf-8", buffering=1)
                    )
                except OSError:
                    logger.exception("Could not open telemetry JSON log")
            async with asyncio.TaskGroup() as tasks:
                drain = tasks.create_task(self._drain(log), name="telemetry-drain")
                memory = tasks.create_task(self._poll_memory(), name="telemetry-memory")
                try:
                    yield
                finally:
                    memory.cancel()
                    if not drain.done():
                        await self.flush()
                    drain.cancel()

    def record(self, event_type: str, **data: object) -> None:
        try:
            self._queue.put_nowait(
                {
                    "event_type": event_type,
                    "timestamp": monotonic(),
                    "wall_time": time(),
                    **data,
                }
            )
        except asyncio.QueueFull:
            self._totals["telemetry_dropped_events_total"] += 1
            self._counters["telemetry_dropped_events_total"].inc()

    @contextmanager
    def request(self, request_id: str, operation: str) -> Iterator[RequestSpan]:
        span = RequestSpan(monotonic())
        self._spans[request_id] = span
        try:
            yield span
        except InvalidPrompt:
            span.finish_reason = "rejected"
            raise
        except asyncio.CancelledError:
            span.finish_reason = "cancelled"
            raise
        except Exception:
            span.finish_reason = "error"
            raise
        finally:
            self._spans.pop(request_id, None)
            self.record(
                "request_finished",
                request_id=request_id,
                operation=operation,
                path=span.path,
                model=self._args.model_name,
                finish_reason=span.finish_reason,
                prompt_tokens=span.prompt_tokens,
                completion_tokens=span.completion_tokens,
                total_latency_seconds=monotonic() - span.started_at,
                ttft_seconds=None
                if span.first_token_at is None
                else span.first_token_at - span.started_at,
                prefill_duration_seconds=span.prefill_duration_seconds,
            )

    def prompt_prepared(self, request_id: str, prompt_tokens: int) -> None:
        span = self._spans.get(request_id)
        if span is not None:
            span.prompt_tokens = prompt_tokens

    def prefill_finished(self, request_id: str, duration: float) -> None:
        span = self._spans.get(request_id)
        if span is not None:
            span.prefill_duration_seconds = duration
        self.record("prefill_end", request_id=request_id, duration_seconds=duration)

    def token_generated(self, request_id: str) -> None:
        span = self._spans.get(request_id)
        if span is None:
            return
        now = monotonic()
        if span.last_token_at is None:
            span.first_token_at = now
            self.record(
                "first_token", request_id=request_id, ttft_seconds=now - span.started_at
            )
        else:
            self.record(
                "token_generated",
                request_id=request_id,
                tpot_seconds=now - span.last_token_at,
            )
        span.last_token_at = now
        span.completion_tokens += 1

    async def flush(self) -> None:
        await self._queue.join()

    def recent_events(self, limit: int) -> list[dict]:
        return list(self._events)[-limit:] if limit else []

    def snapshot(
        self,
        waiting: int,
        running: int,
        cache: KVCacheStats,
        *,
        ready: bool,
    ) -> StatsResponse:
        def percentiles(name: str) -> tuple[float | None, float | None]:
            ordered = sorted(self._samples[name])
            if not ordered:
                return None, None
            size = len(ordered)
            return ordered[size * 50 // 100], ordered[min(size * 99 // 100, size - 1)]

        total_p50, total_p99 = percentiles("total")
        ttft_p50, ttft_p99 = percentiles("ttft")
        tpot_p50, tpot_p99 = percentiles("tpot")
        prefill_p50, prefill_p99 = percentiles("prefill")
        decode_p50, decode_p99 = percentiles("decode_step")
        queue_p50, queue_p99 = percentiles("queue_wait")
        return StatsResponse(
            status="ok" if ready else "unavailable",
            timestamp=time(),
            uptime_seconds=monotonic() - self._started,
            model=self._args.model_name,
            active_requests=len(self._spans),
            scheduler=SchedulerStats(waiting_seqs=waiting, running_seqs=running),
            kvcache_text=CacheStats(
                slots_total=cache.slots_total,
                slots_used=cache.slots_used,
                blocks_total=cache.blocks_total,
                blocks_used=cache.blocks_used,
                blocks_free=cache.blocks_total - cache.blocks_used,
                block_tokens=cache.block_tokens,
                bytes_total=cache.bytes_total,
                bytes_used=cache.bytes_used,
                utilisation_ratio=cache.blocks_used / cache.blocks_total,
            ),
            throughput=ThroughputStats(
                requests_total=self._totals["requests_total"],
                prompt_tokens_total=self._totals["prompt_tokens_total"],
                completion_tokens_total=self._totals["completion_tokens_total"],
                step_loop_iterations_total=self._totals["step_loop_iterations_total"],
            ),
            latency_p50_total_seconds=total_p50,
            latency_p99_total_seconds=total_p99,
            latency_p50_ttft_seconds=ttft_p50,
            latency_p99_ttft_seconds=ttft_p99,
            latency_p50_tpot_seconds=tpot_p50,
            latency_p99_tpot_seconds=tpot_p99,
            latency_p50_prefill_seconds=prefill_p50,
            latency_p99_prefill_seconds=prefill_p99,
            latency_p50_decode_step_seconds=decode_p50,
            latency_p99_decode_step_seconds=decode_p99,
            latency_p50_queue_wait_seconds=queue_p50,
            latency_p99_queue_wait_seconds=queue_p99,
            mlx_memory=MLXMemoryStats(**self._memory),
            http_requests_total=self._totals["http_requests_total"],
            errors_total=self._totals["errors_total"],
            telemetry_dropped_events_total=self._totals[
                "telemetry_dropped_events_total"
            ],
        )

    def prometheus(self, snapshot: StatsResponse) -> bytes:
        scheduler = snapshot.scheduler
        cache = snapshot.kvcache_text
        memory = snapshot.mlx_memory
        gauges = {
            "active_requests": snapshot.active_requests,
            "scheduler_waiting_seqs": scheduler.waiting_seqs,
            "scheduler_running_seqs": scheduler.running_seqs,
            "kvcache_slots_total": cache.slots_total,
            "kvcache_slots_used": cache.slots_used,
            "kvcache_blocks_total": cache.blocks_total,
            "kvcache_blocks_used": cache.blocks_used,
            "kvcache_blocks_free": cache.blocks_free,
            "kvcache_utilisation_ratio": cache.utilisation_ratio,
            "kvcache_bytes_total": cache.bytes_total,
            "kvcache_bytes_used": cache.bytes_used,
            "mlx_active_bytes": memory.active_bytes,
            "mlx_cache_bytes": memory.cache_bytes,
            "mlx_peak_bytes": memory.peak_bytes,
        }
        for name, value in gauges.items():
            self._gauges[name].labels(model=self._args.model_name).set(value)
        return generate_latest(self._registry)

    async def _drain(self, log: TextIO | None) -> None:
        while True:
            event = await self._queue.get()
            try:
                self._events.append(event)
                self._aggregate(event)
                if log is not None:
                    try:
                        log.write(json.dumps(event) + "\n")
                    except OSError:
                        logger.exception("Could not write telemetry JSON log")
                        log = None
            finally:
                self._queue.task_done()
            if not self._queue.empty():
                await asyncio.sleep(0)

    def _observe(self, name: str, duration: float) -> None:
        self._samples[name].append(duration)
        self._histograms[name].labels(model=self._args.model_name).observe(duration)

    def _increment(self, name: str, amount: int = 1, **labels: str) -> None:
        self._totals[name] += amount
        self._counters[name].labels(**labels).inc(amount)

    def _aggregate(self, event: dict) -> None:
        event_type = event["event_type"]
        model = self._args.model_name
        if event_type == "request_finished":
            reason = event["finish_reason"]
            self._increment("requests_total", model=model, finish_reason=reason)
            if reason in ("stop", "length", "prefill"):
                for name in ("prompt_tokens", "completion_tokens"):
                    self._increment(f"{name}_total", event[name], model=model)
            elif reason == "error":
                self._increment("errors_total", model=model)
            self._observe("total", event["total_latency_seconds"])
        elif event_type == "first_token":
            self._observe("ttft", event["ttft_seconds"])
        elif event_type == "token_generated":
            self._observe("tpot", event["tpot_seconds"])
        elif event_type == "prefill_end":
            self._observe("prefill", event["duration_seconds"])
        elif event_type == "decode_step":
            self._increment("step_loop_iterations_total", model=model)
            self._observe("decode_step", event["duration_seconds"])
        elif event_type == "request_dispatched":
            self._observe("queue_wait", event["queue_wait_seconds"])
        elif event_type == "http_request":
            labels = {
                name: str(event[name]) for name in ("method", "endpoint", "status_code")
            }
            self._increment("http_requests_total", **labels)
            self._http_latency.labels(**labels).observe(event["duration_seconds"])
        elif event_type == "mlx_memory":
            self._memory = {name: event[name] for name in self._memory}

    async def _poll_memory(self) -> None:
        while True:
            try:
                memory = await asyncio.to_thread(_mlx_memory)
                self.record("mlx_memory", **memory)
            except RuntimeError:
                logger.debug("Could not sample MLX memory", exc_info=True)
            await asyncio.sleep(self._args.telemetry_memory_poll_interval)


def _mlx_memory() -> dict[str, int]:
    return {
        "active_bytes": mx.get_active_memory(),
        "cache_bytes": mx.get_cache_memory(),
        "peak_bytes": mx.get_peak_memory(),
    }
