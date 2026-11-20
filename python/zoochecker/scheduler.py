import asyncio
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import TYPE_CHECKING

from zoochecker.ipc_connector import WorkerClient, WorkerReply
from zoochecker.types import (
    EmitDelta,
    GeneratedTokens,
    InferenceFailed,
    InvalidPrompt,
    SequenceStatus,
    TokenizedRequest,
)

if TYPE_CHECKING:
    from zoochecker.metrics import Metrics


@dataclass(slots=True)
class _ScheduledRequest:
    request: TokenizedRequest
    reply: asyncio.Future[GeneratedTokens | None]
    queued_at: float
    emit: EmitDelta | None = None
    dispatch: asyncio.Future[GeneratedTokens | None] | None = None
    status: SequenceStatus = SequenceStatus.WAITING


class Scheduler:
    def __init__(
        self,
        worker: WorkerClient,
        *,
        capacity: int,
        batch_wait_seconds: float = 0,
        metrics: "Metrics | None" = None,
    ) -> None:
        if capacity < 1 or not isfinite(batch_wait_seconds) or batch_wait_seconds < 0:
            raise ValueError("Invalid scheduler capacity or batch wait")
        self._worker = worker
        self._capacity = capacity
        self._batch_wait = batch_wait_seconds
        self._metrics = metrics
        self._pending: deque[_ScheduledRequest] = deque()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._scheduled: dict[
            asyncio.Future[GeneratedTokens | None], _ScheduledRequest
        ] = {}

    @property
    def ready(self) -> bool:
        return self._task is not None and not self._task.done() and self._worker.ready

    @property
    def counts(self) -> tuple[int, int]:
        waiting = len(self._pending)
        return waiting, len(self._scheduled) - waiting

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        if self._task is not None:
            raise RuntimeError("Scheduler is already running")
        self._wake = asyncio.Event()
        async with asyncio.TaskGroup() as tasks:
            task = tasks.create_task(self._run_loop())
            self._task = task
            try:
                yield
            finally:
                self._task = None
                task.cancel()
                while self._pending:
                    job = self._pending.popleft()
                    self._set_status(job, SequenceStatus.FINISHED_STOPPED)
                    job.reply.cancel()

    async def run(
        self, request: TokenizedRequest, emit: EmitDelta | None = None
    ) -> GeneratedTokens | None:
        if not self.ready:
            raise RuntimeError("Scheduler is not running")
        reply = asyncio.get_running_loop().create_future()
        job = _ScheduledRequest(request, reply, monotonic(), emit)
        self._scheduled[reply] = job
        self._set_status(job, SequenceStatus.WAITING)
        self._pending.append(job)
        self._wake.set()
        try:
            return await reply
        except asyncio.CancelledError:
            if job.status is SequenceStatus.WAITING:
                self._pending.remove(job)
                self._set_status(job, SequenceStatus.FINISHED_STOPPED)
            elif job.dispatch is not None:
                job.dispatch.cancel()
            raise

    def _set_status(self, job: _ScheduledRequest, status: SequenceStatus) -> None:
        if job.status.is_finished:
            return
        job.status = status
        if status.is_finished:
            self._scheduled.pop(job.reply)
            self._wake.set()
        if self._metrics is not None:
            self._metrics.record(
                "request_status", request_id=job.request.request_id, status=status.value
            )

    def _fail(self, job: _ScheduledRequest, exc: Exception) -> None:
        self._set_status(
            job,
            SequenceStatus.FINISHED_REJECTED
            if isinstance(exc, InvalidPrompt)
            else SequenceStatus.FINISHED_ERROR,
        )
        if not job.reply.done():
            job.reply.set_exception(exc)

    async def _run_loop(self) -> None:
        async with asyncio.TaskGroup() as active:
            while True:
                await self._wake.wait()
                self._wake.clear()
                _, running = self.counts
                available = self._capacity - running
                if not self._pending or available == 0:
                    continue
                if self._batch_wait and running == 0 and len(self._pending) < available:
                    await asyncio.sleep(self._batch_wait)
                _, running = self.counts
                jobs = [
                    self._pending.popleft()
                    for _ in range(min(len(self._pending), self._capacity - running))
                ]
                if not jobs:
                    continue
                for job in jobs:
                    self._set_status(job, SequenceStatus.RUNNING)
                    if self._metrics is not None:
                        self._metrics.record(
                            "request_dispatched",
                            request_id=job.request.request_id,
                            queue_wait_seconds=monotonic() - job.queued_at,
                            batch_size=len(jobs),
                        )
                try:
                    replies = await self._worker.submit(
                        [(job.request, job.emit is not None) for job in jobs]
                    )
                except asyncio.CancelledError:
                    for job in jobs:
                        self._set_status(job, SequenceStatus.FINISHED_STOPPED)
                        job.reply.cancel()
                    raise
                except Exception as exc:
                    for job in jobs:
                        self._fail(job, exc)
                    continue
                for job, reply in zip(jobs, replies, strict=True):
                    active.create_task(self._finish(job, reply))

    async def _finish(self, job: _ScheduledRequest, reply: WorkerReply) -> None:
        try:
            try:
                if job.reply.cancelled():
                    raise asyncio.CancelledError
                job.dispatch = asyncio.create_task(reply.result(job.emit))
                result = await job.dispatch
                if (result is None) != (job.request.sampling is None):
                    raise InferenceFailed(
                        "Worker result does not match the request operation"
                    )
            finally:
                await reply.close()
        except asyncio.CancelledError:
            self._set_status(job, SequenceStatus.FINISHED_STOPPED)
            job.reply.cancel()
        except Exception as exc:
            self._fail(job, exc)
        else:
            if result is None:
                status = SequenceStatus.FINISHED_PREFILL
            elif result.finish_reason == "stop":
                status = SequenceStatus.FINISHED_EOS
            else:
                status = SequenceStatus.FINISHED_LENGTH
            self._set_status(job, status)
            if not job.reply.done():
                job.reply.set_result(result)
