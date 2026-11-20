import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING

from zoochecker.types import (
    EmitDelta,
    GeneratedTokens,
    InvalidPrompt,
    SequenceStatus,
    TokenizedRequest,
)

if TYPE_CHECKING:
    from zoochecker.metrics import Metrics

Dispatch = Callable[
    [TokenizedRequest, EmitDelta | None], Awaitable[GeneratedTokens | None]
]


@dataclass(slots=True)
class _ScheduledRequest:
    request: TokenizedRequest
    reply: asyncio.Future[GeneratedTokens | None]
    queued_at: float
    emit: EmitDelta | None = None
    dispatch: asyncio.Future[GeneratedTokens | None] | None = None
    status: SequenceStatus = SequenceStatus.WAITING


class Scheduler:
    def __init__(self, dispatch: Dispatch, metrics: "Metrics | None" = None) -> None:
        self._dispatch = dispatch
        self._metrics = metrics
        self._pending: asyncio.Queue[_ScheduledRequest] | None = None
        self._task: asyncio.Task[None] | None = None
        self._scheduled: dict[
            asyncio.Future[GeneratedTokens | None], _ScheduledRequest
        ] = {}

    @property
    def ready(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def counts(self) -> tuple[int, int]:
        waiting = sum(
            job.status is SequenceStatus.WAITING for job in self._scheduled.values()
        )
        return waiting, len(self._scheduled) - waiting

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        if self._pending is not None:
            raise RuntimeError("Scheduler is already running")
        pending: asyncio.Queue[_ScheduledRequest] = asyncio.Queue()
        self._pending = pending
        async with asyncio.TaskGroup() as tasks:
            task = tasks.create_task(self._run_loop(pending))
            self._task = task
            try:
                yield
            finally:
                self._pending = None
                self._task = None
                task.cancel()
                while not pending.empty():
                    job = pending.get_nowait()
                    self._set_status(job, SequenceStatus.FINISHED_STOPPED)
                    job.reply.cancel()

    async def run(
        self, request: TokenizedRequest, emit: EmitDelta | None = None
    ) -> GeneratedTokens | None:
        if self._pending is None:
            raise RuntimeError("Scheduler is not running")
        reply = asyncio.get_running_loop().create_future()
        job = _ScheduledRequest(request, reply, monotonic(), emit)
        self._scheduled[reply] = job
        self._set_status(job, SequenceStatus.WAITING)
        self._pending.put_nowait(job)
        try:
            return await reply
        except asyncio.CancelledError:
            if job.status is SequenceStatus.WAITING:
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
        if self._metrics is not None:
            self._metrics.record(
                "request_status", request_id=job.request.request_id, status=status.value
            )

    async def _run_loop(self, pending: asyncio.Queue[_ScheduledRequest]) -> None:
        while True:
            job = await pending.get()
            request, reply = job.request, job.reply
            if reply.cancelled():
                self._set_status(job, SequenceStatus.FINISHED_STOPPED)
                continue
            self._set_status(job, SequenceStatus.RUNNING)
            try:
                if self._metrics is not None:
                    self._metrics.record(
                        "request_dispatched",
                        request_id=request.request_id,
                        queue_wait_seconds=monotonic() - job.queued_at,
                    )
                job.dispatch = asyncio.ensure_future(self._dispatch(request, job.emit))
                result = await job.dispatch
            except asyncio.CancelledError:
                self._set_status(job, SequenceStatus.FINISHED_STOPPED)
                reply.cancel()
                if asyncio.current_task().cancelling():
                    raise
            except Exception as exc:
                self._set_status(
                    job,
                    SequenceStatus.FINISHED_REJECTED
                    if isinstance(exc, InvalidPrompt)
                    else SequenceStatus.FINISHED_ERROR,
                )
                if not reply.done():
                    reply.set_exception(exc)
            else:
                if result is None:
                    status = SequenceStatus.FINISHED_PREFILL
                elif result.finish_reason == "stop":
                    status = SequenceStatus.FINISHED_EOS
                else:
                    status = SequenceStatus.FINISHED_LENGTH
                self._set_status(job, status)
                if not reply.done():
                    reply.set_result(result)
