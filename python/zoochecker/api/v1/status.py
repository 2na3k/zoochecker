from time import monotonic

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from zoochecker.metrics import Metrics
from zoochecker.scheduler import Scheduler

router = APIRouter()


def _ready(request: Request) -> bool:
    scheduler: Scheduler | None = getattr(request.app.state, "scheduler", None)
    tasks = getattr(request.app.state, "ipc_tasks", ())
    return (
        scheduler is not None
        and scheduler.ready
        and bool(tasks)
        and all(not task.done() for task in tasks)
    )


@router.get("/health")
async def health(request: Request) -> Response:
    if _ready(request):
        return JSONResponse({"status": "ok"})
    return JSONResponse(
        {"status": "unavailable", "detail": "worker is not running"}, status_code=503
    )


@router.get("/v1/stats")
async def stats(request: Request) -> Response:
    metrics: Metrics | None = getattr(request.app.state, "metrics", None)
    if metrics is None:
        return JSONResponse({"error": "telemetry disabled"}, status_code=503)
    await metrics.flush()
    return JSONResponse(_snapshot(request, metrics))


@router.get("/v1/events")
async def events(request: Request, limit: int = 200) -> Response:
    metrics: Metrics | None = getattr(request.app.state, "metrics", None)
    if metrics is None:
        return JSONResponse({"error": "telemetry disabled"}, status_code=503)
    await metrics.flush()
    return JSONResponse(metrics.recent_events(max(0, limit)))


@router.get("/metrics")
async def prometheus_metrics(request: Request) -> Response:
    metrics: Metrics | None = getattr(request.app.state, "metrics", None)
    if metrics is None:
        return JSONResponse({"error": "telemetry disabled"}, status_code=503)
    await metrics.flush()
    return Response(
        metrics.prometheus(_snapshot(request, metrics)),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


def _snapshot(request: Request, metrics: Metrics) -> dict:
    scheduler: Scheduler | None = getattr(request.app.state, "scheduler", None)
    waiting, running = scheduler.counts if scheduler is not None else (0, 0)
    cache = request.app.state.cache_manager.stats()
    return {
        "status": "ok" if _ready(request) else "unavailable",
        **metrics.snapshot(waiting, running, cache),
    }


async def record_http(request: Request, call_next) -> Response:
    metrics: Metrics | None = getattr(request.app.state, "metrics", None)
    if metrics is None:
        return await call_next(request)
    started = monotonic()

    def record(status: int) -> None:
        route = request.scope.get("route")
        metrics.record(
            "http_request",
            method=request.method,
            endpoint=getattr(route, "path", "unmatched"),
            status_code=status,
            duration_seconds=monotonic() - started,
        )

    try:
        response = await call_next(request)
    except Exception:
        record(500)
        raise
    body = response.body_iterator

    async def measured_body():
        try:
            async for data in body:
                yield data
        finally:
            record(response.status_code)

    response.body_iterator = measured_body()
    return response
