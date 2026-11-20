from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Annotated, Literal

from openai.types import Completion, CompletionUsage as UsageInfo
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk as SDKChatCompletionChunk,
)
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from openai.types.chat.chat_completion_chunk import Choice as ChatChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_choice import CompletionChoice
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr


class InvalidPrompt(ValueError):
    pass


class InferenceFailed(RuntimeError):
    pass


class SequenceStatus(StrEnum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED_PREFILL = auto()
    FINISHED_EOS = auto()
    FINISHED_LENGTH = auto()
    FINISHED_STOPPED = auto()
    FINISHED_REJECTED = auto()
    FINISHED_ERROR = auto()

    @property
    def is_finished(self) -> bool:
        return self not in (SequenceStatus.WAITING, SequenceStatus.RUNNING)


FinishReason = Literal["stop", "length"]
NonnegativeInt = Annotated[StrictInt, Field(ge=0)]


@dataclass(frozen=True, slots=True)
class Sampling:
    max_tokens: Annotated[StrictInt, Field(ge=1)]
    temperature: Annotated[StrictFloat, Field(ge=0, le=2)]
    top_p: Annotated[StrictFloat, Field(gt=0, le=1)]


@dataclass(frozen=True, slots=True)
class VisionPrompt:
    text: StrictStr
    images: tuple[StrictStr, ...]


@dataclass(frozen=True, slots=True)
class TokenizedRequest:
    request_id: StrictStr
    token_ids: tuple[NonnegativeInt, ...]
    sampling: Sampling | None = None
    vision: VisionPrompt | None = None


@dataclass(frozen=True, slots=True)
class GeneratedTokens:
    token_ids: tuple[NonnegativeInt, ...]
    finish_reason: FinishReason
    prompt_tokens: NonnegativeInt | None = None


@dataclass(frozen=True, slots=True)
class Generation:
    request_id: StrictStr
    text: StrictStr
    prompt_tokens: NonnegativeInt
    completion_tokens: NonnegativeInt
    finish_reason: FinishReason
    output_token_ids: tuple[NonnegativeInt, ...]

    def __post_init__(self) -> None:
        if self.completion_tokens != len(self.output_token_ids):
            raise ValueError("Completion token count must match the output token list")


@dataclass(frozen=True, slots=True)
class OutputDelta:
    request_id: StrictStr
    token_text: StrictStr
    token_id: StrictInt
    finish_reason: FinishReason | None = None
    prompt_tokens: NonnegativeInt = 0
    completion_tokens: NonnegativeInt = 0
    output_token_ids: tuple[NonnegativeInt, ...] | None = None

    def __post_init__(self) -> None:
        if self.finish_reason is None:
            if self.token_id < 0 or self.output_token_ids is not None:
                raise ValueError(
                    "Token delta requires a token ID and no final token list"
                )
        elif (
            self.token_id != -1
            or self.output_token_ids is None
            or len(self.output_token_ids) != self.completion_tokens
        ):
            raise ValueError("Terminal delta requires the complete output token list")


EmitDelta = Callable[[OutputDelta], Awaitable[None]]

AssistantMessage = ChatCompletionMessage
ChatDelta = ChoiceDelta


class ImageURL(BaseModel):
    url: StrictStr = Field(min_length=1)
    detail: Literal["auto", "low", "high"] = "auto"


class ImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


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
    text: StrictStr


class ChatMessage(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str | list[Annotated[TextPart | ImagePart, Field(discriminator="type")]]


@dataclass(frozen=True, slots=True)
class PromptRequest:
    request_id: StrictStr
    prompt: StrictStr | Annotated[list[ChatMessage], Field(min_length=1)]
    sampling: Sampling | None = None


class ChatCompletionRequest(_CompletionOptions):
    messages: list[ChatMessage] = Field(min_length=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)


class PrefillRequest(BaseModel):
    prompt: str


class CompletionResponse(Completion):
    usage: UsageInfo
    output_token_ids: tuple[int, ...]


class ChatCompletionResponse(ChatCompletion):
    choices: list[ChatCompletionChoice]
    usage: UsageInfo
    output_token_ids: tuple[int, ...]


class CompletionChunkChoice(CompletionChoice):
    finish_reason: Literal["stop", "length", "content_filter"] | None


class CompletionChunk(Completion):
    choices: list[CompletionChunkChoice]
    output_token_ids: tuple[int, ...] | None = None


class ChatCompletionChunk(SDKChatCompletionChunk):
    choices: list[ChatChunkChoice]
    output_token_ids: tuple[int, ...] | None = None


class ErrorInfo(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorInfo


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "local"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class PrefillResponse(BaseModel):
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class UnavailableResponse(BaseModel):
    status: Literal["unavailable"] = "unavailable"
    detail: str


class TelemetryDisabledResponse(BaseModel):
    error: Literal["telemetry disabled"] = "telemetry disabled"


class SchedulerStats(BaseModel):
    waiting_seqs: int
    running_seqs: int


class CacheStats(BaseModel):
    slots_total: int
    slots_used: int
    blocks_total: int
    blocks_used: int
    blocks_free: int
    block_tokens: int
    bytes_total: int
    bytes_used: int
    utilisation_ratio: float


class ThroughputStats(BaseModel):
    requests_total: int
    prompt_tokens_total: int
    completion_tokens_total: int
    step_loop_iterations_total: int


class MLXMemoryStats(BaseModel):
    active_bytes: int
    cache_bytes: int
    peak_bytes: int


class StatsResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    timestamp: float
    uptime_seconds: float
    model: str
    active_requests: int
    scheduler: SchedulerStats
    kvcache_text: CacheStats
    throughput: ThroughputStats
    latency_p50_total_seconds: float | None
    latency_p99_total_seconds: float | None
    latency_p50_ttft_seconds: float | None
    latency_p99_ttft_seconds: float | None
    latency_p50_tpot_seconds: float | None
    latency_p99_tpot_seconds: float | None
    latency_p50_prefill_seconds: float | None
    latency_p99_prefill_seconds: float | None
    latency_p50_decode_step_seconds: float | None
    latency_p99_decode_step_seconds: float | None
    latency_p50_queue_wait_seconds: float | None
    latency_p99_queue_wait_seconds: float | None
    mlx_memory: MLXMemoryStats
    http_requests_total: int
    errors_total: int
    telemetry_dropped_events_total: int


class TelemetryEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_type: str
    timestamp: float
    wall_time: float
