from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Literal


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


@dataclass(frozen=True, slots=True)
class Sampling:
    max_tokens: int
    temperature: float
    top_p: float


@dataclass(frozen=True, slots=True)
class PromptRequest:
    request_id: str
    prompt: str | list[dict]
    sampling: Sampling | None = None


@dataclass(frozen=True, slots=True)
class VisionPrompt:
    text: str
    images: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TokenizedRequest:
    request_id: str
    token_ids: tuple[int, ...]
    sampling: Sampling | None = None
    vision: VisionPrompt | None = None


@dataclass(frozen=True, slots=True)
class GeneratedTokens:
    token_ids: tuple[int, ...]
    finish_reason: Literal["stop", "length"]
    prompt_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class Generation:
    request_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length"]


@dataclass(frozen=True, slots=True)
class OutputDelta:
    request_id: str
    token_text: str
    token_id: int
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


EmitDelta = Callable[[OutputDelta], Awaitable[None]]
