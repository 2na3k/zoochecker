from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenizedRequest:
    request_id: str
    token_ids: tuple[int, ...]
