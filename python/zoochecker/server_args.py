"""Arguments for the whole application."""

from dataclasses import dataclass


@dataclass
class ServerArgs:
    model_name: str
    model_path: str | None

    cache_dtype: str = "bfloat16"
    cache_pool_type: str = "mlx_kv"
    cache_slots: int = 1
    cache_max_tokens: int = 2048
