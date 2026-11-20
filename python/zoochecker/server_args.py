"""Arguments for the whole application."""

from dataclasses import dataclass


@dataclass
class ServerArgs:
    model_name: str
    model_path: str | None

    # cache configuration
    cache_dtype: str = "bfloat16"
    cache_pool_type: str = "mlx_kv"
    cache_slots: int = 1
    cache_max_tokens: int = 2048
    cache_block_tokens: int = 16
    cache_decode_partition_tokens: int = 64

    tokenizer_ipc_endpoint: str | None = None
