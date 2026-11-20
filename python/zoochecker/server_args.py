"""Arguments for the whole application."""

from dataclasses import dataclass


@dataclass
class ServerArgs:
    model_name: str
    model_path: str | None

    # cache configuration
    cache_dtype: str = "bfloat16"
    cache_pool_type: str = "mlx_kv"
    cache_slots: int = 2
    cache_max_tokens: int = 2048
    cache_block_tokens: int = 16
    cache_decode_partition_tokens: int = 64

    prefill_step_size: int = 512
    scheduler_batch_wait_seconds: float = 0.002
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    enable_thinking: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    telemetry_enabled: bool = True
    telemetry_event_queue_size: int = 50_000
    telemetry_recent_events: int = 10_000
    telemetry_latency_samples: int = 1024
    telemetry_memory_poll_interval: float = 5.0
    telemetry_json_log: str | None = None
    max_images: int = 4
    max_image_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 16_000_000
    image_timeout_seconds: float = 10.0
