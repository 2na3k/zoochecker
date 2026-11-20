import argparse

from zoochecker.api.v1.service import run_server
from zoochecker.server_args import ServerArgs


def main() -> None:
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--model", dest="model_name", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--cache-pool-type", choices=("mlx_kv", "mlx_paged_kv"))
    parser.add_argument("--cache-dtype", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--cache-slots", type=int)
    parser.add_argument("--cache-max-tokens", type=int)
    parser.add_argument("--cache-block-tokens", type=int)
    parser.add_argument("--prefill-step-size", type=int)
    parser.add_argument("--scheduler-batch-wait-seconds", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-image-bytes", type=int)
    parser.add_argument("--max-image-pixels", type=int)
    parser.add_argument("--image-timeout-seconds", type=float)
    parser.add_argument(
        "--no-telemetry", dest="telemetry_enabled", action="store_false"
    )
    parser.add_argument("--telemetry-memory-poll-interval", type=float)
    parser.add_argument("--telemetry-json-log")
    args = ServerArgs(**vars(parser.parse_args()))
    run_server(args)


if __name__ == "__main__":
    main()
