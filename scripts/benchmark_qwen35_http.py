import argparse
import math
from statistics import median
from time import perf_counter

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    if min(args.requests, args.max_tokens, args.prompt_repeat) < 1 or args.warmups < 0:
        parser.error(
            "requests, max-tokens and prompt-repeat must be positive, and warmups must be nonnegative"
        )

    body = {
        "model": args.model,
        "prompt": "\n".join([args.prompt] * args.prompt_repeat),
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    durations = []
    prompt_tokens = []
    output_tokens = []
    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        for index in range(args.warmups + args.requests):
            started = perf_counter()
            response = client.post("/v1/completions", json=body)
            response.raise_for_status()
            usage = response.json()["usage"]
            elapsed = perf_counter() - started
            if index >= args.warmups:
                durations.append(elapsed)
                prompt_tokens.append(usage["prompt_tokens"])
                output_tokens.append(usage["completion_tokens"])

    p95 = sorted(durations)[math.ceil(0.95 * len(durations)) - 1]
    print(
        f"{args.requests} requests | median prompt/output tokens "
        f"{median(prompt_tokens):g}/{median(output_tokens):g} | "
        f"p50 {median(durations) * 1000:.1f} ms | p95 {p95 * 1000:.1f} ms | "
        f"{sum(output_tokens) / sum(durations):.1f} output tok/s end to end"
    )


if __name__ == "__main__":
    main()
