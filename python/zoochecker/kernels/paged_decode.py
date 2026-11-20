import mlx.core as mx


_PARTIAL = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint head = thread_position_in_grid.y;
    const uint part = thread_position_in_grid.z;
    const uint kv_head = head / HEADS_PER_KV;
    const uint begin = part * PARTITION;
    const uint end = min(begin + PARTITION, uint(length[0]));
    float q[HEAD_DIM / 32];
    float weighted[HEAD_DIM / 32];
    for (uint i = 0; i < HEAD_DIM / 32; ++i) {
        q[i] = float(query[head * HEAD_DIM + lane + i * 32]);
        weighted[i] = 0.0f;
    }

    float peak = -INFINITY;
    float total = 0.0f;
    for (uint token = begin; token < end;) {
        const uint page = uint(table[token / PAGE_TOKENS]);
        const uint page_end = min(end, (token / PAGE_TOKENS + 1) * PAGE_TOKENS);
        if (page >= PHYSICAL_PAGES) {
            token = page_end;
            continue;
        }
        uint base = ((kv_head * PHYSICAL_PAGES + page) * PAGE_TOKENS + token % PAGE_TOKENS) * HEAD_DIM;
        for (; token < page_end; ++token, base += HEAD_DIM) {
            float score = 0.0f;
            for (uint i = 0; i < HEAD_DIM / 32; ++i) {
                score += q[i] * float(keys[base + lane + i * 32]);
            }
            score = simd_sum(score) * scale[0];
            const float next_peak = max(peak, score);
            const bool new_peak = score > peak;
            const float weight = fast::exp(-abs(peak - score));
            const float previous = new_peak ? weight : 1.0f;
            const float current = new_peak ? 1.0f : weight;
            total = fma(total, previous, current);
            peak = next_peak;
            for (uint i = 0; i < HEAD_DIM / 32; ++i) {
                weighted[i] = fma(current, float(values[base + lane + i * 32]),
                                  weighted[i] * previous);
            }
        }
    }
    const uint index = head * partitions[0] + part;
    for (uint i = 0; i < HEAD_DIM / 32; ++i) {
        sums[index * HEAD_DIM + lane + i * 32] = weighted[i];
    }
    if (lane == 0) {
        stats[index * 2] = peak;
        stats[index * 2 + 1] = total;
    }
"""

_MERGE = r"""
    const uint dim = thread_position_in_grid.x;
    const uint head = thread_position_in_grid.y;
    const uint first = head * partitions[0];
    float peak = -INFINITY;
    for (uint i = 0; i < partitions[0]; ++i) {
        peak = max(peak, stats[(first + i) * 2]);
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (uint i = 0; i < partitions[0]; ++i) {
        const uint index = first + i;
        const float count = stats[index * 2 + 1];
        if (count > 0.0f) {
            const float weight = fast::exp(stats[index * 2] - peak);
            denominator = fma(weight, count, denominator);
            numerator = fma(weight, sums[index * HEAD_DIM + dim], numerator);
        }
    }
    out[head * HEAD_DIM + dim] = OUT(denominator > 0.0f ? numerator / denominator : 0.0f);
"""

_partial = mx.fast.metal_kernel(
    name="zoochecker_paged_decode_partial",
    input_names=["query", "keys", "values", "table", "length", "scale", "partitions"],
    output_names=["sums", "stats"],
    source=_PARTIAL,
)
_merge = mx.fast.metal_kernel(
    name="zoochecker_paged_decode_merge",
    input_names=["sums", "stats", "partitions"],
    output_names=["out"],
    source=_MERGE,
)


def paged_decode(
    query: mx.array,
    keys: mx.array,
    values: mx.array,
    table: mx.array,
    *,
    length: int,
    scale: float,
    partition: int,
) -> mx.array:
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[2] != 1:
        raise ValueError("Paged decode requires one query token and one sequence")
    if keys.ndim != 4 or keys.shape != values.shape or keys.dtype != values.dtype:
        raise ValueError(
            "K/V must have matching [kv_heads, pages, page_tokens, head_dim] layouts"
        )
    kv_heads, pages, page_tokens, head_dim = keys.shape
    heads = query.shape[1]
    if (
        head_dim != query.shape[-1]
        or not head_dim
        or head_dim % 32
        or not heads
        or not kv_heads
        or heads % kv_heads
    ):
        raise ValueError("Unsupported grouped-query attention geometry")
    if (
        table.ndim != 1
        or table.dtype != mx.int32
        or not 0 < length <= table.size * page_tokens
    ):
        raise ValueError("Invalid page table or sequence length")
    if partition < 1:
        raise ValueError("Partition size must be positive")

    count = (length + partition - 1) // partition
    partitions = mx.array([count], dtype=mx.uint32)
    sums, stats = _partial(
        inputs=[
            query,
            keys,
            values,
            table,
            mx.array([length], dtype=mx.uint32),
            mx.array([scale], dtype=mx.float32),
            partitions,
        ],
        template=[
            ("HEAD_DIM", head_dim),
            ("HEADS_PER_KV", heads // kv_heads),
            ("PAGE_TOKENS", page_tokens),
            ("PHYSICAL_PAGES", pages),
            ("PARTITION", partition),
        ],
        grid=(32, heads, count),
        threadgroup=(32, 1, 1),
        output_shapes=[(heads, count, head_dim), (heads, count, 2)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return _merge(
        inputs=[sums, stats, partitions],
        template=[("HEAD_DIM", head_dim), ("OUT", query.dtype)],
        grid=(head_dim, heads, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[query.shape],
        output_dtypes=[query.dtype],
    )[0]
