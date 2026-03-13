"""Project constants."""

SLA_TIERS = {
    "strict": {
        "p99_ttft_ms": 500.0,
        "p99_itl_ms": 60.0,
    },
    "balanced": {
        "p99_ttft_ms": 650.0,
        "p99_itl_ms": 80.0,
    },
}

DEFAULT_TOP_K_VERIFY = 3
DEFAULT_PRUNE_TOP_N = 12
DEFAULT_VERIFICATION_MIN_RUNS = 1
