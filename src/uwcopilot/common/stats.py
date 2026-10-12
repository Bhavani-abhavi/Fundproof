"""Small statistics helpers for proportions measured on small samples.

- wilson_interval: the Wilson score 95% interval for k successes out of n. It behaves well at 0/n and n/n, where the
  plain normal approximation gives impossible intervals, which is why the innovation evaluations use it for rates
  such as attack success and checker recall.
- mcnemar_exact: the exact (binomial) McNemar test for paired yes/no outcomes, used to compare two defense layers on
  the same payloads.
"""

import math

Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval for successes / n, as (low, high) rounded to 4 decimals; None when n is 0."""
    if n <= 0:
        return None
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be between 0 and n, got {successes} of {n}")
    p_hat = successes / n
    denominator = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denominator
    return round(max(0.0, center - half_width), 4), round(min(1.0, center + half_width), 4)


def rate_summary(successes: int, n: int) -> dict:
    """{'k', 'n', 'rate', 'wilson_ci95'} for a proportion (rate None when n is 0)."""
    return {
        "k": successes,
        "n": n,
        "rate": None if n == 0 else round(successes / n, 4),
        "wilson_ci95": wilson_interval(successes, n),
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b and c discordant pairs (binomial test with p = 0.5).

    b = pairs where only the first condition succeeded, c = pairs where only the second did. With no discordant
    pairs the p-value is 1.0.
    """
    n = b + c
    if n == 0:
        return 1.0
    smaller = min(b, c)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / 2**n
    return round(min(1.0, 2 * tail), 6)
