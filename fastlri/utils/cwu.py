# fastlri/utils/cwu.py
from __future__ import annotations

# Default relative weights (portable baseline). You can later replace with
# calibrated ns/op from a micro-harness on your machine.
DEFAULT_WEIGHTS = {
    # arithmetic
    "add": 1.0, "mul": 1.0,
    # chart I/O
    "chart_read_hit": 4.0, "chart_read_miss": 6.0,
    "chart_write_new": 10.0, "chart_write_over": 8.0,
    # reserved (not used yet, but won’t hurt if present)
    "chart_rmw": 12.0,
    # you can add agenda/set/hash keys later if you instrument them
}

def cwu(metrics: dict, weights: dict | None = None) -> float:
    """Collapse metrics into a single scalar 'Computation Work Units' (CWU)."""
    w = weights or DEFAULT_WEIGHTS
    IO = metrics.get("IO", {})
    OPS = metrics.get("OPS", {})
    total = 0.0
    for k, v in OPS.items():
        total += w.get(k, 0.0) * v
    for k, v in IO.items():
        total += w.get(k, 0.0) * v
    return total
