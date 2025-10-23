# fastlri/utils/metrics.py

class Metrics:
    """
    Per-instance counters for arithmetic ops and semantic I/O.
    Use metrics.add/mul in hot loops instead of plain +/*.

    Typical usage:
        m = Metrics(enabled=True)
        x = m.add(a, b)
        y = m.mul(x, c)
        snap = m.snapshot()   # {"IO": {...}, "OPS": {...}}
        m.reset()             # clear counters

    If enabled=False, calls are near-zero overhead pass-throughs.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self.IO = {}   # dict[str, int]
        self.OPS = {}  # dict[str, int]

    # ---- arithmetic wrappers ----
    def add(self, a, b):
        if self.enabled:
            self.OPS["add"] = self.OPS.get("add", 0) + 1
        return a + b

    def mul(self, a, b):
        if self.enabled:
            self.OPS["mul"] = self.OPS.get("mul", 0) + 1
        return a * b

    # ---- semantic I/O counters (charts etc.) ----
    def io_inc(self, key: str, n: int = 1):
        if self.enabled:
            self.IO[key] = self.IO.get(key, 0) + int(n)

    # ---- lifecycle helpers ----
    def reset(self):
        """Clear counters (use to isolate one operation, e.g., one token append)."""
        if self.enabled:
            self.IO.clear()
            self.OPS.clear()

    def snapshot(self):
        """
        Shallow-copy counters for serialization/logging without resetting.
        Returns {"IO": dict, "OPS": dict}.
        """
        if not self.enabled:
            return {"IO": {}, "OPS": {}}
        return {"IO": dict(self.IO), "OPS": dict(self.OPS)}

    def dump_and_reset(self):
        """Snapshot and then reset (convenient per-token)."""
        snap = self.snapshot()
        self.reset()
        return snap
