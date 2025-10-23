# fastlri/utils/chart_wrappers.py
from collections import defaultdict
from fastlri.utils.metrics import Metrics

def make_counted_dd(metrics: Metrics):
    """
    Factory that returns a defaultdict subclass wired to a Metrics instance.
    Use:
        DD = make_counted_dd(metrics)
        chart = DD(lambda: 0.0)
    """
    if metrics is None:
        raise TypeError("make_counted_dd(metrics): 'metrics' must be a Metrics instance")

    class CountedDD(defaultdict):
        """defaultdict(float) that counts semantic chart I/O via a Metrics instance."""
        def __init__(self, default_factory=None):
            super().__init__(default_factory)

        def __contains__(self, key):
            # Use defaultdict's behavior; no metrics counted here
            return super().__contains__(key)

        def __getitem__(self, k):
            # defaultdict inserts default on miss; count both cases
            if super().__contains__(k):
                metrics.io_inc("chart_read_hit", 1)
            else:
                metrics.io_inc("chart_read_miss", 1)
            return super().__getitem__(k)

        def __setitem__(self, k, v):
            if super().__contains__(k):
                metrics.io_inc("chart_write_over", 1)
            else:
                metrics.io_inc("chart_write_new", 1)
            return super().__setitem__(k, v)

    return CountedDD


# --- Backward-compat shim (optional) ---
# If some old code still does: CountedDD(lambda: 0.0) without passing metrics,
# make the failure message explicit rather than silently using globals.
class CountedDD(defaultdict):  # pragma: no cover - only to catch legacy calls early
    def __init__(self, default_factory=None, *, metrics: Metrics = None):
        if metrics is None:
            raise TypeError(
                "CountedDD now requires 'metrics=Metrics(...)'. "
                "Prefer: DD = make_counted_dd(metrics); chart = DD(lambda: 0.0)."
            )
        self._metrics = metrics
        super().__init__(default_factory)

    def __getitem__(self, k):
        if super().__contains__(k):
            self._metrics.io_inc("chart_read_hit", 1)
        else:
            self._metrics.io_inc("chart_read_miss", 1)
        return super().__getitem__(k)

    def __setitem__(self, k, v):
        if super().__contains__(k):
            self._metrics.io_inc("chart_write_over", 1)
        else:
            self._metrics.io_inc("chart_write_new", 1)
        return super().__setitem__(k, v)
