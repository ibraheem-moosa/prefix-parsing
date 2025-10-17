# fastlri/utils/chart_wrappers.py
from collections import defaultdict
from fastlri.utils.metrics import IO

class CountedDD(defaultdict):
    """defaultdict(float) that counts semantic chart I/O."""
    def __init__(self, default_factory):
        super().__init__(default_factory)

    def __getitem__(self, k):
        # defaultdict inserts default on miss; count both cases
        if k in self:
            IO["chart_read_hit"] += 1
        else:
            IO["chart_read_miss"] += 1
        return super().__getitem__(k)

    def __setitem__(self, k, v):
        if k in self:
            IO["chart_write_over"] += 1
        else:
            IO["chart_write_new"] += 1
        return super().__setitem__(k, v)
