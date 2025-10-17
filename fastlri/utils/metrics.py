# fastlri/utils/metrics.py
from collections import defaultdict

# semantic I/O and arithmetic counters
IO = defaultdict(int)
OPS = defaultdict(int)

def reset_all():
    IO.clear()
    OPS.clear()

# no-op passthroughs for now; we’ll swap these into hot loops later
def add(a, b):  # arithmetic add
    OPS["add"] += 1
    return a + b

def mul(a, b):  # arithmetic multiply
    OPS["mul"] += 1
    return a * b
