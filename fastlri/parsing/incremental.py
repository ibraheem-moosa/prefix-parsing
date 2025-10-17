from collections import defaultdict as dd
from fastlri.base.symbol import Sym
from fastlri.utils.metrics import reset_all, IO, OPS, add, mul
from fastlri.utils.chart_wrappers import CountedDD

class IncrementalLRI:
    """
    Incremental prefix parser for lri_fast that reuses DP state across appends.
    Requires CNF. Preserves metrics when return_metrics=True.

    Approximation knobs (both OFF by default for exact mode):
      - max_window:    if set (int > 0), only consider split points j in [k-max_window, k-1]
      - max_backspan:  if set (int > 0), only build spans that start no earlier than i0 >= k-max_backspan

    Key exact speedup retained: γ & δ are updated only for the *new* split j = N-1 at each append.
    """
    def __init__(self, cfg, return_metrics=False, max_window=None, max_backspan=None):
        assert cfg.in_cnf
        self.cfg = cfg
        self.return_metrics = return_metrics

        # Approximation controls
        self.max_window = max_window if (isinstance(max_window, int) and max_window > 0) else None
        self.max_backspan = max_backspan if (isinstance(max_backspan, int) and max_backspan > 0) else None

        # arithmetic ops (instrumented when return_metrics=True)
        self._add = add if return_metrics else (lambda a, b: a + b)
        self._mul = mul if return_metrics else (lambda a, b: a * b)

        # ---- static (precomputed once) ----
        self.V = cfg.ordered_V
        self.V_idx = {X: i for i, X in enumerate(self.V)}

        # Sparse fanout indices from grammar rules
        self._xz_from_yz = {}  # (Y,Z) -> [(X,w)]
        self._xz_from_y  = {}  # Y -> [(X,Z,w)]
        for p, w in cfg.binary:
            X, Y, Z = p.head, p.body[0], p.body[1]
            self._xz_from_yz.setdefault((Y, Z), []).append((X, w))
            self._xz_from_y.setdefault(Y, []).append((X, Z, w))

        # Left-corner closure
        self.P_L = self._plc()

        # E[X,Y] as dict
        self.E = dd(lambda: 0.0)
        for X in self.cfg.V:
            for Y in self.cfg.V:
                self.E[X, Y] = self.P_L[self.V_idx[X], self.V_idx[Y]]

        # ---- dynamic DP state (grows with tokens) ----
        self.tokens = []  # list[Sym]
        self.N = 0

        # Use CountedDD when instrumenting IO
        DD = CountedDD if self.return_metrics else dd
        self.beta  = DD(lambda: 0.0)  # CKY chart β[i, X, k]
        self.ppre  = DD(lambda: 0.0)  # prefix chart ppre[i, X, k]
        self.gamma = DD(lambda: 0.0)  # γ[i, j, X, Z]   (no dependence on k)
        self.delta = DD(lambda: 0.0)  # δ[i, j, X, Z]   (no dependence on k)

        # ppre[i,X,i] = 1 is ensured lazily via _ensure_empty_diagonal

    def _plc(self):
        """
        Compute left-corner expectations P_L = (I - P)^(-1) using NumPy.
        In light/pyPy runs you can monkeypatch this method to return a wrapper
        around a cached matrix (your light script does this).
        """
        import numpy as np
        V_idx = self.V_idx
        P = np.zeros((len(self.V), len(self.V)))
        for p, w in self.cfg.binary:
            X, Y = V_idx[p.head], V_idx[p.body[0]]
            P[X, Y] += w
        return np.linalg.inv(np.eye(len(self.V)) - P)

    def _ensure_empty_diagonal(self, upto):
        # Ensure ppre[i,X,i] = 1 for i up to 'upto'
        for i in range(upto + 1):
            for X in self.cfg.V:
                if (i, X, i) not in self.ppre:  # avoid double-counting IO
                    self.ppre[i, X, i] = 1.0

    # ---------- incremental γ/δ slice update for a fixed j ----------
    def _update_gamma_delta_for_j(self, j):
        """
        Build γ[i0, j, *, *] and δ[i0, j, *, *] for required i0.
        If max_backspan is set, restrict to i0 >= j - max_backspan + 1.
        """
        i0_min = 0
        if self.max_backspan is not None:
            # i0 must satisfy (j - i0) < max_backspan  =>  i0 > j - max_backspan
            i0_min = max(0, j - self.max_backspan + 1)

        for i0 in range(i0_min, j):
            # γ[i0, j, X, Z] = sum_Y w(X→Y Z) * β[i0, Y, j]
            for Y in self.cfg.V:
                b = self.beta[i0, Y, j]
                if b == 0.0:
                    continue
                for X, Z, w in self._xz_from_y.get(Y, ()):
                    self.gamma[i0, j, X, Z] = self._add(self.gamma[i0, j, X, Z],
                                                         self._mul(w, b))
            # δ[i0, j, X, Z] = sum_Y E[X,Y] * γ[i0, j, Y, Z]
            for X in self.cfg.V:
                for Z in self.cfg.V:
                    acc = 0.0
                    for Y in self.cfg.V:
                        g = self.gamma[i0, j, Y, Z]
                        if g != 0.0:
                            acc = self._add(acc, self._mul(self.E[X, Y], g))
                    if acc != 0.0:
                        self.delta[i0, j, X, Z] = self._add(self.delta[i0, j, X, Z], acc)

    def append(self, token_str):
        """
        Append one terminal symbol and update β and ppre for k=N.
        γ/δ are updated *only* for the new split j=N-1 (incremental).
        If max_window/max_backspan are set, approximate by restricting j and/or i0.
        """
        # convert token
        tok = Sym(token_str) if not isinstance(token_str, Sym) else token_str

        # extend length
        self.tokens.append(tok)
        self.N += 1
        N = self.N

        if self.return_metrics:
            reset_all()

        self._ensure_empty_diagonal(N)  # ensure ppre[i,X,i]=1 up to new N

        # (1) Terminal initialization: β[N-1, head, N]
        for (head, body), w in self.cfg.terminal:
            if body[0] == tok:
                self.beta[N - 1, head, N] = self._add(self.beta[N - 1, head, N], w)

        # (2) CKY binary updates for spans that end at k = N
        #     Apply max_backspan (cap span length) and max_window (cap j range).
        for span_len in range(2, N + 1):
            if self.max_backspan is not None and span_len > self.max_backspan:
                continue  # skip very long backspans
            i0 = N - span_len
            k = N

            # j range
            j_start = i0 + 1
            if self.max_window is not None:
                j_start = max(j_start, k - self.max_window)

            for Y in self.cfg.V:
                for Z in self.cfg.V:
                    gamma_tmp = 0.0
                    for j in range(j_start, k):
                        b1 = self.beta[i0, Y, j]
                        if b1 == 0.0:
                            continue
                        b2 = self.beta[j, Z, k]
                        if b2 == 0.0:
                            continue
                        gamma_tmp = self._add(gamma_tmp, self._mul(b1, b2))
                    if gamma_tmp == 0.0:
                        continue
                    for X, w in self._xz_from_yz.get((Y, Z), ()):
                        self.beta[i0, X, k] = self._add(self.beta[i0, X, k],
                                                         self._mul(gamma_tmp, w))

        # (3) INCREMENTAL γ/δ update: only for the new split j = N-1 (if it's in window)
        if N - 1 >= 1:
            if (self.max_window is None) or (N - 1 >= N - self.max_window):
                self._update_gamma_delta_for_j(N - 1)

        # (4) Prefix base case for the new 1-length span at end
        for X in self.cfg.V:
            for (p, w) in self.cfg.terminal:
                Y, v = p.head, p.body[0]
                if v == tok:
                    self.ppre[N - 1, X, N] = self._add(self.ppre[N - 1, X, N],
                                                       self._mul(self.E[X, Y], w))

        # (5) Prefix recurrence only for k = N.
        #     Reuse δ[i0, j, X, Z] for j in restricted range; respect max_backspan and max_window.
        for span_len in range(2, N + 1):
            if self.max_backspan is not None and span_len > self.max_backspan:
                continue
            i0 = N - span_len
            k = N

            j_start = i0 + 1
            if self.max_window is not None:
                j_start = max(j_start, k - self.max_window)

            for j in range(j_start, k):
                for X in self.cfg.V:
                    for Z in self.cfg.V:
                        d = self.delta[i0, j, X, Z]
                        if d == 0.0:
                            continue
                        rhs = self.ppre[j, Z, k]
                        if rhs == 0.0:
                            continue
                        self.ppre[i0, X, k] = self._add(self.ppre[i0, X, k],
                                                        self._mul(d, rhs))
        # done

    def score_prefix(self):
        """Probability of the current prefix."""
        return self.ppre[0, self.cfg.S, self.N]

    def score_with_candidate(self, cand_token_str):
        """
        Try a candidate terminal WITHOUT committing permanent state.
        Snapshots DP charts, calls append(), reads score, then restores.
        """
        # snapshot current dicts (values only)
        snap_beta  = dict(self.beta)
        snap_ppre  = dict(self.ppre)
        snap_gamma = dict(self.gamma)
        snap_delta = dict(self.delta)

        # do the append
        self.append(cand_token_str)
        val = self.score_prefix()

        # restore with the same chart type (CountedDD when instrumenting)
        DD = CountedDD if self.return_metrics else dd
        self.beta  = DD(lambda: 0.0); self.beta.update(snap_beta)
        self.ppre  = DD(lambda: 0.0); self.ppre.update(snap_ppre)
        self.gamma = DD(lambda: 0.0); self.gamma.update(snap_gamma)
        self.delta = DD(lambda: 0.0); self.delta.update(snap_delta)

        self.tokens.pop()
        self.N -= 1
        return val
