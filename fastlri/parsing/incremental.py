# fastlri/parsing/incremental.py

from collections import defaultdict as dd
from fastlri.base.symbol import Sym
from fastlri.utils.metrics import Metrics
from fastlri.utils.chart_wrappers import make_counted_dd

class IncrementalLRI:
    def __init__(self, cfg, return_metrics=False, max_window=None, max_backspan=None):
        assert cfg.in_cnf
        self.cfg = cfg
        self.return_metrics = bool(return_metrics)
        self.metrics = Metrics(enabled=self.return_metrics)
        self.max_window = max_window if (isinstance(max_window, int) and max_window > 0) else None
        self.max_backspan = max_backspan if (isinstance(max_backspan, int) and max_backspan > 0) else None

        if self.return_metrics:
            self._add = self.metrics.add
            self._mul = self.metrics.mul
        else:
            self._add = (lambda a, b: a + b)
            self._mul = (lambda a, b: a * b)

        self.V = cfg.ordered_V
        self.V_idx = {X: i for i, X in enumerate(self.V)}

        self._xz_from_yz = {}
        self._xz_from_y  = {}
        for p, w in cfg.binary:
            X, Y, Z = p.head, p.body[0], p.body[1]
            self._xz_from_yz.setdefault((Y, Z), []).append((X, w))
            self._xz_from_y.setdefault(Y, []).append((X, Z, w))

        self.P_L = self._plc()
        self.E = dd(lambda: 0.0)
        for X in self.cfg.V:
            for Y in self.cfg.V:
                self.E[X, Y] = self.P_L[self.V_idx[X], self.V_idx[Y]]

        self.tokens = []
        self.N = 0
        if self.return_metrics:
            DD = make_counted_dd(self.metrics)
        else:
            DD = dd
        self._DD_factory = DD
        self.beta  = DD(lambda: 0.0)
        self.ppre  = DD(lambda: 0.0)
        self.gamma = DD(lambda: 0.0)
        self.delta = DD(lambda: 0.0)
        self.last_ambiguity = {}  # <-- NEW

    def _plc(self):
        import numpy as np
        V_idx = self.V_idx
        P = np.zeros((len(self.V), len(self.V)))
        for p, w in self.cfg.binary:
            X, Y = V_idx[p.head], V_idx[p.body[0]]
            P[X, Y] += w
        return np.linalg.inv(np.eye(len(self.V)) - P)

    def _ensure_empty_diagonal(self, upto):
        for i in range(upto + 1):
            for X in self.cfg.V:
                if (i, X, i) not in self.ppre:
                    self.ppre[i, X, i] = 1.0

    def _update_gamma_delta_for_j(self, j):
        i0_min = 0
        if self.max_backspan is not None:
            i0_min = max(0, j - self.max_backspan + 1)
        for i0 in range(i0_min, j):
            for Y in self.cfg.V:
                b = self.beta[i0, Y, j]
                if b == 0.0:
                    continue
                for X, Z, w in self._xz_from_y.get(Y, ()):
                    self.gamma[i0, j, X, Z] = self._add(self.gamma[i0, j, X, Z],
                                                         self._mul(w, b))
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
        if self.return_metrics:
            self.metrics.reset()

        tok = Sym(token_str) if not isinstance(token_str, Sym) else token_str
        self.tokens.append(tok)
        self.N += 1
        N = self.N
        self._ensure_empty_diagonal(N)

        # Track which (i,X,k) get written this append
        written_cells = set()
        yz_pairs = set()

        # (1) Terminal initialization
        for (head, body), w in self.cfg.terminal:
            if body[0] == tok:
                self.beta[N - 1, head, N] = self._add(self.beta[N - 1, head, N], w)
                written_cells.add((N - 1, head, N))

        # (2) CKY binary updates
        for span_len in range(2, N + 1):
            if self.max_backspan is not None and span_len > self.max_backspan:
                continue
            i0 = N - span_len
            k = N
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
                        yz_pairs.add((Y, Z))
                    if gamma_tmp == 0.0:
                        continue
                    for X, w in self._xz_from_yz.get((Y, Z), ()):
                        self.beta[i0, X, k] = self._add(self.beta[i0, X, k],
                                                         self._mul(gamma_tmp, w))
                        written_cells.add((i0, X, k))

        # (3) Incremental γ/δ update
        if N - 1 >= 1:
            if (self.max_window is None) or (N - 1 >= N - self.max_window):
                self._update_gamma_delta_for_j(N - 1)

        # (4) Prefix base case
        for X in self.cfg.V:
            for (p, w) in self.cfg.terminal:
                Y, v = p.head, p.body[0]
                if v == tok:
                    self.ppre[N - 1, X, N] = self._add(self.ppre[N - 1, X, N],
                                                       self._mul(self.E[X, Y], w))

        # (5) Prefix recurrence
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

        # ---- Ambiguity proxies (computed after all updates) ----
        active_beta_end = sum(1 for (i,X,k) in self.beta if k == N and self.beta[i,X,k] != 0)
        active_gamma_newslice = sum(1 for (i,j,X,Z) in self.gamma if j == N - 1 and self.gamma[i,j,X,Z] != 0)
        active_delta_newslice = sum(1 for (i,j,X,Z) in self.delta if j == N - 1 and self.delta[i,j,X,Z] != 0)
        active_write_targets = len(written_cells)
        active_YZ_pairs = len(yz_pairs)

        self.last_ambiguity = {
            "active_beta_end": active_beta_end,
            "active_gamma_newslice": active_gamma_newslice,
            "active_delta_newslice": active_delta_newslice,
            "active_write_targets": active_write_targets,
            "active_YZ_pairs": active_YZ_pairs,
        }

    def score_prefix(self):
        return self.ppre[0, self.cfg.S, self.N]

    def score_with_candidate(self, cand_token_str):
        snap_beta  = dict(self.beta)
        snap_ppre  = dict(self.ppre)
        snap_gamma = dict(self.gamma)
        snap_delta = dict(self.delta)
        self.append(cand_token_str)
        val = self.score_prefix()
        DD = self._DD_factory
        self.beta  = DD(lambda: 0.0); self.beta.update(snap_beta)
        self.ppre  = DD(lambda: 0.0); self.ppre.update(snap_ppre)
        self.gamma = DD(lambda: 0.0); self.gamma.update(snap_gamma)
        self.delta = DD(lambda: 0.0); self.delta.update(snap_delta)
        self.tokens.pop()
        self.N -= 1
        return val
