from collections import defaultdict as dd
from fastlri.base.symbol import Sym
from fastlri.utils.metrics import reset_all, IO, OPS, add, mul

class IncrementalLRI:
    """
    Incremental prefix parser for lri_fast that reuses DP state across appends.
    Requires CNF. Works with your counted dicts & metrics when return_metrics=True.
    """
    def __init__(self, cfg, return_metrics=False):
        assert cfg.in_cnf
        self.cfg = cfg
        self.return_metrics = return_metrics
        self._add = add if return_metrics else (lambda a,b: a+b)
        self._mul = mul if return_metrics else (lambda a,b: a*b)

        # static (precomputed once)
        self.V = cfg.ordered_V
        self.V_idx = {X:i for i, X in enumerate(self.V)}
        self.W = dd(lambda: 0.0)  # binary weights W[X,Y,Z]
        for p, w in cfg.binary:
            X, Y, Z = p.head, p.body[0], p.body[1]
            self.W[X, Y, Z] = w
        self.P_L = self._plc()    # left-corner closure (numpy -> table)
        self.E = dd(lambda: 0.0)
        for X in self.cfg.V:
            for Y in self.cfg.V:
                self.E[X, Y] = self.P_L[self.V_idx[X], self.V_idx[Y]]

        # dynamic (grows with tokens)
        self.tokens = []          # list[Sym]
        self.N = 0
        self.beta = dd(lambda: 0.0)   # CKY chart β[i, X, k]
        self.ppre = dd(lambda: 0.0)   # prefix chart ppre[i, X, k]
        self.gamma = dd(lambda: 0.0)  # γ[i,j,X,Z]
        self.delta = dd(lambda: 0.0)  # δ[i,j,X,Z]

        # initialize empty prefix: k==i
        # (ppre[i,X,i] = 1 for all i, X — realized lazily when needed)

    def _plc(self):
        import numpy as np
        V = self.V
        V_idx = self.V_idx
        P = np.zeros((len(V), len(V)))
        for p, w in self.cfg.binary:
            X, Y = V_idx[p.head], V_idx[p.body[0]]
            P[X, Y] += w
        return np.linalg.inv(np.eye(len(V)) - P)

    def _ensure_empty_diagonal(self, upto):
        # ppre[i,X,i] = 1 for i up to 'upto'
        for i in range(upto+1):
            for X in self.cfg.V:
                # only set if not set (avoid IO overcount)
                if (i, X, i) not in self.ppre:
                    self.ppre[i, X, i] = 1.0

    def append(self, token_str):
        # convert token
        tok = Sym(token_str) if not isinstance(token_str, Sym) else token_str
        i = self.N  # new token index sits between positions (N,N+1) in 1-based span
        # extend length
        self.tokens.append(tok)
        self.N += 1
        N = self.N

        if self.return_metrics:
            reset_all()
        self._ensure_empty_diagonal(N)  # ensure ppre[i,X,i]=1 up to new N

        # 1) β terminal at [N-1, head, N]
        for (head, body), w in self.cfg.terminal:
            if body[0] == tok:
                self.beta[N-1, head, N] = self._add(self.beta[N-1, head, N], w)

        # 2) CKY binary fills only spans that end at N
        for span_len in range(2, N+1):
            i0 = N - span_len
            k  = N
            # sum over splits j
            for Y in self.cfg.V:
                for Z in self.cfg.V:
                    # accumulate γ_tmp = sum_j β[i0,Y,j] * β[j,Z,k]
                    gamma_tmp = 0.0
                    for j in range(i0+1, k):
                        b1 = self.beta[i0, Y, j]
                        if b1 == 0.0:
                            continue
                        b2 = self.beta[j, Z, k]
                        if b2 == 0.0:
                            continue
                        gamma_tmp = self._add(gamma_tmp, self._mul(b1, b2))
                    if gamma_tmp == 0.0:
                        continue
                    for X in self.cfg.V:
                        w = self.W[X, Y, Z]
                        if w != 0.0:
                            self.beta[i0, X, k] = self._add(self.beta[i0, X, k], self._mul(gamma_tmp, w))

        # 3) update γ, δ for pairs (i,j) that feed k=N
        #    j can be anything from i+1..N; we only need entries that will be used with k=N
        for i0 in range(0, N):
            for j in range(i0+1, N+1):
                # build γ[i0,j,*,*] only if some β[i0, Y, j] exists
                # we’ll lazily compute for all X,Z (standard triple loops)
                for p, w in self.cfg.binary:
                    X, Y, Z = p.head, p.body[0], p.body[1]
                    b = self.beta[i0, Y, j]
                    if b != 0.0:
                        self.gamma[i0, j, X, Z] = self._add(self.gamma[i0, j, X, Z], self._mul(w, b))
                # δ = E * γ
                for X in self.cfg.V:
                    for Y in self.cfg.V:
                        g_row_nonzero = False
                        for Z in self.cfg.V:
                            g = self.gamma[i0, j, Y, Z]
                            if g != 0.0:
                                g_row_nonzero = True
                                self.delta[i0, j, X, Z] = self._add(self.delta[i0, j, X, Z], self._mul(self.E[X, Y], g))
                        # small skip if entire row was zero

        # 4) prefix base case for the new 1-length span at end
        for X in self.cfg.V:
            for (p, w) in self.cfg.terminal:
                Y, v = p.head, p.body[0]
                if v == tok:
                    self.ppre[N-1, X, N] = self._add(self.ppre[N-1, X, N], self._mul(self.E[X, Y], w))

        # 5) prefix recurrence only for k=N
        for span_len in range(2, N+1):
            i0 = N - span_len
            k  = N
            for j in range(i0+1, k):
                for X in self.cfg.V:
                    for Z in self.cfg.V:
                        d = self.delta[i0, j, X, Z]
                        if d == 0.0:
                            continue
                        rhs = self.ppre[j, Z, k]
                        if rhs == 0.0:
                            continue
                        self.ppre[i0, X, k] = self._add(self.ppre[i0, X, k], self._mul(d, rhs))

        # done

    def score_prefix(self):
        # probability of current prefix
        return self.ppre[0, self.cfg.S, self.N]

    # Try a candidate terminal WITHOUT committing (compute-and-discard last-column updates)
    def score_with_candidate(self, cand_token_str):
        # shallow, last-column-only simulation:
        # duplicate the small set of keys that end at k=N+1 and the needed γ/δ slices, then drop
        # For simplicity & safety, we do a snapshot of dictionaries and restore. With |V| small, this is fine.
        snap_beta = dict()
        snap_ppre = dict()
        snap_gamma = dict()
        snap_delta = dict()
        # record sizes by filtering keys that will be touched (those whose right index is N+1, or pairs with j=N or N+1)
        # But for clarity in first version, snapshot entire dicts (still fast for your grammars)
        snap_beta.update(self.beta)
        snap_ppre.update(self.ppre)
        snap_gamma.update(self.gamma)
        snap_delta.update(self.delta)

        # do the append
        self.append(cand_token_str)
        val = self.score_prefix()

        # restore
        self.beta = dd(lambda: 0.0, snap_beta)
        self.ppre = dd(lambda: 0.0, snap_ppre)
        self.gamma = dd(lambda: 0.0, snap_gamma)
        self.delta = dd(lambda: 0.0, snap_delta)
        self.tokens.pop()
        self.N -= 1

        return val
