#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two approximate (W,B) runs: coarse vs finer.
Adds symmetric relative errors and monotonicity checks.
"""

import argparse, csv, math, pathlib
from collections import defaultdict

# optional deps
try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

EPS = 1e-12

def _to_float(x):
    try: return float(x)
    except: return float("nan")

def _read_csv(path: pathlib.Path):
    rows = {}
    with path.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            k = (int(r["seq_id"]), int(r["position"]))
            rows[k] = r
    return rows

def _align(A, B, key):
    xs, ys = [], []
    by_seq_x, by_seq_y = defaultdict(list), defaultdict(list)
    common = sorted(set(A.keys()) & set(B.keys()))
    for sid,pos in common:
        a = _to_float(A[(sid,pos)][key])
        b = _to_float(B[(sid,pos)][key])
        if a==a and b==b:  # both finite
            xs.append(a); ys.append(b)
            by_seq_x[sid].append(a)
            by_seq_y[sid].append(b)
    return xs, ys, by_seq_x, by_seq_y

def _rmse(x, y):
    if np is None:
        n = len(x); return math.sqrt(sum((a-b)**2 for a,b in zip(x,y))/max(n,1))
    x = np.asarray(x,float); y = np.asarray(y,float)
    return float(np.sqrt(np.mean((x-y)**2)))

def _rel_rmse_ref(ref, approx):  # asymmetric (old)
    if np is None:
        s=0.0; n=0
        for r,a in zip(ref,approx):
            denom = abs(r) if abs(r)>EPS else EPS
            s += ((a - r)/denom)**2; n+=1
        return math.sqrt(s/max(n,1))
    ref = np.asarray(ref,float); approx = np.asarray(approx,float)
    denom = np.maximum(np.abs(ref), EPS)
    return float(np.sqrt(np.mean(((approx-ref)/denom)**2)))

def _sym_rel_err(a, b):
    denom = 0.5*(abs(a)+abs(b))
    if denom < EPS: denom = EPS
    return abs(a-b)/denom

def _sym_rel_rmse(x, y):
    if np is None:
        s=0.0; n=0
        for a,b in zip(x,y):
            s += _sym_rel_err(a,b)**2; n+=1
        return math.sqrt(s/max(n,1))
    x = np.asarray(x,float); y = np.asarray(y,float)
    denom = 0.5*(np.abs(x)+np.abs(y))
    denom = np.maximum(denom, EPS)
    return float(np.sqrt(np.mean(((x-y)/denom)**2)))

def _p95_sym_rel_err(x, y):
    errs = [_sym_rel_err(a,b) for a,b in zip(x,y)]
    errs.sort()
    if not errs: return float("nan")
    idx = int(0.95*(len(errs)-1))
    return errs[idx]

def _frac_within_sym(x, y, thr):
    cnt=0; n=0
    for a,b in zip(x,y):
        if _sym_rel_err(a,b) <= thr: cnt += 1
        n += 1
    return cnt/max(n,1)

def _spearman(x, y):
    if spearmanr is not None:
        r,_ = spearmanr(x,y); return float(r)
    # fallback: rank then Pearson
    def _rank(vals):
        pairs = sorted((v,i) for i,v in enumerate(vals))
        ranks = [0.0]*len(vals); i=0
        while i < len(pairs):
            j=i; s=0.0
            while j<len(pairs) and pairs[j][0]==pairs[i][0]:
                s += j+1; j += 1
            avg = s/(j-i)
            for k in range(i,j): ranks[pairs[k][1]] = avg
            i=j
        return ranks
    rx, ry = _rank(x), _rank(y)
    n=len(rx); mx=sum(rx)/n; my=sum(ry)/n
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    denx=math.sqrt(sum((a-mx)**2 for a in rx))
    deny=math.sqrt(sum((b-my)**2 for b in ry))
    return (num/(denx*deny) if denx>0 and deny>0 else float("nan"))

def _monotonicity_stats(coarse, finer):
    # expect coarse <= finer (since pruning removes mass)
    violations = sum(1 for c,f in zip(coarse,finer) if f + 1e-15 < c)
    frac = violations / max(1,len(coarse))
    return violations, frac

def _report_pair(name, xs, ys, rel_thr_main, rel_thr_p95, want_rank=True):
    if len(xs)<3:
        print(f"{name}: not enough points"); return
    # classic (asymmetric) vs symmetric
    rmse = _rmse(xs, ys)
    rrmse_ref = _rel_rmse_ref(xs, ys)  # relative to coarse (legacy)
    srmse = _sym_rel_rmse(xs, ys)
    p95s  = _p95_sym_rel_err(xs, ys)
    frac  = _frac_within_sym(xs, ys, rel_thr_main)
    spr   = _spearman(xs, ys) if want_rank else float("nan")
    viols, fviol = _monotonicity_stats(xs, ys)

    print(f"\n{name} (coarse vs finer)")
    print(f"  N aligned                     : {len(xs)}")
    print(f"  RMSE                          : {rmse:.6g}")
    print(f"  Rel RMSE (ref=coarse, legacy) : {rrmse_ref*100:.3f}%")
    print(f"  SYMMETRIC Rel RMSE            : {srmse*100:.3f}%   (target ≤ {rel_thr_main*100:.1f}%)")
    print(f"  P95 symmetric rel error       : {p95s*100:.3f}%    (target ≤ {rel_thr_p95*100:.1f}%)")
    print(f"  Frac within {rel_thr_main*100:.1f}%         : {frac*100:.2f}%")
    if want_rank:
        print(f"  Spearman (rank)               : {spr:.4f}")
    print(f"  Monotonicity violations       : {viols}  ({fviol*100:.2f}%)  [expect ~0%]")

def _per_seq_summary(by_x, by_y, rel_thr_main):
    stats = []
    for sid in sorted(set(by_x.keys()) & set(by_y.keys())):
        x = by_x[sid]; y = by_y[sid]
        n = min(len(x),len(y))
        if n<3: continue
        x = x[:n]; y=y[:n]
        # per-seq symmetric MAPE
        s=0.0
        for a,b in zip(x,y):
            s += _sym_rel_err(a,b)
        smape = s/n
        spr = _spearman(x,y)
        # monotonicity per seq
        viols, fviol = _monotonicity_stats(x, y)
        stats.append((spr, smape, viols, fviol))
    if not stats:
        print("\n[per-sequence] not enough sequences"); return
    sprs  = [s for s,_,_,_ in stats if s==s]
    smaps = [m for _,m,_,_ in stats if m==m]
    fvs   = [f for *_,f in stats]
    def mean(v): return sum(v)/len(v) if v else float("nan")
    def median(v):
        v2 = sorted(v)
        if not v2: return float("nan")
        m = len(v2)//2
        return v2[m] if len(v2)%2==1 else 0.5*(v2[m-1]+v2[m])
    print("\n--- Per-sequence summaries (coarse vs finer) ---")
    print(f"  Sequences counted             : {len(stats)}")
    print(f"  Spearman  mean                : {mean(sprs):.4f}   median: {median(sprs):.4f}")
    print(f"  Symmetric MAPE mean           : {mean(smaps)*100:.3f}% median: {median(smaps)*100:.3f}%")
    print(f"  Monotonicity viol. mean frac  : {mean(fvs)*100:.2f}% median: {median(fvs)*100:.2f}%")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse_csv", type=pathlib.Path, required=True)
    ap.add_argument("--finer_csv",  type=pathlib.Path, required=True)
    ap.add_argument("--per_sequence", action="store_true")
    args = ap.parse_args()

    A = _read_csv(args.coarse_csv)
    B = _read_csv(args.finer_csv)

    # ΔCWU comparison
    dx, dy, d_byx, d_byy = _align(A, B, "delta_cwu")
    _report_pair("ΔCWU", dx, dy, rel_thr_main=0.05, rel_thr_p95=0.15, want_rank=True)
    if args.per_sequence:
        _per_seq_summary(d_byx, d_byy, rel_thr_main=0.05)

    # CWU(prefix) comparison
    px, py, p_byx, p_byy = _align(A, B, "cwu_prefix")
    _report_pair("CWU(prefix)", px, py, rel_thr_main=0.02, rel_thr_p95=0.05, want_rank=False)
    if args.per_sequence:
        _per_seq_summary(p_byx, p_byy, rel_thr_main=0.02)

if __name__ == "__main__":
    main()
