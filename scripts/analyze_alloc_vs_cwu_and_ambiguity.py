#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlate model allocation with parser compute & ambiguity proxies.

Predictors
  - alloc                : model-chosen layers per token
  - alloc_times_t        : alloc * effective_t, where effective_t = min(t, attention_cap) or t

Targets (if present in CSV)
  Compute / counts:
    - delta_cwu, cwu_prefix
    - ops_add, ops_mul
    - io_read_hit, io_read_miss, io_write_new, io_write_over
  Ambiguity proxies:
    - active_beta_end, active_gamma_newslice, active_delta_newslice
    - active_write_targets, active_YZ_pairs

Reports
  1) Pooled Pearson/Spearman for both predictors vs all targets
  2) Partial Pearson for alloc vs targets, controlling for t (linear)
  3) Within-position (t-fixed) Spearman for alloc vs targets
  4) Per-sequence Pearson/Spearman (mean/median), optional CSV dump
  5) Summary CSV over all targets (optional, --summary_out)
"""

import argparse
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# --- Which columns to try to analyze if present ---
COMPUTE_COLS = [
    "delta_cwu", "cwu_prefix",
    "ops_add", "ops_mul",
    "io_read_hit", "io_read_miss", "io_write_new", "io_write_over",
]

PROXY_COLS = [
    "active_beta_end",
    "active_gamma_newslice",
    "active_delta_newslice",
    "active_write_targets",
    "active_YZ_pairs",
]

ALL_TARGETS = COMPUTE_COLS + PROXY_COLS

# ------------- helpers -----------------
def _drop_nan_pair(x: np.ndarray, y: np.ndarray):
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m], m.sum()

def _safe_pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x, y, n = _drop_nan_pair(x, y)
    if n < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    try:
        return pearsonr(x, y)[0]
    except Exception:
        return np.nan

def _safe_spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x, y, n = _drop_nan_pair(x, y)
    if n < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    try:
        return spearmanr(x, y).correlation
    except Exception:
        return np.nan

def _partial_pearson(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Optional[float]:
    """
    Partial Pearson r(x,y | z): regress out z from x and y via simple linear projection.
    x' = x - Proj_z(x), y' = y - Proj_z(y); then Pearson(x', y').
    Robust to NaNs and constant vectors.
    """
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    n = x.size
    if n < 3:
        return np.nan

    # Add intercept for projection on [1, z]
    Z = np.vstack([np.ones_like(z), z]).T  # [n,2]

    def residuals(v: np.ndarray) -> Optional[np.ndarray]:
        if np.all(v == v[0]):
            # constant -> residuals are zero; correlation undefined
            return None
        try:
            # least squares: minimize ||Z b - v||
            b, *_ = np.linalg.lstsq(Z, v, rcond=None)
            v_hat = Z @ b
            r = v - v_hat
            return r
        except Exception:
            return None

    xr = residuals(x)
    yr = residuals(y)
    if xr is None or yr is None:
        return np.nan
    return _safe_pearson(xr, yr)

def _name_present_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

# ------------- reports (stdout) -----------------
def pooled_correlations(df: pd.DataFrame, targets: List[str]):
    print("=== POOLED correlations ===")
    # predictors
    preds = {
        "alloc": df["alloc"].to_numpy(float),
        "alloc_times_t": df["alloc_times_t"].to_numpy(float),
    }
    for pname, px in preds.items():
        print(f"\n--- Predictor = {pname} ---")
        for col in targets:
            y = df[col].to_numpy(float)
            pr = _safe_pearson(px, y)
            sr = _safe_spearman(px, y)
            _, _, n = _drop_nan_pair(px, y)
            print(f"{col:>18s}  Pearson: {pr: .4f}   Spearman: {sr: .4f}   N={n}")

def partial_correlations_alloc_given_t(df: pd.DataFrame, targets: List[str]):
    print("\n=== PARTIAL Pearson (alloc, target | t) ===")
    x = df["alloc"].to_numpy(float)
    z = df["effective_t"].to_numpy(float)
    for col in targets:
        y = df[col].to_numpy(float)
        pr = _partial_pearson(x, y, z)
        # N reported after dropping NaNs in all three
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        n = int(m.sum())
        print(f"{col:>18s}  partial-Pearson: {pr: .4f}   N={n}")

def within_position_correlations(df: pd.DataFrame, targets: List[str]):
    print("\n=== WITHIN-POSITION Spearman(alloc, target) (fix t) ===")
    rows = []
    for pos, g in df.groupby("position", sort=True):
        if g.shape[0] < 2:
            continue
        x = g["alloc"].to_numpy(float)
        rec = {"position": pos, "N": len(g)}
        for col in targets:
            y = g[col].to_numpy(float)
            rec[col] = _safe_spearman(x, y)
        rows.append(rec)
    if not rows:
        print("No positions with >=2 tokens.")
        return

    pos_df = pd.DataFrame(rows).sort_values("position")
    for col in targets:
        if col not in pos_df.columns:
            continue
        vals = pos_df[col].dropna().to_numpy()
        if vals.size == 0:
            continue
        mean_v = float(np.mean(vals))
        med_v  = float(np.median(vals))
        p25, p75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        print(f"{col:>18s}: positions={len(pos_df):3d} | mean={mean_v: .4f}  median={med_v: .4f}  p25={p25: .4f}  p75={p75: .4f}")

def per_sequence_correlations(df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    if "seq_id" not in df.columns:
        return pd.DataFrame()

    rows = []
    for sid, g in df.groupby("seq_id", sort=True):
        rec = {"seq_id": sid, "N": len(g)}
        x_alloc = g["alloc"].to_numpy(float)
        x_alloc_t = g["alloc_times_t"].to_numpy(float)
        for col in targets:
            y = g[col].to_numpy(float)
            rec[f"{col}__pearson(alloc)"] = _safe_pearson(x_alloc, y)
            rec[f"{col}__spearman(alloc)"] = _safe_spearman(x_alloc, y)
            rec[f"{col}__pearson(alloc*t)"] = _safe_pearson(x_alloc_t, y)
            rec[f"{col}__spearman(alloc*t)"] = _safe_spearman(x_alloc_t, y)
        rows.append(rec)
    per_seq = pd.DataFrame(rows).sort_values("seq_id").reset_index(drop=True)

    print("\n=== PER-SEQUENCE correlations (mean / median across sequences) ===")
    for col in targets:
        for pred_tag in ["alloc", "alloc*t"]:
            pcol = f"{col}__pearson({pred_tag})"
            scol = f"{col}__spearman({pred_tag})"
            if pcol in per_seq.columns:
                pvals = per_seq[pcol].dropna().to_numpy()
                svals = per_seq[scol].dropna().to_numpy()
                if pvals.size:
                    print(
                        f"{col:>18s}  [{pred_tag:7s}]  "
                        f"Pearson mean/med: {np.mean(pvals): .4f} / {np.median(pvals): .4f} | "
                        f"Spearman mean/med: {np.mean(svals): .4f} / {np.median(svals): .4f}"
                    )
    return per_seq

# ------------- summary table (for CSV) -----------------
def build_summary_table(df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    """
    Build a per-target summary table with:
      - pooled Pearson/Spearman for alloc and alloc_times_t
      - partial Pearson (alloc, target | t)
      - within-position Spearman stats
      - per-sequence Pearson/Spearman stats
    """
    rows = []

    alloc = df["alloc"].to_numpy(float)
    alloc_t = df["alloc_times_t"].to_numpy(float)
    t = df["effective_t"].to_numpy(float)

    groups_by_pos = {pos: g for pos, g in df.groupby("position", sort=True)}
    has_seq = "seq_id" in df.columns
    groups_by_seq = {sid: g for sid, g in df.groupby("seq_id", sort=True)} if has_seq else {}

    def _mean_med(arr: List[float]):
        if not arr:
            return (np.nan, np.nan)
        a = np.asarray(arr, dtype=float)
        return float(np.mean(a)), float(np.median(a))

    for col in targets:
        y = df[col].to_numpy(float)

        # pooled
        pooled_pr_alloc = _safe_pearson(alloc, y)
        pooled_sr_alloc = _safe_spearman(alloc, y)
        _, _, pooled_N_alloc = _drop_nan_pair(alloc, y)

        pooled_pr_alloc_t = _safe_pearson(alloc_t, y)
        pooled_sr_alloc_t = _safe_spearman(alloc_t, y)
        _, _, pooled_N_alloc_t = _drop_nan_pair(alloc_t, y)

        # partial (alloc, target | t)
        partial_r = _partial_pearson(alloc, y, t)
        m_part = np.isfinite(alloc) & np.isfinite(y) & np.isfinite(t)
        partial_N = int(m_part.sum())

        # within-position Spearman(alloc, target)
        within_vals = []
        for pos, g in groups_by_pos.items():
            if g.shape[0] < 2:
                continue
            x_pos = g["alloc"].to_numpy(float)
            y_pos = g[col].to_numpy(float)
            r_pos = _safe_spearman(x_pos, y_pos)
            if np.isfinite(r_pos):
                within_vals.append(r_pos)
        if within_vals:
            arr = np.asarray(within_vals, dtype=float)
            within_mean = float(np.mean(arr))
            within_median = float(np.median(arr))
            within_p25 = float(np.percentile(arr, 25))
            within_p75 = float(np.percentile(arr, 75))
            within_positions = int(arr.size)
        else:
            within_mean = within_median = within_p25 = within_p75 = np.nan
            within_positions = 0

        # per-sequence correlations
        per_seq_r_pearson_alloc = []
        per_seq_r_spearman_alloc = []
        per_seq_r_pearson_alloc_t = []
        per_seq_r_spearman_alloc_t = []
        if has_seq:
            for _, g in groups_by_seq.items():
                x_alloc_seq = g["alloc"].to_numpy(float)
                x_alloc_t_seq = g["alloc_times_t"].to_numpy(float)
                y_seq = g[col].to_numpy(float)

                r_pa = _safe_pearson(x_alloc_seq, y_seq)
                r_sa = _safe_spearman(x_alloc_seq, y_seq)
                r_pa_t = _safe_pearson(x_alloc_t_seq, y_seq)
                r_sa_t = _safe_spearman(x_alloc_t_seq, y_seq)

                if np.isfinite(r_pa):
                    per_seq_r_pearson_alloc.append(r_pa)
                if np.isfinite(r_sa):
                    per_seq_r_spearman_alloc.append(r_sa)
                if np.isfinite(r_pa_t):
                    per_seq_r_pearson_alloc_t.append(r_pa_t)
                if np.isfinite(r_sa_t):
                    per_seq_r_spearman_alloc_t.append(r_sa_t)

        mean_pa, med_pa = _mean_med(per_seq_r_pearson_alloc)
        mean_sa, med_sa = _mean_med(per_seq_r_spearman_alloc)
        mean_pa_t, med_pa_t = _mean_med(per_seq_r_pearson_alloc_t)
        mean_sa_t, med_sa_t = _mean_med(per_seq_r_spearman_alloc_t)
        per_seq_num = len(groups_by_seq) if has_seq else 0

        rows.append(
            {
                "target": col,
                # pooled
                "pooled_pearson_alloc": pooled_pr_alloc,
                "pooled_spearman_alloc": pooled_sr_alloc,
                "pooled_N_alloc": pooled_N_alloc,
                "pooled_pearson_alloc_times_t": pooled_pr_alloc_t,
                "pooled_spearman_alloc_times_t": pooled_sr_alloc_t,
                "pooled_N_alloc_times_t": pooled_N_alloc_t,
                # partial
                "partial_pearson_alloc_given_t": partial_r,
                "partial_N": partial_N,
                # within-position
                "within_pos_mean_spearman_alloc": within_mean,
                "within_pos_median_spearman_alloc": within_median,
                "within_pos_p25_spearman_alloc": within_p25,
                "within_pos_p75_spearman_alloc": within_p75,
                "within_pos_num_positions": within_positions,
                # per-sequence
                "per_seq_mean_pearson_alloc": mean_pa,
                "per_seq_median_pearson_alloc": med_pa,
                "per_seq_mean_spearman_alloc": mean_sa,
                "per_seq_median_spearman_alloc": med_sa,
                "per_seq_mean_pearson_alloc_times_t": mean_pa_t,
                "per_seq_median_pearson_alloc_times_t": med_pa_t,
                "per_seq_mean_spearman_alloc_times_t": mean_sa_t,
                "per_seq_median_spearman_alloc_times_t": med_sa_t,
                "per_seq_num_sequences": per_seq_num,
            }
        )

    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)

# ------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="Input CSV with alloc, position, targets, and ambiguity proxies.")
    ap.add_argument("--attention_cap", type=int, default=None,
                    help="Cap effective context length at this value; effective_t = min(position+1, cap).")
    ap.add_argument("--per_seq_out", type=str, default=None,
                    help="Optional path to write per-sequence correlation table (CSV).")
    ap.add_argument("--summary_out", type=str, default=None,
                    help="Optional path to write per-target summary correlations (CSV).")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Sanity
    need = {"position", "alloc"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # effective t
    t = df["position"].astype(int).to_numpy() + 1
    if args.attention_cap and args.attention_cap > 0:
        t = np.minimum(t, args.attention_cap)
    df["effective_t"] = t.astype(float)

    # predictors
    df["alloc_times_t"] = df["alloc"].astype(float) * df["effective_t"]

    # Figure out which targets exist
    targets_present = _name_present_cols(df, ALL_TARGETS)
    if not targets_present:
        raise ValueError(
            "No known target columns found in CSV. Expected some of: "
            + ", ".join(ALL_TARGETS)
        )

    # 1) pooled correlations (both predictors)
    pooled_correlations(df, targets_present)

    # 2) partial corr for alloc given t (linear control)
    partial_correlations_alloc_given_t(df, targets_present)

    # 3) within-position (fix t) Spearman(alloc, target)
    within_position_correlations(df, targets_present)

    # 4) per-sequence breakdown
    per_seq = per_sequence_correlations(df, targets_present)
    if args.per_seq_out and not per_seq.empty:
        per_seq.to_csv(args.per_seq_out, index=False)
        print(f"\n[OK] wrote per-sequence correlations to {args.per_seq_out}")

    # 5) summary CSV
    if args.summary_out:
        summary_df = build_summary_table(df, targets_present)
        summary_df.to_csv(args.summary_out, index=False)
        print(f"[OK] wrote summary correlations to {args.summary_out}")

if __name__ == "__main__":
    main()

