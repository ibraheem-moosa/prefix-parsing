#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel dump of per-token CWU metrics to CSV using IncrementalLRI.
Pure-stdlib inside workers (PyPy friendly): no numpy/scipy/matplotlib.

Writes columns:
  seq_id,position,token,alloc,
  cwu_prefix,delta_cwu,expected_delta_cwu,
  ops_add,ops_mul,
  io_read_hit,io_read_miss,io_write_new,io_write_over,
  active_beta_end,active_gamma_newslice,active_delta_newslice,
  active_write_targets,active_YZ_pairs[,wall_time_ms]

Notes
- Set --compute_expected ONLY if you really need expected ΔCWU (slows down by |Vocab| factor).
- Use --max_window / --max_backspan to bound per-token work.
- Use --max_seqs to uniformly subsample sequences from a large JSONL.
- Use --n_jobs to control parallelism (process-based via joblib/loky).
"""

from __future__ import annotations
import argparse, json, pathlib, time
from typing import List, Dict, Iterable, Optional, Set, Tuple

# tqdm for sequence-level progress
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x  # fallback

from joblib import Parallel, delayed

# rycolab fork (your repo)
from fastlri.base.cfg import CFG as RyCFG
from fastlri.parsing.incremental import IncrementalLRI
from fastlri.utils.cwu import cwu

ARROW = "→"

# ----------------------------- helpers -----------------------------
def parse_generation_text(text: str) -> List[str]:
    """
    Return token list between the first <start> and the first <end>.
    Reject if malformed (e.g., nested <start>).
    """
    if not text or not text.startswith("<start>"):
        return []
    upto_end = text.split("<end>")[0]
    toks = upto_end.split()[1:]  # drop the first <start>
    if "<start>" in toks:
        return []
    return toks

def map_input_tokens_to_ry(tokens: List[str]) -> List[str]:
    # numeric -> t<num>, else lowercase pass-through
    return [f"t{tok}" if tok.isdigit() else tok.lower() for tok in tokens]

def iter_jsonl_lines(path: pathlib.Path) -> Iterable[Tuple[int,str]]:
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                yield i, line

def compute_allocation_from_probs(probs, mode="argmax") -> float:
    # pure-Python
    if not probs:
        return float("nan")
    if mode == "argmax":
        best_i, best_v = 0, float("-inf")
        for i, v in enumerate(probs):
            v = float(v)
            if v > best_v:
                best_i, best_v = i, v
        return float(best_i + 1)  # 1-based number of layers
    elif mode == "expected":
        s = 0.0
        for i, p in enumerate(probs, start=1):
            s += i * float(p)
        return float(s)
    else:
        raise ValueError("alloc must be argmax|expected")

def build_cfg_from_text(grammar_text: str) -> RyCFG:
    # infer start from the first rule's LHS
    first_rule = grammar_text.splitlines()[0]
    try:
        _prob, rest = first_rule.split(":", 1)
        start_sym = rest.split(ARROW, 1)[0].strip()
    except Exception:
        start_sym = "@S"
    return RyCFG.from_string(grammar_text, start=start_sym)

def monkeypatch_plc_from_obj(plc_obj: Dict, cfg: RyCFG):
    """
    Serve P_L lookups from a JSON cache (no NumPy in workers).
    JSON schema:
      {"ordered_V": ["@S","@A",...], "P_L": [[...],[...],...]}
    """
    V_cache = plc_obj["ordered_V"]
    P_L = plc_obj["P_L"]
    idx_by_str = {s: i for i, s in enumerate(V_cache)}
    # remap indices from cfg.ordered_V to cache rows
    remap = [idx_by_str[str(X)] for X in cfg.ordered_V]

    class PLCWrapper:
        def __getitem__(self, ij):
            i, j = ij
            return P_L[remap[i]][remap[j]]

    def _plc_override(self):  # self is IncrementalLRI
        return PLCWrapper()

    # install override
    IncrementalLRI._plc = _plc_override

def maybe_load_terms(terms_path: Optional[pathlib.Path], cfg: RyCFG) -> List[str]:
    if terms_path and terms_path.exists():
        return [ln.strip() for ln in terms_path.read_text().splitlines() if ln.strip()]
    # fallback: derive from cfg terminals
    return sorted({str(prod.body[0]) for (prod, _w) in cfg.terminal})

def choose_sequence_indices(jsonl_path: pathlib.Path, k: int, seed: int) -> List[int]:
    """Uniformly sample k distinct sequence indices in [0..L-1], reproducibly."""
    if k <= 0:
        return []
    L = sum(1 for _ in jsonl_path.open())
    if k >= L:
        return list(range(L))
    import random
    rnd = random.Random(seed)
    idxs = list(range(L))
    rnd.shuffle(idxs)
    idxs = sorted(idxs[:k])
    return idxs

# ----------------------------- worker -----------------------------
def process_one_sequence(
    seq_id: int,
    line: str,
    grammar_text: str,
    plc_obj: Dict,
    terms_list: List[str],
    alloc_mode: str,
    max_prefix_len: int,
    compute_expected: bool,
    max_window: Optional[int],
    max_backspan: Optional[int],
    log_time: bool
) -> List[str]:
    """
    Return list of CSV lines (excluding header) for this sequence.
    Everything needed is passed in (no globals).
    """
    # --- per-process setup ---
    cfg = build_cfg_from_text(grammar_text)
    monkeypatch_plc_from_obj(plc_obj, cfg)
    terms = terms_list  # already built in parent

    obj = json.loads(line)
    raw = parse_generation_text(obj.get("text",""))
    if not raw:
        return []
    toks = map_input_tokens_to_ry(raw)
    T = len(toks)
    probs_seq = obj.get("compute_probs", [])[:T]
    if not probs_seq:
        return []

    K = max_prefix_len if (max_prefix_len and max_prefix_len > 0) else T

    inc = IncrementalLRI(
        cfg,
        return_metrics=True,
        max_window=max_window,
        max_backspan=max_backspan
    )

    rows = []
    cum_cwu = 0.0

    for t in range(min(T, K)):
        tok = toks[t]
        cwu_prefix_val = cum_cwu

        t0 = time.perf_counter() if log_time else None
        inc.append(tok)
        wall_ms = (time.perf_counter() - t0) * 1000.0 if log_time else None

        # per-append metrics
        m_append = inc.metrics.snapshot()
        cwu_inc = cwu(m_append)
        cum_cwu += cwu_inc

        delta_cwu_val = cwu_inc
        alloc_val = compute_allocation_from_probs(probs_seq[t], alloc_mode)

        # optional expected ΔCWU (expensive!)
        exp_delta_val = float("nan")
        if compute_expected:
            probs, deltas = [], []
            for a in terms:
                p_pa = inc.score_with_candidate(a)   # internal append+restore
                m_sim = inc.metrics.snapshot()       # metrics from that simulated append
                deltas.append(cwu(m_sim))
                probs.append(p_pa)
            Z = sum(probs)
            exp_delta_val = (sum((p/Z)*d for p, d in zip(probs, deltas)) if Z > 0 else 0.0)

        # ambiguity proxies (populated by IncrementalLRI.append)
        amb = inc.last_ambiguity

        row = [
            str(seq_id), str(t), tok,
            f"{alloc_val}",
            f"{cwu_prefix_val}",
            f"{delta_cwu_val}",
            f"{exp_delta_val}",
            str(m_append["OPS"].get("add", 0)),
            str(m_append["OPS"].get("mul", 0)),
            str(m_append["IO"].get("chart_read_hit", 0)),
            str(m_append["IO"].get("chart_read_miss", 0)),
            str(m_append["IO"].get("chart_write_new", 0)),
            str(m_append["IO"].get("chart_write_over", 0)),
            str(amb.get("active_beta_end", 0)),
            str(amb.get("active_gamma_newslice", 0)),
            str(amb.get("active_delta_newslice", 0)),
            str(amb.get("active_write_targets", 0)),
            str(amb.get("active_YZ_pairs", 0)),
        ]
        if log_time:
            row.append(f"{wall_ms:.3f}")
        rows.append(",".join(row))

    return rows

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar_ry",   type=pathlib.Path, required=True, help="rycolab CNF grammar text")
    ap.add_argument("--plc_cache",    type=pathlib.Path, required=True, help="JSON with ordered_V and P_L")
    ap.add_argument("--terms",        type=pathlib.Path, default=None,   help="Optional terminals list")
    ap.add_argument("--gens",         type=pathlib.Path, required=True, help="Generations .jsonl")
    ap.add_argument("--alloc",        choices=["argmax","expected"], default="argmax")
    ap.add_argument("--max_prefix_len", type=int, default=0, help="Cap tokens per sequence (0 = all)")
    ap.add_argument("--compute_expected", action="store_true", help="Also compute Expected ΔCWU (expensive)")
    ap.add_argument("--out_csv",      type=pathlib.Path, required=True)

    # Approximation knobs
    ap.add_argument("--max_window", type=int, default=None,
                    help="Only consider splits j in [k-max_window, k-1] (approx).")
    ap.add_argument("--max_backspan", type=int, default=None,
                    help="Only spans with length <= max_backspan (approx).")

    # Sequence subsampling
    ap.add_argument("--max_seqs", type=int, default=0,
                    help="If >0, uniformly sample this many sequences from the JSONL.")
    ap.add_argument("--seq_sample_seed", type=int, default=1234,
                    help="Seed for sequence sampling.")
    ap.add_argument("--save_sampled_ids", type=pathlib.Path, default=None,
                    help="Optional path to save chosen sequence indices (one per line).")

    # Optional: record wall time
    ap.add_argument("--log_time", action="store_true",
                    help="Record per-token wall_time_ms (adds a little overhead).")

    # Parallelism
    ap.add_argument("--n_jobs", type=int, default=-1,
                    help="Number of processes for joblib. -1 = all cores.")

    args = ap.parse_args()

    # Preload grammar text & PLC JSON once in the parent
    grammar_text = args.grammar_ry.read_text()
    plc_obj = json.loads(args.plc_cache.read_text())

    # Build a temporary CFG in parent to derive terms if not provided
    cfg_tmp = build_cfg_from_text(grammar_text)
    terms_list = ( [ln.strip() for ln in args.terms.read_text().splitlines() if ln.strip()]
                   if args.terms and args.terms.exists()
                   else sorted({str(prod.body[0]) for (prod, _w) in cfg_tmp.terminal}) )

    # Optional: sample sequences deterministically
    chosen_idxs: Optional[Set[int]] = None
    if args.max_seqs and args.max_seqs > 0:
        idxs = choose_sequence_indices(args.gens, args.max_seqs, args.seq_sample_seed)
        chosen_idxs = set(idxs)
        print(f"[INFO] Sampling {len(idxs)} sequences from file.")
        if args.save_sampled_ids:
            args.save_sampled_ids.parent.mkdir(parents=True, exist_ok=True)
            args.save_sampled_ids.write_text("\n".join(str(i) for i in idxs))
            print(f"[OK] wrote sampled indices to {args.save_sampled_ids}")

    # Collect tasks (seq_id, line)
    tasks: List[Tuple[int,str]] = []
    for seq_id, line in iter_jsonl_lines(args.gens):
        if chosen_idxs is not None and seq_id not in chosen_idxs:
            continue
        tasks.append((seq_id, line))

    # Run in parallel (sequence-level)
    results = Parallel(n_jobs=args.n_jobs, prefer="processes", batch_size=1)(
        delayed(process_one_sequence)(
            seq_id, line,
            grammar_text, plc_obj, terms_list,
            args.alloc, args.max_prefix_len, args.compute_expected,
            args.max_window, args.max_backspan, args.log_time
        )
        for (seq_id, line) in tqdm(tasks, desc="dispatching", dynamic_ncols=True)
    )

    # Write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w") as fout:
        header = (
            "seq_id,position,token,alloc,"
            "cwu_prefix,delta_cwu,expected_delta_cwu,"
            "ops_add,ops_mul,"
            "io_read_hit,io_read_miss,io_write_new,io_write_over,"
            "active_beta_end,active_gamma_newslice,active_delta_newslice,"
            "active_write_targets,active_YZ_pairs"
        )
        if args.log_time:
            header += ",wall_time_ms"
        fout.write(header + "\n")

        # keep original file order by seq_id
        for rows in tqdm(results, desc="writing", dynamic_ncols=True):
            for r in rows:
                fout.write(r + "\n")

    print(f"[OK] wrote {args.out_csv}")

if __name__ == "__main__":
    main()
