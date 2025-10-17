#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump per-token CWU metrics to CSV using the incremental LRI parser.
Pure-stdlib (PyPy friendly): no numpy/scipy/matplotlib.

Writes columns:
  seq_id,position,token,alloc,cwu_prefix,delta_cwu,expected_delta_cwu,ops_add,ops_mul[,wall_time_ms]

Notes
- Set --compute_expected ONLY if you really need expected ΔCWU (slows down by |Vocab| factor).
- Use --max_window / --max_backspan to bound per-token work.
- Use --max_seqs to uniformly subsample sequences from a large JSONL.
"""

import argparse, json, pathlib, time
from typing import List, Dict, Iterable, Optional, Set

# Optional tqdm (pure Python). If missing, fall back to identity.
try:
    from tqdm import tqdm as _tqdm
    def itqdm(x, **kw): return _tqdm(x, **kw)
except Exception:
    def itqdm(x, **kw): return x

# rycolab fork (your repo)
from fastlri.base.cfg import CFG as RyCFG
from fastlri.parsing.incremental import IncrementalLRI
from fastlri.utils.cwu import cwu
from fastlri.utils.metrics import IO, OPS

ARROW = "→"

# ----------------------------- helpers -----------------------------
def parse_generation_text(text: str) -> List[str]:
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

def iter_jsonl_lines(path: pathlib.Path) -> Iterable[str]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

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
        return float(best_i + 1)
    elif mode == "expected":
        s = 0.0
        for i, p in enumerate(probs, start=1):
            s += i * float(p)
        return float(s)
    else:
        raise ValueError("alloc must be argmax|expected")

def monkeypatch_plc_from_cache(plc_path: pathlib.Path, cfg: RyCFG):
    """
    Avoid NumPy by serving PLC lookups from a JSON cache prepared earlier.
    JSON schema:
      {"ordered_V": ["@S","@A",...], "P_L": [[...],[...],...]}
    """
    data = json.loads(plc_path.read_text())
    V_cache = data["ordered_V"]
    P_L = data["P_L"]
    # map cfg.ordered_V to cache indices once
    idx_by_str = {s: i for i, s in enumerate(V_cache)}
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

    args = ap.parse_args()

    # Build CFG from rycolab text; infer start as first rule's LHS
    text = args.grammar_ry.read_text()
    first_rule = text.splitlines()[0]
    try:
        _prob, rest = first_rule.split(":", 1)
        start_sym = rest.split(ARROW, 1)[0].strip()
    except Exception:
        start_sym = "@S"
    cfg = RyCFG.from_string(text, start=start_sym)

    # Install PLC override to use cache (no NumPy dependency)
    monkeypatch_plc_from_cache(args.plc_cache, cfg)

    # Terminals
    term_list = maybe_load_terms(args.terms, cfg)

    # If subsampling, pre-choose which sequence indices to keep
    chosen_idxs: Optional[Set[int]] = None
    if args.max_seqs and args.max_seqs > 0:
        idxs = choose_sequence_indices(args.gens, args.max_seqs, args.seq_sample_seed)
        chosen_idxs = set(idxs)
        print(f"[INFO] Sampling {len(idxs)} sequences from file.")
        if args.save_sampled_ids:
            args.save_sampled_ids.parent.mkdir(parents=True, exist_ok=True)
            args.save_sampled_ids.write_text("\n".join(str(i) for i in idxs))
            print(f"[OK] wrote sampled indices to {args.save_sampled_ids}")

    # Iterate sequences
    try:
        total_lines = sum(1 for _ in args.gens.open())
    except Exception:
        total_lines = None

    # Write CSV header
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fout = args.out_csv.open("w")
    header = "seq_id,position,token,alloc,cwu_prefix,delta_cwu,expected_delta_cwu,ops_add,ops_mul"
    if args.log_time:
        header += ",wall_time_ms"
    fout.write(header + "\n")

    with args.gens.open() as f:
        for seq_id, line in enumerate(itqdm(f, total=total_lines, desc="sequences", dynamic_ncols=True)):
            if chosen_idxs is not None and seq_id not in chosen_idxs:
                continue
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw = parse_generation_text(obj.get("text",""))
            if not raw:
                continue
            toks = map_input_tokens_to_ry(raw)
            T = len(toks)
            probs_seq = obj.get("compute_probs", [])[:T]
            if not probs_seq:
                continue

            K = args.max_prefix_len if (args.max_prefix_len and args.max_prefix_len > 0) else T

            # Instantiate incremental parser with approximation knobs
            inc = IncrementalLRI(
                cfg,
                return_metrics=True,
                max_window=args.max_window,
                max_backspan=args.max_backspan
            )

            cum_cwu = 0.0
            desc = f"seq{seq_id} tokens"
            if args.max_window or args.max_backspan:
                desc += f" [win={args.max_window or '∞'}, back={args.max_backspan or '∞'}]"

            for t in itqdm(range(min(T, K)), desc=desc, leave=False, dynamic_ncols=True):
                tok = toks[t]
                cwu_prefix_val = cum_cwu

                t0 = time.perf_counter() if args.log_time else None
                inc.append(tok)
                if args.log_time:
                    wall_ms = (time.perf_counter() - t0) * 1000.0
                else:
                    wall_ms = None

                # metrics for the append()
                m_append = {"IO": dict(IO), "OPS": dict(OPS)}
                cwu_inc = cwu(m_append)
                cum_cwu += cwu_inc

                delta_cwu_val = cwu_inc
                alloc_val = compute_allocation_from_probs(probs_seq[t], args.alloc)

                # optional expected ΔCWU (expensive!)
                exp_delta_val = float("nan")
                if args.compute_expected:
                    probs, deltas = [], []
                    for a in term_list:
                        p_pa = inc.score_with_candidate(a)   # this calls append() internally and restores
                        m_sim = {"IO": dict(IO), "OPS": dict(OPS)}
                        deltas.append(cwu(m_sim))
                        probs.append(p_pa)
                    Z = sum(probs)
                    exp_delta_val = (sum((p/Z)*d for p, d in zip(probs, deltas)) if Z > 0 else 0.0)

                row = [
                    str(seq_id), str(t), tok,
                    f"{alloc_val}",
                    f"{cwu_prefix_val}",
                    f"{delta_cwu_val}",
                    f"{exp_delta_val}",
                    str(m_append["OPS"].get("add", 0)),
                    str(m_append["OPS"].get("mul", 0)),
                ]
                if args.log_time:
                    row.append(f"{wall_ms:.3f}")
                fout.write(",".join(row) + "\n")

    fout.close()
    print(f"[OK] wrote {args.out_csv}")

if __name__ == "__main__":
    main()
