#!/usr/bin/env python3
import argparse, json, pathlib, sys
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.grammar import PCFG, Nonterminal

# rycolab fork + incremental
from fastlri.base.cfg import CFG as RyCFG
from fastlri.parsing.incremental import IncrementalLRI
from fastlri.utils.cwu import cwu
from fastlri.utils.metrics import IO, OPS  # read counters populated per append/sim

ARROW = "→"  # rycolab expects this Unicode arrow


# ----------------------------- small utilities -----------------------------
def report(name, x: np.ndarray, y: np.ndarray):
    print(f"\n=== Correlation: {name} ===")
    if x.size < 2 or y.size < 2:
        print("Not enough data."); return
    mask = np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[mask], y[mask]
    if x2.size < 2:
        print("Not enough finite data."); return
    pr, pp = pearsonr(x2, y2)
    sr, sp = spearmanr(x2, y2)
    print(f"Pearson : r={pr:.4f}, p={pp:.3g}")
    print(f"Spearman: ρ={sr:.4f}, p={sp:.3g}")


# ----------------------------- CNF conversion with renaming -----------------------------
def load_nltk_pcfg(path: pathlib.Path) -> PCFG:
    return PCFG.fromstring(path.read_text())

def _is_nt(x) -> bool:
    return isinstance(x, Nonterminal)

def _nt_name(sym: Nonterminal) -> str:
    return f"@{str(sym)}"  # make NTs clearly nonterminals for rycolab

def _term_name(tok: str) -> str:
    return f"t{tok}"      # terminals become lowercase like t1, t2, t3

def nltk_pcfg_to_ry_text_cnf(pcfg: PCFG) -> Tuple[str, str]:
    """
    Convert NLTK PCFG to rycolab CNF:
      - rename NTs (@22) and terminals (t1),
      - lift terminals from any len>=2 RHS via preterminals T__tX,
      - right-binarize len>2, keep original prob on first edge, 1.0 on helpers.
    """
    lines: List[str] = []
    helper_id = 0
    term2pt: Dict[str, str] = {}

    def preterminal_for(term_tok: str) -> str:
        if term_tok not in term2pt:
            pt = f"T__{term_tok}"
            term2pt[term_tok] = pt
            lines.append(f"1.0: {pt} {ARROW} {term_tok}")
        return term2pt[term_tok]

    for prod in pcfg.productions():
        lhs_nt = _nt_name(prod.lhs())
        rhs = list(prod.rhs())
        prob = prod.prob()

        if len(rhs) == 0:
            raise ValueError(f"Epsilon rule not supported: {prod}")
        if len(rhs) == 1 and _is_nt(rhs[0]):
            raise ValueError(f"Unary nonterminal rule not supported yet: {prod}")

        if len(rhs) >= 2:
            lifted: List[str] = []
            for sym in rhs:
                if _is_nt(sym):
                    lifted.append(_nt_name(sym))
                else:
                    lifted.append(preterminal_for(_term_name(str(sym))))
            rhs_syms = lifted
        else:
            # len == 1 (terminal)
            rhs_syms = [_term_name(str(rhs[0]))]

        if len(rhs_syms) <= 2:
            lines.append(f"{prob}: {lhs_nt} {ARROW} {' '.join(rhs_syms)}")
            continue

        # right-binarize
        curr_lhs = lhs_nt
        for i in range(len(rhs_syms) - 2):
            helper = f"{lhs_nt}__BIN{helper_id}__{i}"
            lines.append(f"{prob if i==0 else 1.0}: {curr_lhs} {ARROW} {rhs_syms[i]} {helper}")
            curr_lhs = helper
        lines.append(f"1.0: {curr_lhs} {ARROW} {rhs_syms[-2]} {rhs_syms[-1]}")
        helper_id += 1

    start_nt = _nt_name(pcfg.start())
    return "\n".join(lines), start_nt

def ry_cfg_from_nltk_any(pcfg: PCFG) -> Tuple[RyCFG, str, str]:
    text, start_mapped = nltk_pcfg_to_ry_text_cnf(pcfg)
    cfg = RyCFG.from_string(text, start=start_mapped)
    return cfg, text, start_mapped


# ----------------------------- generations parsing -----------------------------
def parse_generation_text(text: str) -> List[str]:
    if not text.startswith("<start>"):
        return []
    upto_end = text.split("<end>")[0]
    toks = upto_end.split()[1:]  # drop the first <start>
    if "<start>" in toks:
        return []
    return toks

def map_input_tokens_to_ry(tokens: List[str]) -> List[str]:
    return [f"t{tok}" if tok.isdigit() else tok.lower() for tok in tokens]

def compute_allocation_from_probs(probs: List[float], mode: str = "argmax") -> float:
    arr = np.asarray(probs, dtype=float)
    if arr.size == 0:
        return float("nan")
    if mode == "argmax":
        return float(np.argmax(arr)) + 1.0
    if mode == "expected":
        idx = np.arange(1, arr.size + 1, dtype=float)
        return float((idx * arr).sum())
    raise ValueError("mode must be 'argmax' or 'expected'")

def iter_jsonl(path: pathlib.Path) -> Iterable[Dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ----------------------------- plotting -----------------------------
def make_plots(A, P, D, E, out_base: pathlib.Path):
    out_base.parent.mkdir(parents=True, exist_ok=True)

    def scatter_with_fit(x, y, title, out_path):
        mask = np.isfinite(x) & np.isfinite(y)
        x2, y2 = x[mask], y[mask]
        if x2.size < 2:
            print(f"[WARN] Not enough points to plot {title}")
            return
        a, b = np.polyfit(x2, y2, 1)
        xs = np.linspace(x2.min(), x2.max(), 100)
        ys = a * xs + b

        plt.figure(figsize=(6, 4.5))
        plt.scatter(x2, y2, alpha=0.4, edgecolors="none")
        plt.plot(xs, ys, linewidth=2)
        plt.xlabel("Allocated compute (layers)")
        plt.ylabel(title)
        plt.title(f"{title} vs allocated compute")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[OK] wrote plot: {out_path}")

    scatter_with_fit(A, P, "CWU(prefix)",    out_base.with_name(out_base.stem + "_cwu_prefix.png"))
    scatter_with_fit(A, D, "ΔCWU(realized)", out_base.with_name(out_base.stem + "_delta_realized.png"))
    if np.isfinite(E).any():
        scatter_with_fit(A, E, "Expected ΔCWU", out_base.with_name(out_base.stem + "_expected_delta.png"))


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar_nltk", type=pathlib.Path, required=True)
    ap.add_argument("--gens",          type=pathlib.Path, required=True)
    ap.add_argument("--alloc", choices=["argmax","expected"], default="argmax")
    ap.add_argument("--out_csv", type=pathlib.Path, default=None)
    ap.add_argument("--dump_ry", type=pathlib.Path, default=None)
    ap.add_argument("--plot_png", type=pathlib.Path, default=None)

    # speed knob (optional): cap how deep into each sequence we go
    ap.add_argument("--max_prefix_len", type=int, default=0,
                    help="If >0, only consider first K tokens per sequence; 0 means all tokens.")
    # Expected ΔCWU is optional (with 3 terminals it's fine)
    ap.add_argument("--compute_expected", action="store_true",
                    help="If set, compute Expected ΔCWU over the full terminal set.")
    ap.add_argument("--seed", type=int, default=1234)

    args = ap.parse_args()

    # load + convert grammar
    pcfg = load_nltk_pcfg(args.grammar_nltk)
    rycfg, ry_text, start_mapped = ry_cfg_from_nltk_any(pcfg)

    if args.dump_ry:
        args.dump_ry.parent.mkdir(parents=True, exist_ok=True)
        args.dump_ry.write_text(ry_text)
        print(f"[INFO] wrote converted CNF grammar to {args.dump_ry}")

    if not rycfg.in_cnf:
        print("[ERROR] Converted grammar is not CNF (cfg.in_cnf=False).", file=sys.stderr)
        sys.exit(1)

    term_list = sorted({str(prod.body[0]) for (prod, _w) in rycfg.terminal})  # e.g., ['t1','t2','t3']

    # iterate sequences
    try:
        total_lines = sum(1 for _ in args.gens.open())
    except Exception:
        total_lines = None

    all_alloc, all_cwu, all_dlt, all_exp = [], [], [], []
    rows = []

    with args.gens.open() as f:
        for seq_id, line in enumerate(tqdm(f, total=total_lines, desc="sequences", dynamic_ncols=True)):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            raw_tokens = parse_generation_text(obj.get("text",""))
            if not raw_tokens:
                continue
            tokens_full = map_input_tokens_to_ry(raw_tokens)
            T = len(tokens_full)
            if T == 0:
                continue

            probs_seq = obj.get("compute_probs", [])[:T]
            if len(probs_seq) == 0:
                continue

            # optional truncation
            K = args.max_prefix_len if args.max_prefix_len and args.max_prefix_len > 0 else T

            # incremental parser for this sequence
            inc = IncrementalLRI(rycfg, return_metrics=True)
            cum_cwu = 0.0  # CWU(prefix) = sum of per-append CWU increments

            for t in tqdm(range(min(T, K)), desc="tokens (incremental)", leave=False, dynamic_ncols=True):
                tok = tokens_full[t]

                # CWU(prefix) BEFORE appending token t
                cwu_prefix_val = cum_cwu

                # Append real token t (triggers metrics for this step)
                _p_before = inc.score_prefix() if t > 0 else 1.0  # not used, but handy for sanity checks
                inc.append(tok)
                m_append = {"IO": dict(IO), "OPS": dict(OPS)}
                cwu_inc = cwu(m_append)  # CWU increment for this append
                cum_cwu += cwu_inc
                _p_after = inc.score_prefix()  # prob(prefix up to t)

                # realized ΔCWU
                delta_cwu_val = cwu_inc

                # allocated compute from your LM at this position
                alloc_val = compute_allocation_from_probs(probs_seq[t], args.alloc)

                # expected ΔCWU over next-token distribution (optional)
                exp_delta_val = np.nan
                if args.compute_expected:
                    probs = []
                    deltas = []
                    for a in term_list:
                        # simulate-and-rollback next token a (triggers metrics for simulated append)
                        p_pa = inc.score_with_candidate(a)
                        m_sim = {"IO": dict(IO), "OPS": dict(OPS)}
                        d_cwu = cwu(m_sim)  # CWU increment for that hypothetical append
                        probs.append(p_pa)
                        deltas.append(d_cwu)
                    Z = float(sum(probs))
                    exp_delta_val = float(sum((p/Z) * d for p, d in zip(probs, deltas))) if Z > 0 else 0.0

                # collect
                all_alloc.append(alloc_val)
                all_cwu.append(cwu_prefix_val)
                all_dlt.append(delta_cwu_val)
                all_exp.append(exp_delta_val)

                if args.out_csv:
                    rows.append((seq_id, t, tok, alloc_val, cwu_prefix_val, delta_cwu_val, exp_delta_val))

    # pack arrays
    A = np.asarray(all_alloc, float)
    P = np.asarray(all_cwu,   float)
    D = np.asarray(all_dlt,   float)
    E = np.asarray(all_exp,   float)

    # correlations
    report("Allocated compute  vs  CWU(prefix)",     A, P)
    report("Allocated compute  vs  ΔCWU(realized)",  A, D)
    if np.isfinite(E).any():
        report("Allocated compute  vs  Expected ΔCWU",   A, E)

    # plots
    if args.plot_png:
        args.plot_png.parent.mkdir(parents=True, exist_ok=True)
        make_plots(A, P, D, E, args.plot_png)

    # csv
    if args.out_csv and rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w") as f:
            f.write("seq_id,position,token,alloc,cwu_prefix,delta_cwu,expected_delta_cwu\n")
            for r in rows:
                f.write("{},{},{},{},{},{},{}\n".format(*r))
        print(f"\n[OK] wrote per-token rows to {args.out_csv}")

if __name__ == "__main__":
    main()
