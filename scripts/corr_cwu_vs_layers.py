#!/usr/bin/env python3
import argparse, json, pathlib, sys
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.grammar import PCFG, Nonterminal

# Rycolab fork (your instrumented version)
from fastlri.base.cfg import CFG as RyCFG
from fastlri.parsing.parser import Parser
from fastlri.parsing.incremental import IncrementalLRI
from fastlri.utils.cwu import cwu

ARROW = "→"  # rycolab expects this exact Unicode arrow


# ----------------------------- Correlation reporting -----------------------------
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


# ----------------------------- NLTK PCFG loading & CNF conversion -----------------------------
def load_nltk_pcfg(path: pathlib.Path) -> PCFG:
    return PCFG.fromstring(path.read_text())

def _is_nt(x) -> bool:
    return isinstance(x, Nonterminal)

def _nt_name(sym: Nonterminal) -> str:
    """Map any NLTK nonterminal (e.g., numeric '22') to an rycolab-looking NT '@22'."""
    return f"@{str(sym)}"

def _term_name(tok: str) -> str:
    """Map raw terminal string (e.g., '1') to lowercase 't1'."""
    return f"t{tok}"

def nltk_pcfg_to_ry_text_cnf(pcfg: PCFG) -> Tuple[str, str]:
    """
    Convert arbitrary NLTK PCFG to rycolab CNF text:
      - Rename NTs to '@<raw>' and terminals to 't<raw>' (lowercase),
      - Lift terminals from any RHS with len >= 2,
      - Right-binarize len > 2; first edge keeps original prob, helpers get 1.0.
    Returns (cnf_text, mapped_start_symbol).
    """
    lines: List[str] = []
    helper_id = 0

    term2pt: Dict[str, str] = {}
    def preterminal_for(term_tok: str) -> str:
        if term_tok not in term2pt:
            pt = f"T__{term_tok}"   # clearly a nonterminal
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
            rhs_syms = lifted  # all NTs now
        else:
            # len == 1 (terminal)
            rhs_syms = [_term_name(str(rhs[0]))]

        if len(rhs_syms) <= 2:
            lines.append(f"{prob}: {lhs_nt} {ARROW} {' '.join(rhs_syms)}")
            continue

        # Right-binarize
        curr_lhs = lhs_nt
        for i in range(len(rhs_syms) - 2):
            helper = f"{lhs_nt}__BIN{helper_id}__{i}"
            first = (i == 0)
            lines.append(f"{prob if first else 1.0}: {curr_lhs} {ARROW} {rhs_syms[i]} {helper}")
            curr_lhs = helper
        lines.append(f"1.0: {curr_lhs} {ARROW} {rhs_syms[-2]} {rhs_syms[-1]}")
        helper_id += 1

    start_nt = _nt_name(pcfg.start())
    return "\n".join(lines), start_nt

def ry_cfg_from_nltk_any(pcfg: PCFG) -> Tuple[RyCFG, str, str]:
    text, start_mapped = nltk_pcfg_to_ry_text_cnf(pcfg)
    cfg = RyCFG.from_string(text, start=start_mapped)
    return cfg, text, start_mapped


# ----------------------------- Generations parsing -----------------------------
def parse_generation_text(text: str) -> List[str]:
    # Must start with <start>, take tokens up to first <end>, drop initial <start>
    if not text.startswith("<start>"):
        return []
    upto_end = text.split("<end>")[0]
    toks = upto_end.split()[1:]  # drop the first <start>
    if "<start>" in toks:
        return []
    return toks

def map_input_tokens_to_ry(tokens: List[str]) -> List[str]:
    """Map numeric tokens like '1','2','3' to 't1','t2','t3' for the converted CNF."""
    out = []
    for tok in tokens:
        if tok.isdigit():
            out.append(f"t{tok}")
        else:
            out.append(tok.lower())
    return out

def compute_allocation_from_probs(probs: List[float], mode: str = "argmax") -> float:
    arr = np.asarray(probs, dtype=float)
    if arr.size == 0:
        return float("nan")
    if mode == "argmax":
        return float(np.argmax(arr)) + 1.0  # 1-based layer count
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


# ----------------------------- Helpers: terminals, token sampling, caching -----------------------------
def terminals_from_cfg(cfg: RyCFG) -> List[str]:
    return sorted({str(prod.body[0]) for (prod, _w) in cfg.terminal})

def choose_token_indices(T: int,
                         stride: int,
                         max_tokens: Optional[int],
                         mode: str,
                         rng: np.random.Generator) -> List[int]:
    """
    Choose which token positions to evaluate.
      - stride > 1: take 0, stride, 2*stride, ...
      - then cap to max_tokens (None means no cap)
      - mode: 'first' takes earliest, 'uniform' samples uniformly, 'random' same as 'uniform'
    """
    idx = list(range(0, T, max(1, stride)))
    if max_tokens is not None and len(idx) > max_tokens:
        if mode == "first":
            idx = idx[:max_tokens]
        else:  # uniform / random
            idx = list(rng.choice(idx, size=max_tokens, replace=False))
            idx.sort()
    return idx

class EvalCache:
    """Cache (p(prefix), cwu(prefix)) for strings to avoid recomputation."""
    def __init__(self, parser: Parser):
        self.parser = parser
        self.cache: Dict[str, Tuple[float, float]] = {}

    def eval(self, s: str) -> Tuple[float, float]:
        if s in self.cache:
            return self.cache[s]
        p, m = self.parser.lri_fast(s, chart=False, return_metrics=True)
        val = (p, cwu(m))
        self.cache[s] = val
        return val


# ----------------------------- CWU series per sequence (with sampling) -----------------------------
def cwu_prefix_series(parser: Parser,
                      tokens: List[str],
                      term_list: List[str],
                      rng: np.random.Generator,
                      token_indices: List[int],
                      term_sample_m: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute arrays (aligned to the *full* token list, filled only at sampled positions):
      cwu_pref, cwu_dlt, exp_dlt
    If term_sample_m is set and smaller than |term_list|, estimate Expected ΔCWU / entropy
    using self-normalized importance from a uniform terminal subsample.
    """
    T = len(tokens)
    cwu_pref = np.full(T, np.nan)
    cwu_dlt  = np.full(T, np.nan)
    exp_dlt  = np.full(T, np.nan)

    cache = EvalCache(parser)

    for t in tqdm(token_indices, desc=f"tokens (sampled)", leave=False, dynamic_ncols=True):
        prefix = " ".join(tokens[:t])
        _, c_pre = cache.eval(prefix)
        cwu_pref[t] = c_pre

        s_post = (prefix + " " + tokens[t]) if prefix else tokens[t]
        _, c_post = cache.eval(s_post)
        cwu_dlt[t] = c_post - c_pre

        # Expected ΔCWU over terminals (approx)
        if term_sample_m is not None and term_sample_m < len(term_list):
            sample_terms = list(rng.choice(term_list, size=term_sample_m, replace=False))
        else:
            sample_terms = term_list

        p_list = []
        d_list = []
        for a in sample_terms:
            s = (prefix + " " + a) if prefix else a
            p_pa, c_pa = cache.eval(s)
            p_list.append(p_pa)
            d_list.append(c_pa - c_pre)

        Z = float(np.sum(p_list))
        if Z > 0:
            q = np.array(p_list, dtype=float) / Z
            exp_dlt[t] = float(np.dot(q, np.array(d_list, dtype=float)))
        else:
            exp_dlt[t] = 0.0

    return cwu_pref, cwu_dlt, exp_dlt


# ----------------------------- Plotting -----------------------------
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

    scatter_with_fit(A, P, "CWU(prefix)",        out_base.with_name(out_base.stem + "_cwu_prefix.png"))
    scatter_with_fit(A, E, "Expected ΔCWU",      out_base.with_name(out_base.stem + "_expected_delta.png"))
    scatter_with_fit(A, D, "ΔCWU(realized)",     out_base.with_name(out_base.stem + "_delta_realized.png"))


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar_nltk", type=pathlib.Path, required=True, help="Path to NLTK PCFG file")
    ap.add_argument("--gens",          type=pathlib.Path, required=True, help="Path to JSONL generations")
    ap.add_argument("--alloc", choices=["argmax","expected"], default="argmax", help="How to collapse compute_probs")
    ap.add_argument("--out_csv",       type=pathlib.Path, default=None, help="Optional per-token CSV output")
    ap.add_argument("--dump_ry",       type=pathlib.Path, default=None, help="Optional path to write converted CNF grammar")
    ap.add_argument("--plot_png",      type=pathlib.Path, default=None, help="Base filename (no suffix) for scatter plots")

    # NEW: sampling controls
    ap.add_argument("--token_stride", type=int, default=1, help="Take every k-th token (>=1)")
    ap.add_argument("--max_tokens_per_seq", type=int, default=50, help="Cap tokens evaluated per sequence (None for all)")
    ap.add_argument("--token_sample_mode", choices=["first","uniform","random"], default="uniform", help="How to pick tokens when capping")
    ap.add_argument("--term_sample_m", type=int, default=16, help="Sample this many terminals for Expected ΔCWU (use all if smaller)")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed for sampling")

    args = ap.parse_args()

    # 1) Load grammar and convert to CNF text for rycolab (with renaming)
    pcfg = load_nltk_pcfg(args.grammar_nltk)
    try:
        rycfg, ry_text, start_mapped = ry_cfg_from_nltk_any(pcfg)
    except ValueError as e:
        print(f"[ERROR] grammar conversion failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Optional: dump converted grammar
    if args.dump_ry:
        args.dump_ry.parent.mkdir(parents=True, exist_ok=True)
        args.dump_ry.write_text(ry_text)
        print(f"[INFO] wrote converted CNF grammar to {args.dump_ry}")

    if not rycfg.in_cnf:
        print("[ERROR] Converted grammar is not CNF according to rycolab (cfg.in_cnf=False).", file=sys.stderr)
        print("        Use --dump_ry to examine the converted grammar text.", file=sys.stderr)
        sys.exit(1)

    parser = Parser(rycfg)
    term_list = terminals_from_cfg(rycfg)
    rng = np.random.default_rng(args.seed)

    # 2) Iterate generations with tqdm
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

            probs_seq = obj.get("compute_probs", [])
            probs_seq = probs_seq[:len(tokens_full)]
            if len(probs_seq) == 0:
                continue

            # choose token positions to evaluate
            idxs = choose_token_indices(
                T=len(tokens_full),
                stride=max(1, args.token_stride),
                max_tokens=None if args.max_tokens_per_seq is None or args.max_tokens_per_seq < 1 else args.max_tokens_per_seq,
                mode=args.token_sample_mode,
                rng=np.random.default_rng(args.seed + seq_id)
            )

            # compute alloc only at sampled positions
            alloc = np.full(len(tokens_full), np.nan, dtype=float)
            for t in idxs:
                probs = probs_seq[t]
                alloc[t] = compute_allocation_from_probs(probs, args.alloc)

            # run CWU series only for sampled positions (and estimate expected ΔCWU using terminal subsample)
            term_sample_m = None if args.term_sample_m is None or args.term_sample_m < 1 else args.term_sample_m
            cwu_pref, cwu_dlt, exp_dlt = cwu_prefix_series(
                parser, tokens_full, term_list,
                rng=np.random.default_rng(args.seed + 10_000 + seq_id),
                token_indices=idxs,
                term_sample_m=term_sample_m
            )

            # collect sampled points
            for t in idxs:
                all_alloc.append(alloc[t])
                all_cwu.append(cwu_pref[t])
                all_dlt.append(cwu_dlt[t])
                all_exp.append(exp_dlt[t])
                if args.out_csv:
                    rows.append((seq_id, t, tokens_full[t], alloc[t], cwu_pref[t], cwu_dlt[t], exp_dlt[t]))

    # 3) Correlations
    A = np.asarray(all_alloc, float)
    P = np.asarray(all_cwu,   float)
    D = np.asarray(all_dlt,   float)
    E = np.asarray(all_exp,   float)

    report("Allocated compute  vs  CWU(prefix)",     A, P)
    report("Allocated compute  vs  ΔCWU(realized)",  A, D)
    report("Allocated compute  vs  Expected ΔCWU",   A, E)

    # 4) Optional plots
    if args.plot_png:
        make_plots(A, P, D, E, args.plot_png)

    # 5) Optional CSV
    if args.out_csv and rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w") as f:
            f.write("seq_id,position,token,alloc,cwu_prefix,delta_cwu,expected_delta_cwu\n")
            for r in rows:
                f.write("{},{},{},{},{},{},{}\n".format(*r))
        print(f"\n[OK] wrote per-token rows to {args.out_csv}")

if __name__ == "__main__":
    main()
