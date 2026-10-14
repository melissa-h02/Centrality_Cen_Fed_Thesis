"""
═══════════════════════════════════════════════════════════════════════════
 Correction formulas — HEAD TO HEAD, per centrality   [LOUVAIN]
═══════════════════════════════════════════════════════════════════════════

 Compares THREE correction formulas by measuring the average error (MAE)
 before and after each one, per centrality. Lower error after = better.

 METHOD 1 — "size + cut"  (YOUR new formula)
     corrected = local * (n_global - 1) / ((client_size - 1) * (1 - cut_ratio))
     Fixes two things at once:
       • the (n-1) normalization gap between the big graph and a small client
       • the edge-loss shrinkage, via 1/(1-cut_ratio)

 METHOD 2 — "exact"  (the proven formula; exact for degree)
     degree:  corrected = local + cut_edges     (Step 2 proof -> 100% exact,
              using UNDIRECTED degree so it matches how cut_edges is counted)
     others:  corrected = local / (1 - cut_ratio)   (no such identity exists)

 METHOD 3 — "fitted"  (one factor a per metric, least squares)
     corrected = local * (1 + a * cut_ratio),   a = Σ(x·y)/Σ(x·x)

 MAE = average of | true global value - corrected value |  over all nodes.

 Output: correction_compare.csv  + a printed table  + correction_compare.png
 Run: put next to node_distortion.csv, then  python correction_compare.py
═══════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────
NODE_CSV        = "node_distortion.csv"
ALGO            = "Louvain (res=0.8)"
ONLY_DEGREE_GE2 = True
FLOOR           = 0.1          # never divide by less than this (cut_ratio -> 1 guard)
METRICS         = ["degree", "betweenness", "closeness", "pagerank"]
LAB = {"degree": "Degree", "betweenness": "Betweenness",
       "closeness": "Closeness", "pagerank": "PageRank"}

OUT_CSV = "correction_compare.csv"
OUT_PNG = "correction_compare.png"


def mae(true, pred):
    return float(np.mean(np.abs(true - pred)))


def main():
    if not os.path.exists(NODE_CSV):
        print(f"Could not find {NODE_CSV} in this folder.")
        return

    full = pd.read_csv(NODE_CSV)
    algo_all = full[full["algorithm"] == ALGO] if "algorithm" in full.columns else full

    # n_global = total nodes in the graph for this algorithm
    n_global = len(algo_all)

    # client_size = number of nodes in each client (ALL nodes, before any filter)
    client_size_map = algo_all.groupby("client").size().to_dict()

    d = algo_all.copy()
    if ONLY_DEGREE_GE2 and "degree_undir" in d.columns:
        d = d[d["degree_undir"] >= 2]
    d = d.copy()
    d["client_size"] = d["client"].map(client_size_map)

    print("=" * 100)
    print(f"  CORRECTION FORMULAS — head to head   [{ALGO}, degree >= 2,  "
          f"n_global = {n_global}]")
    print("  (error = average |global - value|;  lower AFTER is better)")
    print("=" * 100)
    print(f"  {'Metric':<12}{'before':>12}"
          f"{'M1 size+cut':>14}{'M2 exact':>12}{'M3 fitted':>12}"
          f"   best method")
    print(f"  {'-'*12}{'-'*12}{'-'*14}{'-'*12}{'-'*12}   {'-'*11}")

    rows = []
    for m in METRICS:
        gcol, lcol = f"global_val_{m}", f"local_val_{m}"
        if gcol not in d.columns or lcol not in d.columns:
            continue

        cols = [gcol, lcol, "cut_ratio", "client_size"]
        # For DEGREE we also need the exact cut-edge count and the undirected
        # degree, so Method 2 can use the proven identity (see below).
        if m == "degree":
            for c in ("cut_edges", "degree_undir"):
                if c in d.columns and c not in cols:
                    cols.append(c)
        sub = d[cols].dropna()
        if sub.empty:
            continue

        g  = sub[gcol].to_numpy(float)
        l  = sub[lcol].to_numpy(float)
        cr = sub["cut_ratio"].to_numpy(float)
        cs = sub["client_size"].to_numpy(float)

        keep = np.maximum(1.0 - cr, FLOOR)      # guarded (1 - cut_ratio)

        # METHOD 1 — your size + cut formula
        size_term = (n_global - 1.0) / np.maximum(cs - 1.0, 1.0)
        m1 = l * size_term / keep

        # METHOD 2 — EXACT formula:  global = local + cut_edges
        # ---------------------------------------------------------------
        # Step 2 proof:  local = global - cut_edges  ->  global = local + cut_edges
        # This is EXACT for degree, but only when degree is measured the SAME
        # way as cut_edges — i.e. UNDIRECTED. node_distortion.csv stores the
        # DIRECTED degree (in + out), while cut_ratio/cut_edges are undirected,
        # so we rebuild the undirected degree pair here:
        #     global (undirected) = degree_undir
        #     local  (undirected) = degree_undir - cut_edges
        # Doing this makes the degree correction land at exactly 100%.
        # For the other metrics there is no such identity, so Method 2 falls
        # back to the divide form.
        if m == "degree" and {"cut_edges", "degree_undir"} <= set(sub.columns):
            ce = sub["cut_edges"].to_numpy(float)
            g  = sub["degree_undir"].to_numpy(float)      # true undirected degree
            l  = g - ce                                    # what the client sees
            m1 = l * size_term / keep                      # recompute M1 on same basis
            m2 = l + ce                                    # EXACT -> error 0
        else:
            m2 = l / keep                                  # divide (other metrics)

        # METHOD 3 — fitted single factor a
        x = l * cr
        y = g - l
        a = float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else 0.0
        m3 = l * (1.0 + a * cr)

        e_before = mae(g, l)
        e1, e2, e3 = mae(g, m1), mae(g, m2), mae(g, m3)

        errs = {"M1 size+cut": e1, "M2 exact": e2, "M3 fitted": e3}
        best = min(errs, key=errs.get)

        def pct(e):
            return 100.0 * (e_before - e) / e_before if e_before > 0 else 0.0

        print(f"  {LAB[m]:<12}{e_before:>12.4g}"
              f"{e1:>14.4g}{e2:>12.4g}{e3:>12.4g}   {best}")

        rows.append({
            "Centrality": LAB[m], "error_before": e_before,
            "M1_size_cut": e1, "M2_divide": e2, "M3_fitted": e3,
            "impr_M1_pct": round(pct(e1), 1),
            "impr_M2_pct": round(pct(e2), 1),
            "impr_M3_pct": round(pct(e3), 1),
            "best_method": best, "a_fitted": round(a, 3),
        })

    print("=" * 100)
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False, float_format="%.6f")
    print(f"  saved: {OUT_CSV}")

    # ── SECOND TABLE: percentage better / worse (cleaner to read) ───────
    # positive % = error went DOWN (better); negative % = error went UP (worse)
    print("\n" + "=" * 100)
    print(f"  SAME RESULT AS % CHANGE IN ERROR   [{ALGO}, degree >= 2]")
    print("  positive = BETTER (error reduced)   |   negative = WORSE (error grew)")
    print("=" * 100)
    print(f"  {'Metric':<12}{'M1 size+cut':>15}{'M2 exact':>13}{'M3 fitted':>13}"
          f"   best method")
    print(f"  {'-'*12}{'-'*15}{'-'*13}{'-'*13}   {'-'*11}")
    for r in rows:
        def tag(p):
            return f"{p:+.0f}%"
        print(f"  {r['Centrality']:<12}"
              f"{tag(r['impr_M1_pct']):>15}{tag(r['impr_M2_pct']):>13}"
              f"{tag(r['impr_M3_pct']):>13}   {r['best_method']}")
    print("=" * 100)
    print("  (a value like +84% means the error dropped by 84%; -600% means the")
    print("   error became 6x larger — that formula badly over-corrected.)")

    # ── chart: for each metric, BEFORE vs the BEST method's AFTER ───────
    if not res.empty:
        TEXT, MUTED, GRID, GREY, GREEN = "#1A2330", "#5B6B7A", "#E0E6EC", "#90A4AE", "#2E7D32"
        method_err = {"M1 size+cut": "M1_size_cut", "M2 exact": "M2_divide",
                      "M3 fitted": "M3_fitted"}
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor="white")
        fig.suptitle("Best correction per centrality — error before vs. after   "
                     "(Louvain, degree \u2265 2)",
                     fontsize=15, fontweight="bold", color=TEXT, y=0.98)
        for ax, (_, r) in zip(axes.flat, res.iterrows()):
            best = r["best_method"]
            after = r[method_err[best]]
            before = r["error_before"]
            impr = 100.0 * (before - after) / before if before > 0 else 0.0
            vals = [before, after]
            bars = ax.bar(["Before", f"After\n({best})"], vals, width=0.55,
                          color=[GREY, GREEN], edgecolor="white",
                          linewidth=1.5, zorder=3)
            ax.set_title(f"{r['Centrality']}   \u2014   {impr:.0f}% lower error",
                         fontsize=12.5, fontweight="bold", color=TEXT, pad=10)
            top = max(vals) if max(vals) > 0 else 1
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + top * 0.03, f"{v:.4g}",
                        ha="center", va="bottom", fontsize=11, color=TEXT, fontweight="bold")
            ax.set_ylim(0, top * 1.25)
            ax.set_facecolor("#F7F9FB")
            ax.tick_params(colors=MUTED, labelsize=10)
            ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            for s in ["left", "bottom"]:
                ax.spines[s].set_color(GRID)
            ax.set_ylabel("avg error (MAE)", color=MUTED, fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  saved: {OUT_PNG}")


if __name__ == "__main__":
    main()