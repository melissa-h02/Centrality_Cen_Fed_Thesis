"""
═══════════════════════════════════════════════════════════════════════════
 Correction function — BEFORE vs. AFTER error, per centrality   [LOUVAIN]
═══════════════════════════════════════════════════════════════════════════

 What it does
 ------------
 Reads node_distortion.csv (from your Step 1 run) and, for each centrality,
 applies the simple correction:

     corrected = local * (1 + a * cut_ratio)

 where a (one number per metric) is fit from the data by least squares:

     a = sum(x*y) / sum(x*x),   with  x = local*cut_ratio,  y = global-local

 Then it measures how close the values are to the TRUE global values,
 before and after correction, using MAE (mean absolute error):

     MAE = average of | true value - your value |   over all nodes

 Outputs
 -------
   • correction_results.csv        (factor a, error before, error after, % improvement)
   • correction_before_after.png   (one before/after bar panel per centrality)
   • a printed table in the console

 Run
 ---
   put this file next to node_distortion.csv, then:  python correction_before_after.py
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
ALGO            = "Louvain (res=0.8)"     # Louvain only
ONLY_DEGREE_GE2 = True                     # same population as Step 1
METRICS         = ["degree", "betweenness", "closeness", "pagerank"]
LAB = {"degree": "Degree", "betweenness": "Betweenness",
       "closeness": "Closeness", "pagerank": "PageRank"}

OUT_CSV = "correction_results.csv"
OUT_PNG = "correction_before_after.png"

# colours
GREY, GREEN, TEXT, MUTED, GRID = "#90A4AE", "#2E7D32", "#1A2330", "#5B6B7A", "#E0E6EC"


def main():
    if not os.path.exists(NODE_CSV):
        print(f"Could not find {NODE_CSV} in this folder.")
        return

    d = pd.read_csv(NODE_CSV)
    if "algorithm" in d.columns:
        d = d[d["algorithm"] == ALGO]
    if ONLY_DEGREE_GE2 and "degree_undir" in d.columns:
        d = d[d["degree_undir"] >= 2]
    if d.empty:
        print(f"No rows for {ALGO} in {NODE_CSV}.")
        return

    rows = []
    for m in METRICS:
        gcol, lcol = f"global_val_{m}", f"local_val_{m}"
        if gcol not in d.columns or lcol not in d.columns:
            continue
        sub = d[[gcol, lcol, "cut_ratio"]].dropna()
        if sub.empty:
            continue

        g  = sub[gcol].to_numpy(float) # the TRUE global values
        l  = sub[lcol].to_numpy(float)  # the LOCAL (distorted) values
        cr = sub["cut_ratio"].to_numpy(float) # each node's cut ratio

        # fit the single factor a  (least squares)
        x = l * cr # x = local * cut_ratio   (for every node)
        y = g - l # y = global - local      (for every node)
        a = float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else 0.0  # a = Σ(x·y) / Σ(x·x)

        corrected = l * (1.0 + a * cr)  # corrected = local * (1 + a * cut_ratio)

        mae_before = float(np.mean(np.abs(g - l)))          # local vs true
        mae_after  = float(np.mean(np.abs(g - corrected)))  # corrected vs true
        impr = 100.0 * (mae_before - mae_after) / mae_before if mae_before > 0 else 0.0

        rows.append({"Centrality": LAB[m], "factor_a": round(a, 3),
                     "n": len(sub), "error_before": mae_before,
                     "error_after": mae_after,
                     "improvement_pct": round(impr, 1)})

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False, float_format="%.6f")

    # printed table
    print("=" * 78)
    print(f"  CORRECTION — before vs. after error   [{ALGO}, degree >= 2]")
    print("=" * 78)
    print(res.to_string(index=False))
    print("=" * 78)
    print(f"  saved: {OUT_CSV}")

    # ── 2x2 before/after chart, each metric on its own scale ────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor="white")
    fig.suptitle("Correction function — average error before vs. after   "
                 "(Louvain, degree \u2265 2)",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.98)

    for ax, (_, r) in zip(axes.flat, res.iterrows()):
        vals = [r["error_before"], r["error_after"]]
        bars = ax.bar(["Before", "After"], vals, width=0.55,
                      color=[GREY, GREEN], edgecolor="white",
                      linewidth=1.5, zorder=3)
        ax.set_title(f"{r['Centrality']}   \u2014   "
                     f"{r['improvement_pct']:.0f}% lower error",
                     fontsize=12.5, fontweight="bold", color=TEXT, pad=10)
        top = max(vals) if max(vals) > 0 else 1
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.03,
                    f"{v:.4g}", ha="center", va="bottom",
                    fontsize=11, color=TEXT, fontweight="bold")
        ax.set_ylim(0, top * 1.25)
        ax.set_facecolor("#F7F9FB")
        ax.tick_params(colors=MUTED, labelsize=11)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
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