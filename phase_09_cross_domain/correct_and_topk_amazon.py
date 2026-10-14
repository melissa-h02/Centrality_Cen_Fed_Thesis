"""
═══════════════════════════════════════════════════════════════════════════
 AMAZON — CENTRALITY CORRECTION + RANKING RECOVERY  (the real generalization test)
═══════════════════════════════════════════════════════════════════════════

 Classification turned out to be a noisy, unreliable way to measure whether
 centralities carry structural information (it flipped sign between datasets
 and between global/local). RANKING is the clean instrument — and it is where
 the correction functions proved their worth on the IDS data.

 This script asks the same label-free question on Amazon:

   When the co-purchase graph is split across 3 clients, the local centrality
   of each product is distorted. Does that distortion change WHICH products
   rank as most central — and can our correction restore the true ranking?

 It reuses the SAME correction methods as the IDS study (nothing IDS-specific):
     - no correction        (safety floor)
     - M2 exact             (degree only: true = local + cut_edges)
     - M2b divide           (local / (1 - cut_ratio))
     - M3 fitted factor a   (local * (1 + a*cut_ratio), a fit by MAE)
     - LIN  linear regression on (local, cut_ratio, size_frac)
     - RF   random forest    on the same three inputs
   The winner is chosen PER centrality on held-out nodes (no leakage), exactly
   as before. The method is a graph tool, not an IDS tool — only the fitted
   numbers change.

 Then, Part 2 (label-free):
     Overlap@K = how much the top-K by (local / corrected) matches the
                 top-K by the TRUE global centrality.  1.0 = identical.
   If corrected overlaps the truth more than local does, correction recovers
   the federated ranking — the same finding as IDS, now in a second domain.

 Inputs (already on disk from the earlier steps):
     amazon_graph_centralities.pkl   (true global centralities)
     amazon_local_cache_3.pkl        (local values + cut info + client map)
 Output: amazon_correction_quality.csv, amazon_topk_agreement.csv,
         amazon_topk_agreement.png
 Run:    python correct_and_topk_amazon.py
═══════════════════════════════════════════════════════════════════════════
"""

import os, pickle, time
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GLOBAL_CACHE = "amazon_graph_centralities.pkl"
LOCAL_CACHE  = "amazon_local_cache_3.pkl"
CENTRALITIES = ["degree_undir", "betweenness", "closeness", "pagerank"]
K_VALUES = [10, 25, 50, 100, 200]
FLOOR = 0.1
SEED = 42

TEXT, MUTED, GRID = "#1A2330", "#5B6B7A", "#E0E6EC"
C_LOCAL, C_CORR = "#90A4AE", "#2E7D32"


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def build_node_table(gc, blob):
    local, cuts = blob["local"], blob["cuts"]
    n2c, csize, N = blob["node_to_client"], blob["client_size"], blob["n_global"]
    rows = []
    for nd in range(N):
        deg_full, cut_edges, cut_ratio = cuts.get(nd, (0, 0, 0.0))
        n_c = csize.get(n2c.get(nd), 1)
        r = {"node": nd, "cut_ratio": cut_ratio, "cut_edges": cut_edges,
             "size_frac": n_c / N}
        for m in CENTRALITIES:
            r[f"local_{m}"] = local[m].get(nd, 0.0)
            r[f"global_{m}"] = gc[m].get(nd, 0.0)
        rows.append(r)
    return pd.DataFrame(rows)


def correct(tbl):
    """Fit on half the nodes, choose winner on the other half, apply to all.
    Same candidate set and selection logic as the IDS correction."""
    rng = np.random.default_rng(SEED)
    nodes = tbl["node"].to_numpy()
    perm = rng.permutation(len(nodes))
    half = len(nodes) // 2
    fitset = tbl.iloc[perm[:half]].reset_index(drop=True)
    selset = tbl.iloc[perm[half:]].reset_index(drop=True)

    L = lambda t, m: t[f"local_{m}"].to_numpy(float)
    CR = lambda t: t["cut_ratio"].to_numpy(float)
    KEEP = lambda t: np.maximum(1.0 - CR(t), FLOOR)

    corrected, chosen, quality = {}, {}, []
    for m in CENTRALITIES:
        g_fit = fitset[f"global_{m}"].to_numpy(float)
        l_fit, cr_fit = L(fitset, m), CR(fitset)
        # MAE-fitted factor a
        best_a, best_e = 0.0, np.inf
        for a in np.linspace(-5, 60, 651):
            e = mae(g_fit, l_fit * (1 + a * cr_fit))
            if e < best_e:
                best_a, best_e = a, e

        cands = [("no correction", lambda t, m=m: L(t, m)),
                 ("M2b divide", lambda t, m=m: L(t, m) / KEEP(t)),
                 (f"M3 fitted a={best_a:.3g}",
                  lambda t, m=m, a=best_a: L(t, m) * (1 + a * CR(t)))]
        if m == "degree_undir":
            cands.append(("M2 exact",
                          lambda t, m=m: L(t, m) + t["cut_edges"].to_numpy(float)))

        feats = lambda t, m=m: np.column_stack(
            [L(t, m), CR(t), t["size_frac"].to_numpy(float)])
        lin = LinearRegression().fit(feats(fitset), g_fit)
        cands.append(("LIN", lambda t, lin=lin, m=m: lin.predict(feats(t))))
        rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                                   random_state=SEED, n_jobs=-1).fit(feats(fitset), g_fit)
        cands.append(("RF", lambda t, rf=rf, m=m: rf.predict(feats(t))))

        g_sel = selset[f"global_{m}"].to_numpy(float)
        before = mae(g_sel, L(selset, m))
        best_name, best_fn, best_err = "no correction", cands[0][1], before
        for name, fn in cands:
            try:
                e = mae(g_sel, fn(selset))
            except Exception:
                continue
            if np.isfinite(e) and e < best_err:
                best_name, best_fn, best_err = name, fn, e

        corrected[m] = np.asarray(best_fn(tbl), float)
        chosen[m] = best_name
        imp = 100 * (before - best_err) / before if before > 0 else 0.0
        quality.append({"centrality": m, "chosen_method": best_name,
                        "error_before": before, "error_after": best_err,
                        "improvement_pct": round(imp, 1)})
    return corrected, chosen, pd.DataFrame(quality)


def topk_set(values, nodes, k):
    order = np.lexsort((nodes, -np.asarray(values, float)))
    return set(np.asarray(nodes)[order[:k]])


def main():
    if not (os.path.exists(GLOBAL_CACHE) and os.path.exists(LOCAL_CACHE)):
        print("Missing caches. Run build_amazon_global.py and "
              "build_amazon_local.py first.")
        return
    t0 = time.time()
    gc = pickle.load(open(GLOBAL_CACHE, "rb"))["cents"]
    blob = pickle.load(open(LOCAL_CACHE, "rb"))
    tbl = build_node_table(gc, blob)
    nodes = tbl["node"].to_numpy()
    print(f"  {len(nodes):,} products, 3 clients")

    print("\n  fitting corrections (best method per centrality) ...")
    corrected, chosen, quality = correct(tbl)

    print("\n" + "=" * 78)
    print("  CORRECTION QUALITY   (MAE vs true global, held-out nodes)")
    print("=" * 78)
    print(f"  {'centrality':<14}{'method':<22}{'before':>12}{'after':>12}{'improve':>10}")
    print(f"  {'-'*14}{'-'*22}{'-'*12}{'-'*12}{'-'*10}")
    for _, r in quality.iterrows():
        print(f"  {r['centrality']:<14}{r['chosen_method']:<22}"
              f"{r['error_before']:>12.5f}{r['error_after']:>12.5f}"
              f"{r['improvement_pct']:>9.1f}%")
    print("=" * 78)
    quality.to_csv("amazon_correction_quality.csv", index=False, float_format="%.6f")

    print("\n" + "=" * 78)
    print("  PART 2 — DOES CORRECTION RECOVER THE TRUE RANKING?")
    print("  Overlap@K with the true global top-K   (1.0 = identical)")
    print("=" * 78)
    rows = []
    for m in CENTRALITIES:
        print(f"\n  {m}")
        print(f"      {'K':>5}{'local vs true':>16}{'corrected vs true':>20}{'change':>10}")
        g = tbl[f"global_{m}"].to_numpy(float)
        l = tbl[f"local_{m}"].to_numpy(float)
        c = corrected[m]
        for k in K_VALUES:
            tg = topk_set(g, nodes, k)
            ov_l = len(topk_set(l, nodes, k) & tg) / k
            ov_c = len(topk_set(c, nodes, k) & tg) / k
            print(f"      {k:>5}{ov_l:>16.3f}{ov_c:>20.3f}{ov_c-ov_l:>+10.3f}")
            rows.append({"centrality": m, "K": k, "overlap_local": ov_l,
                         "overlap_corrected": ov_c, "gain": ov_c - ov_l})
    print("=" * 78)
    print("  positive change = correction recovers the true global ranking")
    agr = pd.DataFrame(rows)
    agr.to_csv("amazon_topk_agreement.csv", index=False, float_format="%.6f")

    # chart
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), facecolor="white")
    fig.suptitle("Amazon — does correction recover the true centrality ranking?\n"
                 "Overlap@K with the true global top-K (1.0 = identical)",
                 fontsize=13.5, fontweight="bold", color=TEXT, y=1.05)
    for ax, m in zip(axes, CENTRALITIES):
        sub = agr[agr["centrality"] == m]
        x = np.arange(len(sub)); w = 0.38
        ax.bar(x - w/2, sub["overlap_local"], w, color=C_LOCAL,
               edgecolor="white", linewidth=1.2, label="local")
        ax.bar(x + w/2, sub["overlap_corrected"], w, color=C_CORR,
               edgecolor="white", linewidth=1.2, label="corrected")
        ax.set_title(m, fontsize=12, fontweight="bold", color=TEXT, pad=8)
        ax.set_xticks(x); ax.set_xticklabels([f"K={v}" for v in sub["K"]],
                                             fontsize=8.5, color=TEXT, rotation=20)
        ax.set_ylim(0, 1.05); ax.set_ylabel("overlap with truth",
                                            color=MUTED, fontsize=10)
        ax.set_facecolor("#F7F9FB"); ax.tick_params(colors=MUTED, labelsize=9)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]: ax.spines[s].set_color(GRID)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig("amazon_topk_agreement.png", dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"\n  saved: amazon_correction_quality.csv, amazon_topk_agreement.csv,")
    print(f"         amazon_topk_agreement.png")
    print(f"  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
