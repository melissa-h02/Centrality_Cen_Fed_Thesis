"""
═══════════════════════════════════════════════════════════════════════════
 STEP B — Train the correction model, test it, and measure how much it helps
═══════════════════════════════════════════════════════════════════════════

 THE MODEL (one linear regression per centrality metric):

     predicted_global = w1*local_val + w2*cut_ratio + w3*client_size_frac + b

also trying RANDOM FOREST

   • It LEARNS the four numbers (w1, w2, w3, b) from the 3-client data,
     where it can see both the local values AND the true global values.
   • Then it is TESTED on the 5-client data — a split it never trained on —
     using only the three inputs, to predict the true global value.

 WHY train on 3 and test on 5?
   In real federation you never have the global values. Training on 3 clients
   is the "practice" where the answer is known; testing on 5 clients proves
   the correction still works on a NEW situation without seeing the answer.
   If the error drops on the 5-client test, the correction generalizes.

 HOW WE SCORE IT (on the 5-client TEST set):
   • error BEFORE = average | global - local |        (no correction)
   • error AFTER  = average | global - predicted |    (model correction)
   • improvement  = how much the error dropped, in %.
   (MAE = mean absolute error = average size of the gap.)

 Output: ml_correction_results.csv  +  ml_correction_chart.png  +  printed table
 Run:  python step_b_train_test.py   (after step_a has created the two csv files)
═══════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────
TRAIN_CSV = "features_3client.csv"     # train here
TEST_CSV  = "features_5client.csv"     # test here
METRICS   = ["degree", "betweenness", "closeness", "pagerank"]
LAB = {"degree": "Degree", "betweenness": "Betweenness",
       "closeness": "Closeness", "pagerank": "PageRank"}
INPUTS = ["cut_ratio", "client_size_frac"]   # + local_val_<m> added per metric
SEED   = 42

OUT_CSV = "ml_correction_results.csv"
OUT_PNG = "ml_correction_chart.png"


def mae(true, pred):
    return float(np.mean(np.abs(np.asarray(true) - np.asarray(pred))))


def main():
    for f in (TRAIN_CSV, TEST_CSV):
        if not os.path.exists(f):
            print(f"❌ Missing {f}. Run step_a_build_features.py first.")
            return

    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)

    print("=" * 100)
    print("  ML CORRECTION — TRAIN on 3 clients, TEST on 5 clients")
    print("  Best method per metric:  degree -> exact formula,  others -> random forest")
    print("  (error = average |global - value|;  lower AFTER is better)")
    print("=" * 100)
    print(f"  {'Metric':<12}{'before':>11}"
          f"{'Linear after':>14}{'Lin %':>8}"
          f"{'Forest after':>14}{'Forest %':>10}"
          f"   {'BEST':<8}{'best %':>7}")
    print(f"  {'-'*12}{'-'*11}{'-'*14}{'-'*8}{'-'*14}{'-'*10}   {'-'*8}{'-'*7}")

    rows = []
    for m in METRICS:
        lcol, gcol = f"local_val_{m}", f"global_val_{m}"
        feat_cols = [lcol] + INPUTS

        extra = ["cut_edges"] if (m == "degree" and "cut_edges" in test.columns) else []
        tr = train[feat_cols + [gcol] + extra].dropna()
        te = test[feat_cols + [gcol] + extra].dropna()
        if len(tr) < 5 or len(te) < 5:
            continue

        Xtr, ytr = tr[feat_cols].to_numpy(), tr[gcol].to_numpy()
        Xte, yte = te[feat_cols].to_numpy(), te[gcol].to_numpy()
        local_te = te[lcol].to_numpy()
        cut_te   = te["cut_ratio"].to_numpy()

        before = mae(yte, local_te)

        # ── Model 1: linear regression ──────────────────────────────────
        lin = LinearRegression().fit(Xtr, ytr)
        lin_after = mae(yte, lin.predict(Xte))
        lin_impr  = 100.0 * (before - lin_after) / before if before > 0 else 0.0
        lin_r2    = lin.score(Xte, yte)

        # ── Model 2: random forest (non-linear) ─────────────────────────
        rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                                   random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
        rf_after = mae(yte, rf.predict(Xte))
        rf_impr  = 100.0 * (before - rf_after) / before if before > 0 else 0.0
        rf_r2    = rf.score(Xte, yte)

# ── Model 3: EXACT formula (degree only) ────────────────────────
        # Step 2 proof: local = global - cut_edges -> global = local + cut_edges
        if m == "degree":
            if "cut_edges" in te.columns:
                formula_pred = local_te + te["cut_edges"].to_numpy()
            else:
                keep = np.maximum(1.0 - cut_te, 0.1)
                formula_pred = local_te / keep
            formula_after = mae(yte, formula_pred)
            formula_impr  = 100.0 * (before - formula_after) / before if before > 0 else 0.0
        else:
            formula_after = np.nan
            formula_impr  = np.nan
        

        # ── Choose the best method per metric ───────────────────────────
        # degree -> its exact formula; the others -> the better ML model.
        if m == "degree":
            best_method, best_after, best_impr = "Formula", formula_after, formula_impr
        else:
            if rf_after <= lin_after:
                best_method, best_after, best_impr = "Forest", rf_after, rf_impr
            else:
                best_method, best_after, best_impr = "Linear", lin_after, lin_impr

        print(f"  {LAB[m]:<12}{before:>11.4g}"
              f"{lin_after:>14.4g}{lin_impr:>7.0f}%"
              f"{rf_after:>14.4g}{rf_impr:>9.0f}%"
              f"   {best_method:<8}{best_impr:>7.0f}%")

        rows.append({
            "Centrality": LAB[m], "test_error_before": before,
            "linear_after": lin_after, "linear_impr_pct": round(lin_impr, 1),
            "linear_R2": round(lin_r2, 3),
            "forest_after": rf_after, "forest_impr_pct": round(rf_impr, 1),
            "forest_R2": round(rf_r2, 3),
            "formula_after": formula_after,
            "formula_impr_pct": (round(formula_impr, 1)
                                 if not np.isnan(formula_impr) else np.nan),
            "best_method": best_method,
            "best_impr_pct": round(best_impr, 1),
            "n_train": len(tr), "n_test": len(te),
        })

    print("=" * 100)
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    print(f"  saved: {OUT_CSV}")

    # ── chart: Linear vs Forest (+ Formula for degree), % error reduction ──
    if not res.empty:
        TEXT, MUTED, GRID, WORSE_C = "#1A2330", "#5B6B7A", "#E0E6EC", "#C8553D"
        LIN_C, RF_C, FORM_C = "#1C6E8C", "#2A9D8F", "#E9A23B"
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor="white")
        fig.suptitle("ML correction — % error reduction on the 5-client test   "
                     "[Linear vs Random forest (+ exact formula for degree)]",
                     fontsize=14, fontweight="bold", color=TEXT, y=0.99)
        for ax, (_, r) in zip(axes.flat, res.iterrows()):
            labels = ["Linear", "Random\nforest"]
            vals   = [r["linear_impr_pct"], r["forest_impr_pct"]]
            base_c = [LIN_C, RF_C]
            # add the exact formula bar only for degree
            if not np.isnan(r.get("formula_impr_pct", np.nan)):
                labels.append("Formula")
                vals.append(r["formula_impr_pct"])
                base_c.append(FORM_C)
            colors = [WORSE_C if v < 0 else c for v, c in zip(vals, base_c)]

            bars = ax.bar(labels, vals, width=0.6, color=colors,
                          edgecolor="white", linewidth=1.5, zorder=3)
            ax.set_title(f"{r['Centrality']}   \u2014   best: {r['best_method']}",
                         fontsize=13, fontweight="bold", color=TEXT, pad=10)
            ax.axhline(0, color="#333", linewidth=1, zorder=2)
            span = max(abs(min(vals)), abs(max(vals)), 10)
            ax.set_ylim(min(min(vals) * 1.35, -6), max(max(vals) * 1.22, 12))
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2,
                        v + (span * 0.03 if v >= 0 else -span * 0.03),
                        f"{v:+.0f}%", ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=12.5, color=TEXT, fontweight="bold")
            ax.set_facecolor("#F7F9FB")
            ax.tick_params(colors=MUTED, labelsize=11)
            ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
            for s in ["top", "right"]: ax.spines[s].set_visible(False)
            for s in ["left", "bottom"]: ax.spines[s].set_color(GRID)
            ax.set_ylabel("% error reduction", color=MUTED, fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  saved: {OUT_PNG}")

    print("\n  How to read it:")
    print("   - 'before' = raw local error; 'after' = error once the model corrects it.")
    print("   - %  = improvement on the 5-client split neither model trained on.")
    print("   - R2 = how well the inputs explain the true value (1 best, <0 = poor fit).")
    print("   - 'winner' = the model with the smaller test error for that metric.")


if __name__ == "__main__":
    main()