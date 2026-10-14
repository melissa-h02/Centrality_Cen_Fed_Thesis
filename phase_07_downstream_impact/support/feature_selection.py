"""
═══════════════════════════════════════════════════════════════════════════
 FEATURE SELECTION — rank the features, choose the traffic set to keep
═══════════════════════════════════════════════════════════════════════════

 Runs on edge_features_global.csv (already built), so it's fast — no graph
 rebuild. It ranks every feature by TWO standard methods and combines them:

   1) Mutual Information  (filter, model-free) — how much each feature tells
      us about the attack class, including non-linear relationships.
   2) Random Forest importance (embedded) — how much the forest actually
      relies on each feature when splitting.

 Each score is scaled to 0..1 and averaged into a combined rank. Features are
 tagged 'traffic' (raw NetFlow) or 'centrality' so you can pick a principled
 SELECTED TRAFFIC set for Model A / LSTM / CNN, and see which centralities
 matter for Model B.

 For speed + low memory on ~1.2M rows, ranking is done on a stratified
 SAMPLE (every class kept in proportion) — plenty accurate for ranking.

 Output:  feature_selection_ranking.csv  +  feature_selection.png
          + a ready-to-paste SELECTED_TRAFFIC list printed at the end
 Run:     python feature_selection.py     (needs edge_features_global.csv)
═══════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────
FEATURES_CSV = "edge_features_global.csv"
TARGET  = "Attack"
ID_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label"]   # never features

SAMPLE_N   = 150_000      # stratified sample used for ranking (speed/memory)
SEED       = 42
KEEP_FRAC  = 0.15         # keep traffic features scoring >= 15% of the top one
MIN_KEEP   = 5            # ...but always keep at least this many
N_JOBS     = 2            # limit parallelism to stay light on RAM

CENTRALITIES = ["degree_undir", "in_degree", "out_degree",
                "betweenness", "closeness", "pagerank", "eigenvector"]

OUT_CSV = "feature_selection_ranking.csv"
OUT_PNG = "feature_selection.png"
TEXT, MUTED, GRID = "#1A2330", "#5B6B7A", "#E0E6EC"
TRAF_C, CENT_C = "#90A4AE", "#2E7D32"


def stratified_sample(d, n, seed):
    if len(d) <= n:
        return d
    frac = n / len(d)
    parts = []
    for _, g in d.groupby(TARGET):
        parts.append(g.sample(max(1, int(round(len(g) * frac))), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def main():
    if not os.path.exists(FEATURES_CSV):
        print(f"Could not find {FEATURES_CSV}. Run the feature builder first.")
        return

    print(f"Loading {FEATURES_CSV} ...")
    d = pd.read_csv(FEATURES_CSV)

    # identify columns: selection runs on TRAFFIC (dataset) features ONLY.
    # centralities are NOT candidates for selection — they are the thing we
    # later test the value of, added on top in Model B.
    cent_cols = ([f"src_{c}" for c in CENTRALITIES if f"src_{c}" in d.columns] +
                 [f"dst_{c}" for c in CENTRALITIES if f"dst_{c}" in d.columns])
    non_feats = set(ID_COLS) | {TARGET} | set(cent_cols)
    traffic_cols = [c for c in d.columns if c not in non_feats]
    all_feats = traffic_cols                       # rank traffic features only
    ftype = {c: "traffic" for c in all_feats}

    # stratified sample for ranking
    s = stratified_sample(d[all_feats + [TARGET]].dropna(), SAMPLE_N, SEED)
    X = s[all_feats].to_numpy(float)
    y = LabelEncoder().fit_transform(s[TARGET].astype(str))
    print(f"  ranking on {len(s):,} sampled rows  |  "
          f"{len(traffic_cols)} traffic features (centralities excluded)")

    # drop zero-variance (constant) features — they carry no information
    var = X.var(axis=0)
    dead = [f for f, v in zip(all_feats, var) if v == 0]
    if dead:
        print(f"  NOTE: constant (zero-variance) features found: {dead}")

    # 1) mutual information
    print("  computing mutual information ...")
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=SEED)

    # 2) random forest importance
    print("  computing random forest importance ...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15,
                                min_samples_leaf=5, random_state=SEED,
                                n_jobs=N_JOBS).fit(X, y)
    rfi = rf.feature_importances_

    # combine (scale each to 0..1, then average)
    def norm(a):
        a = np.asarray(a, float)
        return a / a.max() if a.max() > 0 else a
    mi_n, rf_n = norm(mi), norm(rfi)
    combined = (mi_n + rf_n) / 2.0

    res = (pd.DataFrame({"feature": all_feats,
                         "type": [ftype[f] for f in all_feats],
                         "mutual_info": mi, "rf_importance": rfi,
                         "combined_score": combined})
             .sort_values("combined_score", ascending=False)
             .reset_index(drop=True))
    res.to_csv(OUT_CSV, index=False, float_format="%.6f")

    # printed ranking
    print("\n" + "=" * 62)
    print("  TRAFFIC FEATURE RANKING  (scaled MI + RF importance)")
    print("=" * 62)
    print(f"  {'#':>2}  {'feature':<28}{'MI':>8}{'RF':>8}{'combined':>10}")
    print(f"  {'-'*2}  {'-'*28}{'-'*8}{'-'*8}{'-'*10}")
    for i, r in res.iterrows():
        print(f"  {i+1:>2}  {r['feature']:<28}"
              f"{r['mutual_info']:>8.3f}{r['rf_importance']:>8.3f}"
              f"{r['combined_score']:>10.3f}")
    print("=" * 62)
    print(f"  saved: {OUT_CSV}")

    # ── recommended SELECTED TRAFFIC set ──────────────────────────────
    top = res["combined_score"].iloc[0]
    keep = res[res["combined_score"] >= KEEP_FRAC * top]["feature"].tolist()
    if len(keep) < MIN_KEEP:
        keep = res["feature"].head(MIN_KEEP).tolist()
    dropped = [f for f in traffic_cols if f not in keep]

    print("\n  RECOMMENDED  ── selected traffic features "
          f"(score >= {KEEP_FRAC:.0%} of the top feature):")
    print(f"  keep ({len(keep)}): {keep}")
    print(f"  drop ({len(dropped)}): {dropped}")
    print("\n  Paste this into your classifier:")
    print(f"  SELECTED_TRAFFIC = {keep}")

    # ── chart: traffic feature ranking ────────────────────────────────
    show = res.head(min(20, len(res)))[::-1]
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    ax.barh(show["feature"], show["combined_score"], color=TRAF_C,
            edgecolor="white", linewidth=1.0, zorder=3)
    ax.set_title("Traffic feature importance (scaled MI + RF)",
                 fontsize=14, fontweight="bold", color=TEXT, pad=12)
    ax.set_xlabel("combined score (0..1)", color=MUTED, fontsize=11)
    ax.set_facecolor("#F7F9FB"); ax.tick_params(colors=MUTED, labelsize=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]: ax.spines[sp].set_color(GRID)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved: {OUT_PNG}")


if __name__ == "__main__":
    main()