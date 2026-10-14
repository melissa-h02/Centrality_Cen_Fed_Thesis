"""
═══════════════════════════════════════════════════════════════════════════
 STEP 3 — CORRECTED CENTRALITIES:  best correction per metric -> features
═══════════════════════════════════════════════════════════════════════════

 Takes the LOCAL (distorted) centralities a client computed, corrects them
 back toward the TRUE global value, and writes the corrected values as the
 graph features for classification.

 ── THE CANDIDATE CORRECTIONS (your correction study) ─────────────────────
   M1  size + cut     local * (N-1) / ((n_client-1) * (1-cut_ratio))
   M2  exact          degree only:  global = local + cut_edges     [proven]
   M2b divide         local / (1 - cut_ratio)
   M3  fitted factor  local * (1 + a * cut_ratio)
   LIN linear reg.    w1*local + w2*cut_ratio + w3*size_frac + b
   RF  random forest  non-linear model on the same three inputs
   --  no correction  (safety floor: keep the local value)

 Each metric gets whichever of these performs best FOR THAT METRIC — the
 modular idea: degree, betweenness, closeness and PageRank distort for
 different reasons, so they do not share one formula.

 ── HOW WE AVOID FLATTERING THE RESULT ────────────────────────────────────
 Fitting needs the true global values, which a real client never has. Worse,
 the SAME nodes appear in both partitions with the same target, so fitting
 and testing on the same nodes would let a model memorise node identities.
 So we separate everything three ways:

     fit parameters   ->  5-client partition, node half A
     choose winner    ->  5-client partition, node half B   (unseen nodes)
     report quality   ->  3-client partition, node half B   (unseen nodes
                                                             AND partition)
     apply winner     ->  all nodes of the 3-client partition (to build
                          the feature file we then classify)

 ── VISIBILITY RULE (identical to Step 2, so the comparison is fair) ──────
 A destination in another client is invisible, so its features stay 0 — a
 client cannot correct a value it cannot see.

 Output:  edge_features_corrected_3client.csv   -> feed to run_all_models.py
          correction_quality.csv / .png         -> before vs after, per metric
          correction_candidates.csv             -> every method's error
 Run:     python build_corrected_features.py
═══════════════════════════════════════════════════════════════════════════
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
DATASET_PATH = "NF-ToN-IoT.csv"
SRC_COL, DST_COL      = "IPV4_SRC_ADDR", "IPV4_DST_ADDR"
LABEL_COL, ATTACK_COL = "Label", "Attack"

APPLY_TO = 3          # partition we correct, then classify
FIT_ON   = 5          # partition used ONLY to fit / choose the corrections

CACHE_APPLY  = f"local_graph_cache_{APPLY_TO}.pkl"
CACHE_FIT    = f"local_graph_cache_{FIT_ON}.pkl"
GLOBAL_CACHE = "global_centralities.pkl"

RAW_FLOW_COLS = ["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
                 "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
                 "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS"]

CENTRALITIES = ["degree_undir", "betweenness", "closeness", "pagerank"]

FLOOR = 0.1           # never divide by less than this (guards cut_ratio -> 1)
SEED  = 42
DROP_EXACT_DUPES = True

OUT_CSV        = f"edge_features_corrected_{APPLY_TO}client.csv"
OUT_QUALITY    = "correction_quality.csv"
OUT_CANDIDATES = "correction_candidates.csv"
OUT_PNG        = "correction_quality.png"
GREY, GREEN, RED = "#90A4AE", "#2E7D32", "#C8553D"
TEXT, MUTED, GRID = "#1A2330", "#5B6B7A", "#E0E6EC"


# ── graph + centralities ────────────────────────────────────────────────
def build_graph(df):
    G = nx.DiGraph()
    for ip in pd.concat([df[SRC_COL], df[DST_COL]]).unique():
        G.add_node(ip)
    agg = df.groupby([SRC_COL, DST_COL]).size().reset_index(name="w")
    for _, r in agg.iterrows():
        G.add_edge(r[SRC_COL], r[DST_COL], weight=r["w"])
    return G


def centralities(G):
    n = G.number_of_nodes()
    UG = G.to_undirected()
    out = {}
    out["degree_undir"] = dict(UG.degree())
    out["betweenness"]  = (nx.betweenness_centrality(G, normalized=False)
                           if n > 2 else {v: 0.0 for v in G})
    out["closeness"]    = nx.closeness_centrality(G, wf_improved=False)
    try:
        out["pagerank"] = nx.pagerank(G, alpha=0.85, max_iter=200)
    except Exception:
        out["pagerank"] = {v: 1.0 / n for v in G}
    return out


def load_cache(path):
    if not os.path.exists(path):
        print(f"  Missing {path}. Run build_local_features.py with the "
              f"matching N_CLIENTS first.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def node_table(blob, global_cents):
    """One row per node: local value, true global value, and the three inputs
    a real client actually has (local, cut_ratio, client_size_frac)."""
    local, cuts = blob["local"], blob["cuts"]
    client_size, node_to_client = blob["client_size"], blob["node_to_client"]
    N = blob["n_global"]
    rows = []
    for nd, lv in local.items():
        deg, cut_edges, cut_ratio = cuts.get(nd, (0, 0, 0.0))
        n_c = client_size.get(node_to_client.get(nd), 1)
        r = {"node": nd, "cut_ratio": cut_ratio, "cut_edges": cut_edges,
             "n_client": n_c, "n_global": N, "size_frac": n_c / N}
        for m in CENTRALITIES:
            r[f"local_{m}"]  = lv.get(m, 0.0)
            r[f"global_{m}"] = global_cents[m].get(nd, 0.0)
        rows.append(r)
    return pd.DataFrame(rows)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


# ── helpers used by the candidate formulas ──────────────────────────────
def L(t, m):  return t[f"local_{m}"].to_numpy(float)
def CR(t):    return t["cut_ratio"].to_numpy(float)
def CE(t):    return t["cut_edges"].to_numpy(float)
def KEEP(t):  return np.maximum(1.0 - CR(t), FLOOR)      # guarded (1-cut)


def fit_a_mae(tune, m):
    """Fit `a` in  local*(1 + a*cut_ratio)  by minimising MEAN ABSOLUTE error.

    Least squares was tried first and failed badly on betweenness: a handful
    of hub nodes with enormous values dominate the squared error, so the fit
    chased those outliers (a came out around 58) and the result did not
    transfer to another partition. We score with MAE, so we fit with MAE.
    """
    l, cr = L(tune, m), CR(tune)
    g = tune[f"global_{m}"].to_numpy(float)
    grid = np.linspace(-5.0, 20.0, 251)
    best_a, best_e = 0.0, np.inf
    for a in grid:
        e = mae(g, l * (1.0 + a * cr))
        if e < best_e:
            best_a, best_e = a, e
    step = grid[1] - grid[0]
    for _ in range(4):                       # refine around the winner
        for a in np.linspace(best_a - step, best_a + step, 21):
            e = mae(g, l * (1.0 + a * cr))
            if e < best_e:
                best_a, best_e = a, e
        step /= 10.0
    return float(best_a)


def candidates_for(m, tune):
    """Build every candidate correction for metric m, fitted on `tune`."""
    cands = []

    # safety floor — keep the local value untouched
    cands.append(("-- no correction", lambda t: L(t, m)))

    # M2 exact — proven identity, degree only
    if m == "degree_undir":
        cands.append(("M2 exact: local + cut_edges", lambda t: L(t, m) + CE(t)))

    # M2b divide form
    cands.append(("M2b divide: local/(1-cut)", lambda t: L(t, m) / KEEP(t)))

    # M1 size + cut
    def m1(t):
        size_term = ((t["n_global"].to_numpy(float) - 1.0) /
                     np.maximum(t["n_client"].to_numpy(float) - 1.0, 1.0))
        return L(t, m) * size_term / KEEP(t)
    cands.append(("M1 size+cut", m1))

    # M3 fitted factor a (MAE-fitted)
    a = fit_a_mae(tune, m)
    cands.append((f"M3 fitted a={a:.3g}",
                  lambda t, a=a: L(t, m) * (1.0 + a * CR(t))))

    # LIN / RF — the three inputs a client really has
    feats = lambda t: np.column_stack([L(t, m), CR(t),
                                       t["size_frac"].to_numpy(float)])
    ytune = tune[f"global_{m}"].to_numpy(float)

    lin = LinearRegression().fit(feats(tune), ytune)
    cands.append(("LIN linear regression",
                  lambda t, lin=lin: lin.predict(feats(t))))

    rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                               random_state=SEED, n_jobs=2).fit(feats(tune), ytune)
    cands.append(("RF random forest",
                  lambda t, rf=rf: rf.predict(feats(t))))

    return cands


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Could not find {DATASET_PATH} in this folder.")
        return
    t0 = time.time()

    blob_apply = load_cache(CACHE_APPLY)
    blob_fit   = load_cache(CACHE_FIT)
    if blob_apply is None or blob_fit is None:
        return

    print(f"Loading {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df[SRC_COL] = df[SRC_COL].astype(str)
    df[DST_COL] = df[DST_COL].astype(str)
    raw_cols = [c for c in RAW_FLOW_COLS if c in df.columns]

    sig = (os.path.basename(DATASET_PATH), len(df), tuple(CENTRALITIES))
    gc = None
    if os.path.exists(GLOBAL_CACHE):
        try:
            cand = pickle.load(open(GLOBAL_CACHE, "rb"))
            if cand.get("sig") == sig:
                gc = cand["cents"]
                print(f"  loaded global centralities from {GLOBAL_CACHE}")
        except Exception:
            pass
    if gc is None:
        print("  computing TRUE global centralities on the full graph ...")
        gc = centralities(build_graph(df))
        pickle.dump({"sig": sig, "cents": gc}, open(GLOBAL_CACHE, "wb"))
        print(f"  saved {GLOBAL_CACHE}")

    tbl_fit   = node_table(blob_fit, gc)      # 5-client view
    tbl_apply = node_table(blob_apply, gc)    # 3-client view

    # ── split NODES into two halves, so no node is both fitted and judged ──
    rng = np.random.default_rng(SEED)
    all_nodes = tbl_fit["node"].tolist()
    perm = rng.permutation(len(all_nodes))
    half = len(all_nodes) // 2
    nodes_A = set(np.array(all_nodes)[perm[:half]])      # fit here
    nodes_B = set(np.array(all_nodes)[perm[half:]])      # judge here

    tune   = tbl_fit[tbl_fit["node"].isin(nodes_A)].reset_index(drop=True)
    select = tbl_fit[tbl_fit["node"].isin(nodes_B)].reset_index(drop=True)
    report = tbl_apply[tbl_apply["node"].isin(nodes_B)].reset_index(drop=True)

    print(f"\n  fit parameters : {FIT_ON}-client partition, "
          f"{len(tune):,} nodes (half A)")
    print(f"  choose winner  : {FIT_ON}-client partition, "
          f"{len(select):,} nodes (half B, unseen)")
    print(f"  report quality : {APPLY_TO}-client partition, "
          f"{len(report):,} nodes (unseen nodes AND unseen partition)")
    print(f"  apply winner   : {APPLY_TO}-client partition, "
          f"all {len(tbl_apply):,} nodes")

    # ── compare every candidate, per metric ───────────────────────────
    print("\n" + "=" * 92)
    print("  CANDIDATE CORRECTIONS   (error on unseen nodes; lower = better)")
    print("=" * 92)
    chosen, all_rows = {}, []
    for m in CENTRALITIES:
        g_sel = select[f"global_{m}"].to_numpy(float)
        base  = mae(g_sel, L(select, m))
        print(f"\n  {m}    (local error = {base:.6g})")
        best_name, best_fn, best_err = None, None, np.inf
        for name, fn in candidates_for(m, tune):
            try:
                err = mae(g_sel, fn(select))
            except Exception as e:
                print(f"      {name:<32} failed ({e})")
                continue
            if not np.isfinite(err):
                continue
            tag = ""
            if err < best_err:
                best_name, best_fn, best_err = name, fn, err
                tag = "  <-- best"
            impr = 100.0 * (base - err) / base if base > 0 else 0.0
            print(f"      {name:<32}{err:>14.6g}{impr:>9.1f}%{tag}")
            all_rows.append({"centrality": m, "method": name,
                             "selection_error": err,
                             "improvement_pct": round(impr, 1)})
        chosen[m] = (best_name, best_fn)
        print(f"      CHOSEN: {best_name}")
    print("=" * 92)
    pd.DataFrame(all_rows).to_csv(OUT_CANDIDATES, index=False,
                                  float_format="%.8f")

    # ── quality of the chosen corrections, on fully unseen data ───────
    print("\n" + "=" * 92)
    print(f"  CORRECTION QUALITY   ({APPLY_TO}-client partition, "
          f"unseen nodes)   MAE = average |true - value|")
    print("=" * 92)
    print(f"  {'centrality':<15}{'chosen method':<30}"
          f"{'before':>13}{'after':>13}{'improvement':>13}")
    print(f"  {'-'*15}{'-'*30}{'-'*13}{'-'*13}{'-'*13}")
    quality, corrected_cols = [], {}
    for m in CENTRALITIES:
        name, fn = chosen[m]
        g = report[f"global_{m}"].to_numpy(float)
        before = mae(g, L(report, m))
        after  = mae(g, fn(report))
        impr = 100.0 * (before - after) / before if before > 0 else 0.0
        print(f"  {m:<15}{name:<30}{before:>13.6g}{after:>13.6g}{impr:>12.1f}%")
        quality.append({"centrality": m, "chosen_method": name,
                        "error_before": before, "error_after": after,
                        "improvement_pct": round(impr, 1)})
        # apply to EVERY node so the feature file is complete
        corrected_cols[m] = np.asarray(fn(tbl_apply), dtype=float)
    print("=" * 92)
    print("  positive improvement = the corrected value is closer to the truth")
    pd.DataFrame(quality).to_csv(OUT_QUALITY, index=False, float_format="%.8f")
    print(f"  saved: {OUT_QUALITY}, {OUT_CANDIDATES}")

    # ── attach corrected values to every flow ─────────────────────────
    print("\n  attaching CORRECTED centralities to each flow ...")
    nodes = tbl_apply["node"].tolist()
    cmaps = {m: dict(zip(nodes, corrected_cols[m])) for m in CENTRALITIES}

    node_to_client = blob_apply["node_to_client"]
    cuts = blob_apply["cuts"]
    client_size = blob_apply["client_size"]
    N = blob_apply["n_global"]

    out = df[[SRC_COL, DST_COL] + raw_cols + [LABEL_COL, ATTACK_COL]].copy()
    out["src_client"] = out[SRC_COL].map(node_to_client)
    out["dst_client"] = out[DST_COL].map(node_to_client)
    out["is_cross_client"] = (out["src_client"] != out["dst_client"]).astype(int)

    for m in CENTRALITIES:
        cm = cmaps[m]
        out[f"src_{m}"] = out[SRC_COL].map(cm).fillna(0.0).astype(float)
        dst_vals = out[DST_COL].map(cm).fillna(0.0).astype(float)
        out[f"dst_{m}"] = np.where(out["is_cross_client"] == 1, 0.0, dst_vals)

    out["src_cut_ratio"] = out[SRC_COL].map(lambda v: cuts.get(v, (0, 0, 0))[2])
    out["dst_cut_ratio"] = out[DST_COL].map(lambda v: cuts.get(v, (0, 0, 0))[2])
    out["client_size"]      = out["src_client"].map(client_size)
    out["client_size_frac"] = out["client_size"] / N

    if DROP_EXACT_DUPES:
        n_before = len(out)
        out = out.drop_duplicates(keep="first").reset_index(drop=True)
        print(f"  dropped {n_before - len(out):,} exact-duplicate rows")

    out.to_csv(OUT_CSV, index=False, float_format="%.6f")

    try:
        check = pd.read_csv(OUT_CSV, usecols=[ATTACK_COL])
        if len(check) != len(out):
            print(f"\n  ** ERROR: {OUT_CSV} is INCOMPLETE "
                  f"({len(check):,} of {len(out):,}). Delete it and rerun.")
        else:
            print(f"\n  verified: {OUT_CSV} holds {len(check):,} rows, "
                  f"{check[ATTACK_COL].nunique()} classes, "
                  f"{os.path.getsize(OUT_CSV)/1e6:.0f} MB")
    except Exception as e:
        print(f"\n  ** ERROR verifying {OUT_CSV}: {e}")

    # ── chart ─────────────────────────────────────────────────────────
    qdf = pd.DataFrame(quality)
    n = len(qdf)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.6), facecolor="white")
    fig.suptitle("Correction quality — distance to the true global value\n"
                 f"(fitted on the {FIT_ON}-client partition, measured on "
                 f"unseen nodes of the {APPLY_TO}-client partition)",
                 fontsize=13.5, fontweight="bold", color=TEXT, y=1.03)
    for ax, (_, r) in zip(np.atleast_1d(axes), qdf.iterrows()):
        vals = [r["error_before"], r["error_after"]]
        col = GREEN if r["improvement_pct"] >= 0 else RED
        bars = ax.bar(["local", "corrected"], vals, width=0.55,
                      color=[GREY, col], edgecolor="white",
                      linewidth=1.5, zorder=3)
        ax.set_title(f"{r['centrality']}\n{r['improvement_pct']:+.0f}% error\n"
                     f"{r['chosen_method']}",
                     fontsize=10.5, fontweight="bold", color=TEXT, pad=8)
        top = max(vals) if max(vals) > 0 else 1
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.03, f"{v:.3g}",
                    ha="center", va="bottom", fontsize=10, color=TEXT,
                    fontweight="bold")
        ax.set_ylim(0, top * 1.32)
        ax.set_facecolor("#F7F9FB")
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]: ax.spines[s].set_color(GRID)
        ax.set_ylabel("avg error (MAE)", color=MUTED, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved: {OUT_PNG}")

    print(f"\n  done in {time.time()-t0:.1f}s")
    print(f"  Next: in run_all_models.py set")
    print(f"        FEATURES_CSV = \"{OUT_CSV}\"")
    print(f"        TAG          = \"step3_corrected\"")
    print(f"        OUTPUT_ROOT  = \"results_step3_corrected\"")
    print(f"        CENTRALITIES = {CENTRALITIES}")


if __name__ == "__main__":
    main()
