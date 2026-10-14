"""
═══════════════════════════════════════════════════════════════════════════
 EDGE CLASSIFICATION — STEP 1: build the per-flow (edge) feature table
═══════════════════════════════════════════════════════════════════════════

 Goal (centralized, before federation): classify each FLOW into its attack
 type (multi-class), and test whether adding CENTRALITY features helps.

 We classify EDGES, not nodes, because the same IP can send benign traffic
 and attack traffic — the label belongs to the flow. Centralities are still
 node-level, so we attach the SOURCE node's and DESTINATION node's centrality
 values to each flow (src_* and dst_*).

 One row per flow, with:

   TARGET
     Attack               -> Benign / dos / ddos / injection / scanning / ...

   RAW FLOW FEATURES  (Model A — "no centralities")
     L4_SRC_PORT, L4_DST_PORT, PROTOCOL, L7_PROTO,
     IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS,
     TCP_FLAGS, FLOW_DURATION_MILLISECONDS
     (only the columns present in your CSV are used)

   CENTRALITY FEATURES  (Model B — "with centralities")
     src_<c> and dst_<c> for c in
        degree_undir, in_degree, out_degree,
        betweenness, closeness, pagerank, eigenvector
     computed once on the FULL global graph (same definitions as your Step 1).

 LEAKAGE GUARDS
   • IP address strings are NEVER used as features (they'd just be memorised).
   • The binary `Label` column is NEVER a feature (it gives away Benign vs not).
   • Exact duplicate flow rows are dropped by default (see DROP_EXACT_DUPES),
     so the same row can't sit in both train and test.

 Output:  edge_features_global.csv
 Run:     python build_global_features.py    (needs NF-ToN-IoT.csv beside it)
═══════════════════════════════════════════════════════════════════════════
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
DATASET_PATH = "NF-ToN-IoT.csv"
SRC_COL, DST_COL      = "IPV4_SRC_ADDR", "IPV4_DST_ADDR"
LABEL_COL, ATTACK_COL = "Label", "Attack"

# pkl cache: the graph + centralities never change for the same dataset, so
# compute them once and save. Later runs load in ~1s instead of ~95s.
CACHE_PATH = "graph_centralities.pkl"
USE_CACHE  = True     # set False (or delete the .pkl) to force a full recompute

# raw flow attributes to keep as features IF present (missing ones skipped)
RAW_FLOW_COLS = ["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
                 "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
                 "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS"]

CENTRALITIES = ["degree_undir", "in_degree", "out_degree",
                "betweenness", "closeness", "pagerank", "eigenvector"]

# duplicate handling
DROP_EXACT_DUPES     = True    # drop rows identical across ALL columns (safe)
COLLAPSE_CONFLICTS   = False   # also collapse feature-identical rows that carry
                               # DIFFERENT labels down to ONE label (keeps first).
                               # Off by default so you SEE the conflict count.

OUT_CSV = "edge_features_global.csv"


def build_graph(df):
    df[SRC_COL] = df[SRC_COL].astype(str)
    df[DST_COL] = df[DST_COL].astype(str)
    G = nx.DiGraph()
    for ip in pd.concat([df[SRC_COL], df[DST_COL]]).unique():
        G.add_node(ip)
    agg = df.groupby([SRC_COL, DST_COL]).size().reset_index(name="w")
    for _, r in agg.iterrows():
        G.add_edge(r[SRC_COL], r[DST_COL], weight=r["w"])
    print(f"  graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def centralities(G):
    n = G.number_of_nodes()
    UG = G.to_undirected()
    out = {}
    out["degree_undir"] = dict(UG.degree())
    out["in_degree"]    = dict(G.in_degree())
    out["out_degree"]   = dict(G.out_degree())
    out["betweenness"]  = (nx.betweenness_centrality(G, normalized=False)
                           if n > 2 else {v: 0.0 for v in G})
    out["closeness"]    = nx.closeness_centrality(G, wf_improved=False)
    try:
        out["pagerank"] = nx.pagerank(G, alpha=0.85, max_iter=200)
    except Exception:
        out["pagerank"] = {v: 1.0 / n for v in G}
    try:
        out["eigenvector"] = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
    except Exception:
        try:
            out["eigenvector"] = nx.eigenvector_centrality(UG, max_iter=1000, tol=1e-06)
        except Exception:
            out["eigenvector"] = {v: 0.0 for v in G}
    return out


def report_duplicates(df, feat_cols):
    """Print how many exact-duplicate rows and feature-conflict rows exist."""
    n = len(df)
    exact = int(df.duplicated(keep="first").sum())
    # feature-identical rows carrying >1 distinct Attack label
    grp = df.groupby(feat_cols)[ATTACK_COL].nunique()
    conflict_keys = int((grp > 1).sum())
    conflict_rows = int(df.groupby(feat_cols)[ATTACK_COL]
                          .transform("nunique").gt(1).sum())
    print(f"  rows: {n:,}")
    print(f"  exact duplicate rows (all columns identical): {exact:,}")
    print(f"  flows whose features map to >1 attack label : {conflict_rows:,}"
          f"  ({conflict_keys:,} distinct feature-tuples)")
    return exact, conflict_rows


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Could not find {DATASET_PATH} in this folder.")
        return

    t0 = time.time()
    print(f"Loading {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df[SRC_COL] = df[SRC_COL].astype(str)
    df[DST_COL] = df[DST_COL].astype(str)

    raw_cols = [c for c in RAW_FLOW_COLS if c in df.columns]

    # ── graph + centralities (once, on the full graph) ────────────────
    # ── graph + centralities, with pkl caching ────────────────────────
    # signature = dataset name + row count, so the cache auto-invalidates if
    # you swap in a different / larger dataset.
    sig = (os.path.basename(DATASET_PATH), len(df))
    cents = None
    if USE_CACHE and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                blob = pickle.load(f)
            if blob.get("sig") == sig:
                cents = blob["cents"]
                print(f"  loaded graph + centralities from cache "
                      f"({CACHE_PATH}) — skipped recompute")
            else:
                print("  cache found but dataset changed -> recomputing")
        except Exception as e:
            print(f"  cache unreadable ({e}) -> recomputing")

    if cents is None:
        G = build_graph(df)
        print("Computing 7 global centralities on the full graph ...")
        cents = centralities(G)
        if USE_CACHE:
            with open(CACHE_PATH, "wb") as f:
                pickle.dump({"sig": sig, "cents": cents}, f)
            print(f"  saved graph + centralities to cache ({CACHE_PATH})")

    # ── attach src_* and dst_* centralities to every flow (vectorised) ─
    print("Attaching src/dst centralities to each flow ...")
    out = df[[SRC_COL, DST_COL] + raw_cols + [LABEL_COL, ATTACK_COL]].copy()
    for c in CENTRALITIES:
        cmap = cents[c]
        out[f"src_{c}"] = out[SRC_COL].map(cmap).astype(float)
        out[f"dst_{c}"] = out[DST_COL].map(cmap).astype(float)

    # ── duplicate diagnostics + handling ──────────────────────────────
    print("-" * 70)
    print("  DUPLICATE / MULTI-LABEL CHECK")
    exact, conflicts = report_duplicates(out, raw_cols)

    if DROP_EXACT_DUPES and exact > 0:
        out = out.drop_duplicates(keep="first").reset_index(drop=True)
        print(f"  -> dropped {exact:,} exact-duplicate rows "
              f"(prevents train/test leakage)")
    if COLLAPSE_CONFLICTS and conflicts > 0:
        before = len(out)
        out = out.drop_duplicates(subset=raw_cols, keep="first").reset_index(drop=True)
        print(f"  -> collapsed feature-conflicts: {before-len(out):,} rows removed "
              f"(kept first label per feature-tuple)")
    print("-" * 70)

    out.to_csv(OUT_CSV, index=False, float_format="%.6f")

    # ── class distribution ────────────────────────────────────────────
    print("  final rows:", f"{len(out):,}")
    print("  class distribution (Attack):")
    vc = out[ATTACK_COL].value_counts()
    for cls, cnt in vc.items():
        print(f"      {str(cls):<14} {cnt:>8,}  ({100*cnt/len(out):5.1f}%)")
    print(f"  feature columns: {len(raw_cols)} raw  +  {2*len(CENTRALITIES)} centrality")
    print(f"  saved: {OUT_CSV}")
    print(f"  done in {time.time()-t0:.1f}s.  Next: python classify_edges_compare.py")


if __name__ == "__main__":
    main()