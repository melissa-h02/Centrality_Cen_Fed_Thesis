"""
═══════════════════════════════════════════════════════════════════════════
 STEP A — Build the training and testing data
═══════════════════════════════════════════════════════════════════════════

 The machine-learning correction needs two datasets:
   • a 3-client partition  -> used to TRAIN the model
   • a 5-client partition  -> used to TEST the model (a setup it never saw)

 Your existing node_distortion.csv only has the 3-client split, so this
 script rebuilds everything from the raw dataset and produces BOTH splits,
 each with the exact inputs the model needs.

 For every node (with 2+ connections) it writes these columns, per metric:

   local_val_<metric>   -> what the client computed (distorted)      [INPUT 1]
   cut_ratio            -> fraction of the node's edges that were cut [INPUT 2]
   client_size_frac     -> client size / whole-graph size            [INPUT 3]
   global_val_<metric>  -> the TRUE value on the full graph          [TARGET]

 Output files (written next to this script):
   features_3client.csv   (train)
   features_5client.csv   (test)

 Run:  python step_a_build_features.py
 (needs the raw NF-ToN-IoT.csv in the same folder)
═══════════════════════════════════════════════════════════════════════════
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
DATASET_PATH = "NF-ToN-IoT.csv"
SRC_COL, DST_COL, LABEL_COL = "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label"
ATTACK_COL, BYTES_COL       = "Attack", "IN_BYTES"

LOUVAIN_RES = 0.8
SEED        = 42
MIN_DEGREE  = 2
METRICS     = ["degree", "betweenness", "closeness", "pagerank"]

# The two partitions we need: 3 clients to train, 5 clients to test.
PARTITIONS  = {3: "features_3client.csv", 5: "features_5client.csv"}


# ── Graph building (same as your Step 1) ────────────────────────────────
def build_graph(path):
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    df[SRC_COL] = df[SRC_COL].astype(str)
    df[DST_COL] = df[DST_COL].astype(str)

    G = nx.DiGraph()
    attack_ips = (set(df[df[LABEL_COL] == 1][SRC_COL]) |
                  set(df[df[LABEL_COL] == 1][DST_COL]))
    for ip in pd.concat([df[SRC_COL], df[DST_COL]]).unique():
        G.add_node(ip, role="attack" if ip in attack_ips else "benign")

    agg = (df.groupby([SRC_COL, DST_COL, LABEL_COL])
             .agg(flow_count=(LABEL_COL, "count"),
                  total_bytes=(BYTES_COL, "sum"))
             .reset_index())
    for _, r in agg.iterrows():
        G.add_edge(r[SRC_COL], r[DST_COL], weight=r["flow_count"])
    print(f"  graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# ── Centralities (same definitions as your Step 1) ──────────────────────
def centralities(G):
    n = G.number_of_nodes()
    if n == 0:
        return {m: {} for m in METRICS}
    if n == 1:
        node = list(G.nodes())[0]
        return {m: {node: 0.0} for m in METRICS}
    out = {}
    # Degree as UNDIRECTED count (in + out collapsed), so it is consistent
    # with cut_ratio/cut_edges (also undirected) and with the Step 2 proof
    # local = global - cut_edges. This makes the degree correction exact.
    out["degree"] = dict(G.to_undirected().degree())       # undirected count
    out["betweenness"] = (nx.betweenness_centrality(G, normalized=False)
                          if n > 2 else {v: 0.0 for v in G})
    out["closeness"] = nx.closeness_centrality(G, wf_improved=False)
    try:
        out["pagerank"] = nx.pagerank(G, alpha=0.85, max_iter=200)
    except Exception:
        out["pagerank"] = {v: 1.0 / n for v in G}
    return out


# ── Partition into K clients from the REAL Louvain communities ──────────
def make_clients(G, k):
    """Split into `k` clients using the real Louvain communities.

    Louvain res=0.8 finds 5 communities on this graph. We pack them into k
    clients with the same greedy rule as Step 1 (add each community to the
    currently-smallest client):
      • k = 5 -> each community becomes its own client (a natural 5-way split)
      • k = 3 -> the 5 communities are merged down into 3 clients

    If a graph ever has FEWER communities than k, we split the largest
    community to reach k (fallback only; not needed for this dataset).
    """
    communities = [set(c) for c in nx.community.louvain_communities(
        G.to_undirected(), seed=SEED, resolution=LOUVAIN_RES)]

    # Fallback: only if there are somehow fewer communities than clients.
    rng = np.random.default_rng(SEED)
    while len(communities) < k:
        communities.sort(key=len, reverse=True)
        biggest = list(communities.pop(0))
        rng.shuffle(biggest)
        half = len(biggest) // 2
        communities.append(set(biggest[:half]))
        communities.append(set(biggest[half:]))

    # Greedy packing into k clients (identical rule to Step 1).
    parts = [set() for _ in range(k)]
    sizes = [0] * k
    for comm in sorted(communities, key=len, reverse=True):
        j = sizes.index(min(sizes))
        parts[j].update(comm)
        sizes[j] += len(comm)

    node_to_client = {nd: i for i, part in enumerate(parts) for nd in part}
    return parts, node_to_client


def cut_ratio(G, node_to_client):
    UG = G.to_undirected()
    out = {}
    for v in UG.nodes():
        nbrs = list(UG.neighbors(v))
        deg = len(nbrs)
        if deg == 0:
            out[v] = (0, 0, np.nan)          # (degree, cut_edges, cut_ratio)
            continue
        cut = sum(1 for nb in nbrs
                  if node_to_client.get(nb) != node_to_client.get(v))
        out[v] = (deg, cut, cut / deg)       # keep the exact whole-number count
    return out


def local_centralities(G, parts):
    per_client = {}
    for cid, part in enumerate(parts):
        C = G.subgraph(part).copy()
        per_client[cid] = centralities(C)
    return per_client


# ── Build one feature table for a given number of clients ───────────────
def build_features(G, global_cents, k):
    n_global = G.number_of_nodes()
    parts, node_to_client = make_clients(G, k)
    client_size = {cid: len(part) for cid, part in enumerate(parts)}
    local_per_client = local_centralities(G, parts)
    cut = cut_ratio(G, node_to_client)

    rows = []
    for v in G.nodes():
        deg_undir, cut_edges_v, cr = cut[v]
        if deg_undir < MIN_DEGREE:      # same filter as Step 1
            continue
        cid = node_to_client[v]
        row = {
            "node": v,
            "role": G.nodes[v].get("role"),
            "client": cid,
            "client_size": client_size[cid],
            "client_size_frac": client_size[cid] / n_global,   # INPUT 3
            "degree_undir": deg_undir,
            "cut_ratio": cr,                                    # INPUT 2
            "cut_edges": cut_edges_v,                           # exact whole-number count
        }
        for m in METRICS:
            row[f"local_val_{m}"]  = local_per_client[cid][m].get(v, np.nan)  # INPUT 1
            row[f"global_val_{m}"] = global_cents[m].get(v, np.nan)          # TARGET
        rows.append(row)
    return pd.DataFrame(rows), client_size


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Could not find {DATASET_PATH} in this folder.")
        return

    t0 = time.time()
    G = build_graph(DATASET_PATH)

    print("Computing global centralities (once, on the full graph)...")
    global_cents = centralities(G)

    for k, fname in PARTITIONS.items():
        print(f"\nBuilding {k}-client partition -> {fname}")
        feats, full_sizes = build_features(G, global_cents, k)
        print(f"  {len([s for s in full_sizes.values() if s > 0])} non-empty clients")
        print(f"  client sizes (all nodes): {dict(sorted(full_sizes.items()))}")
        print(f"  feature rows (degree>=2): {len(feats)}")
        feats.to_csv(fname, index=False, float_format="%.6f")
        print(f"  saved: {fname}")

    print(f"\nDone in {time.time()-t0:.1f}s. Next: run step_b_train_test.py")


if __name__ == "__main__":
    main()