"""
═══════════════════════════════════════════════════════════════════════════
 STEP 2 — FEDERATED FEATURES:  Louvain partition + LOCAL centralities
═══════════════════════════════════════════════════════════════════════════

 Partitions the global graph into K client subgraphs, then each client
 computes centralities using ONLY its own subgraph — it cannot see the rest
 of the network. Those LOCAL (distorted) values become the graph features.

 WHAT CHANGES vs STEP 1
   Step 1: centralities computed once on the FULL graph      (global, true)
   Step 2: centralities computed inside each client subgraph (local, distorted)
   The traffic features are IDENTICAL in both steps — partitioning does not
   change ports/bytes/packets. So only the centrality columns differ.

 FEDERATED VISIBILITY RULE (important, and stated in the output)
   Each flow is owned by the client that owns its SOURCE node.
     • if the destination is in the SAME client -> that client sees both
       endpoints, so dst_<c> is the destination's local centrality
     • if the destination is in ANOTHER client -> the destination is INVISIBLE
       to this client, so dst_<c> = 0 and the flow is marked is_cross_client=1
   This mirrors reality: cross-partition edges are exactly what federation
   loses, and it keeps the row count identical to Step 1 so the two steps
   stay directly comparable.

 ALSO SAVED (needed for the Step 3 correction work)
   cut_ratio, cut_edges, degree_undir, client_size, client_size_frac
   for both endpoints, per node.

 Output:  edge_features_local_<K>client.csv
          partition_stats_<K>client.csv
          local_graph_cache_<K>.pkl        (skips recompute on reruns)
 Run:     python build_local_features.py   (needs NF-ToN-IoT.csv beside it)
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

N_CLIENTS   = 3            # number of federated clients (try 3, then 5)
LOUVAIN_RES = 0.8          # your chosen resolution from Week 4
SEED        = 42
METHOD      = "louvain"    # "louvain" or "label_prop"

RAW_FLOW_COLS = ["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
                 "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
                 "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS"]

CENTRALITIES = ["degree_undir", "in_degree", "out_degree",
                "betweenness", "closeness", "pagerank", "eigenvector"]

DROP_EXACT_DUPES = True    # same rule as Step 1, so row counts match
USE_CACHE = True

OUT_CSV   = f"edge_features_local_{N_CLIENTS}client.csv"
OUT_STATS = f"partition_stats_{N_CLIENTS}client.csv"
CACHE     = f"local_graph_cache_{N_CLIENTS}.pkl"


def build_graph(df):
    G = nx.DiGraph()
    for ip in pd.concat([df[SRC_COL], df[DST_COL]]).unique():
        G.add_node(ip)
    agg = df.groupby([SRC_COL, DST_COL]).size().reset_index(name="w")
    for _, r in agg.iterrows():
        G.add_edge(r[SRC_COL], r[DST_COL], weight=r["w"])
    print(f"  global graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")
    return G


def centralities(G):
    """Same seven definitions as Step 1 — applied here to a SUBGRAPH."""
    n = G.number_of_nodes()
    UG = G.to_undirected()
    out = {}
    out["degree_undir"] = dict(UG.degree())
    out["in_degree"]    = dict(G.in_degree())
    out["out_degree"]   = dict(G.out_degree())
    if n > 2:
        out["betweenness"] = nx.betweenness_centrality(G, normalized=False)
    else:
        out["betweenness"] = {v: 0.0 for v in G}
    out["closeness"] = nx.closeness_centrality(G, wf_improved=False)
    try:
        out["pagerank"] = nx.pagerank(G, alpha=0.85, max_iter=200)
    except Exception:
        out["pagerank"] = {v: (1.0 / n if n else 0.0) for v in G}
    try:
        out["eigenvector"] = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
    except Exception:
        try:
            out["eigenvector"] = nx.eigenvector_centrality(UG, max_iter=1000, tol=1e-6)
        except Exception:
            out["eigenvector"] = {v: 0.0 for v in G}
    return out


def make_clients(G, k):
    """Partition into k clients from real communities (Louvain res=0.8 or
    label propagation), packing communities greedily into the smallest client."""
    UG = G.to_undirected()
    if METHOD == "louvain":
        comms = [set(c) for c in nx.community.louvain_communities(
            UG, seed=SEED, resolution=LOUVAIN_RES)]
    else:
        comms = [set(c) for c in nx.community.asyn_lpa_communities(UG, seed=SEED)]
    print(f"  {METHOD}: {len(comms)} communities found")

    # If there are fewer communities than clients, split the LARGEST one by
    # running Louvain again on its own subgraph at a higher resolution.
    # (An earlier version split it randomly — that destroys the community
    # structure and collapses edge preservation to roughly 1/k, exactly the
    # failure mode of random partitioning. Never split randomly.)
    guard = 0
    while len(comms) < k and guard < 50:
        guard += 1
        comms.sort(key=len, reverse=True)
        big = comms.pop(0)
        if len(big) < 2:                      # cannot split a singleton
            comms.append(big)
            break
        sub = UG.subgraph(big)
        pieces = []
        for res in (LOUVAIN_RES * 2, LOUVAIN_RES * 4, LOUVAIN_RES * 8, 20.0):
            pieces = [set(c) for c in nx.community.louvain_communities(
                sub, seed=SEED, resolution=res)]
            if len(pieces) > 1:
                break
        if len(pieces) > 1:
            comms.extend(pieces)
            print(f"    split a {len(big)}-node community into "
                  f"{len(pieces)} sub-communities (recursive Louvain)")
        else:
            comms.append(big)                 # genuinely indivisible
            print(f"    could not split the {len(big)}-node community further")
            break

    parts, sizes = [set() for _ in range(k)], [0] * k
    for c in sorted(comms, key=len, reverse=True):
        j = sizes.index(min(sizes))
        parts[j].update(c); sizes[j] += len(c)
    node_to_client = {nd: i for i, p in enumerate(parts) for nd in p}
    return parts, node_to_client


def cut_stats(G, node_to_client):
    """Per node: undirected degree, how many of its edges cross clients, ratio."""
    UG = G.to_undirected()
    out = {}
    for v in UG.nodes():
        nbrs = list(UG.neighbors(v))
        deg = len(nbrs)
        cut = sum(1 for nb in nbrs
                  if node_to_client.get(nb) != node_to_client.get(v))
        out[v] = (deg, cut, (cut / deg) if deg else 0.0)
    return out


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

    sig = (os.path.basename(DATASET_PATH), len(df), N_CLIENTS, METHOD, LOUVAIN_RES)
    blob = None
    if USE_CACHE and os.path.exists(CACHE):
        try:
            with open(CACHE, "rb") as f:
                cand = pickle.load(f)
            if cand.get("sig") == sig:
                blob = cand
                print(f"  loaded partition + local centralities from {CACHE}")
        except Exception as e:
            print(f"  cache unreadable ({e}) -> recomputing")

    if blob is None:
        G = build_graph(df)
        parts, node_to_client = make_clients(G, N_CLIENTS)
        n_global = G.number_of_nodes()

        print(f"  computing LOCAL centralities inside each of "
              f"{N_CLIENTS} client subgraphs ...")
        local = {}                       # node -> {metric: local value}
        client_size = {}
        for cid, part in enumerate(parts):
            sub = G.subgraph(part).copy()
            client_size[cid] = len(part)
            cvals = centralities(sub)
            for nd in part:
                local[nd] = {m: cvals[m].get(nd, 0.0) for m in CENTRALITIES}
            print(f"    client {cid}: {len(part):,} nodes, "
                  f"{sub.number_of_edges():,} internal edges")

        cuts = cut_stats(G, node_to_client)
        blob = {"sig": sig, "local": local, "node_to_client": node_to_client,
                "client_size": client_size, "cuts": cuts, "n_global": n_global}
        if USE_CACHE:
            with open(CACHE, "wb") as f:
                pickle.dump(blob, f)
            print(f"  saved cache -> {CACHE}")

    local          = blob["local"]
    node_to_client = blob["node_to_client"]
    client_size    = blob["client_size"]
    cuts           = blob["cuts"]
    n_global       = blob["n_global"]

    # ── partition statistics ──────────────────────────────────────────
    total_edges = sum(1 for _ in range(0))  # computed below from cuts
    deg_sum = sum(v[0] for v in cuts.values())
    cut_sum = sum(v[1] for v in cuts.values())
    kept_pct = 100 * (1 - cut_sum / deg_sum)
    print(f"\n  partition summary ({METHOD}, res={LOUVAIN_RES}, "
          f"{N_CLIENTS} clients)")
    print(f"    client sizes: {dict(sorted(client_size.items()))}")
    print(f"    edges kept inside clients: {kept_pct:.1f}%  "
          f"(cut: {100 - kept_pct:.1f}%)")

    # quality warnings — a bad partition invalidates the federation experiment
    if kept_pct < 70:
        print(f"    ** WARNING: only {kept_pct:.1f}% of edges kept. A good "
              f"community partition should keep ~85%+.")
        print( "       This many clients does not fit the graph's community "
               "structure. Try fewer clients.")
    tiny = {c: s for c, s in client_size.items() if s < 10}
    if tiny:
        print(f"    ** WARNING: near-empty client(s) {tiny} — too few real "
              f"communities for {N_CLIENTS} clients.")
    biggest = max(client_size.values()) / sum(client_size.values())
    if biggest > 0.60:
        print(f"    ** NOTE: largest client holds {100*biggest:.1f}% of nodes "
              f"(imbalanced federation; it will dominate FedAvg).")

    stats = pd.DataFrame([{"client": c, "n_nodes": s,
                           "size_frac": s / n_global}
                          for c, s in sorted(client_size.items())])
    stats.to_csv(OUT_STATS, index=False, float_format="%.6f")

    # ── attach LOCAL centralities to every flow ───────────────────────
    print("\n  attaching local centralities to each flow ...")
    out = df[[SRC_COL, DST_COL] + raw_cols + [LABEL_COL, ATTACK_COL]].copy()
    out["src_client"] = out[SRC_COL].map(node_to_client)
    out["dst_client"] = out[DST_COL].map(node_to_client)
    out["is_cross_client"] = (out["src_client"] != out["dst_client"]).astype(int)

    for c in CENTRALITIES:
        src_map = {nd: vals[c] for nd, vals in local.items()}
        out[f"src_{c}"] = out[SRC_COL].map(src_map).fillna(0.0).astype(float)
        # destination is only visible if it sits in the SAME client
        dst_vals = out[DST_COL].map(src_map).fillna(0.0).astype(float)
        out[f"dst_{c}"] = np.where(out["is_cross_client"] == 1, 0.0, dst_vals)

    # extra columns for Step 3 (correction functions)
    out["src_degree_undir_global"] = out[SRC_COL].map(lambda v: cuts.get(v, (0,0,0))[0])
    out["src_cut_edges"]  = out[SRC_COL].map(lambda v: cuts.get(v, (0,0,0))[1])
    out["src_cut_ratio"]  = out[SRC_COL].map(lambda v: cuts.get(v, (0,0,0))[2])
    out["dst_cut_ratio"]  = out[DST_COL].map(lambda v: cuts.get(v, (0,0,0))[2])
    out["client_size"]      = out["src_client"].map(client_size)
    out["client_size_frac"] = out["client_size"] / n_global

    if DROP_EXACT_DUPES:
        before = len(out)
        out = out.drop_duplicates(keep="first").reset_index(drop=True)
        print(f"  dropped {before - len(out):,} exact-duplicate rows")

    out.to_csv(OUT_CSV, index=False, float_format="%.6f")

    print(f"\n  final rows: {len(out):,}")
    print(f"  cross-client flows (destination invisible): "
          f"{out['is_cross_client'].sum():,} "
          f"({100 * out['is_cross_client'].mean():.1f}%)")
    print("  class distribution:")
    vc = out[ATTACK_COL].value_counts()
    for cls, cnt in vc.items():
        print(f"      {str(cls):<14}{cnt:>9,}  ({100*cnt/len(out):5.1f}%)")
    print(f"  saved: {OUT_CSV}")
    print(f"  saved: {OUT_STATS}")
    print(f"  done in {time.time()-t0:.1f}s.  Next: python run_all_models.py")


if __name__ == "__main__":
    main()