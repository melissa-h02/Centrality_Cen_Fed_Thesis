"""
═══════════════════════════════════════════════════════════════════════════
 AMAZON COMPUTERS — STEP 2 (FEDERATED / LOCAL): local centralities
═══════════════════════════════════════════════════════════════════════════

 Same idea as the IDS Step 2, for the node-classification setting.

 Split the co-purchase graph into 3 clients with Louvain (resolution 0.8).
 Each client then computes centralities using ONLY the edges inside its own
 group — cross-client edges are invisible, exactly as in a real federation.
 A node's local centralities are therefore distorted (smaller) whenever some
 of its neighbours live in other clients.

 The product's own review features (feat_*) and label are unchanged — only
 the 4 centrality columns differ from Step 1.

 Output:  amazon_features_local_3client.csv
          amazon_local_cache_3.pkl   (per-node local values + cut info,
                                       kept for a future correction step)
 Run:     python build_amazon_local.py
═══════════════════════════════════════════════════════════════════════════
"""

import os, time, pickle, random, collections
import numpy as np, pandas as pd, networkx as nx
from torch_geometric.datasets import Amazon

try:
    import community as community_louvain      # python-louvain
except Exception:
    community_louvain = None

DATA_ROOT   = "amazon_data"
N_CLIENTS   = 3
RESOLUTION  = 0.8
BETW_K      = 500
CLOSE_K     = 500
SEED        = 42
GLOBAL_CACHE = "amazon_graph_centralities.pkl"   # reuse Step 1 global values
OUT_CSV     = "amazon_features_local_3client.csv"
CACHE_OUT   = f"amazon_local_cache_{N_CLIENTS}.pkl"
CENTRALITIES = ["degree_undir", "betweenness", "closeness", "pagerank"]


def approx_closeness(G, k, seed):
    if G.number_of_nodes() == 0:
        return {}
    random.seed(seed)
    pivots = random.sample(list(G.nodes()), min(k, G.number_of_nodes()))
    dist_sum = collections.defaultdict(float); reach = collections.defaultdict(int)
    for p in pivots:
        for node, dd in nx.single_source_shortest_path_length(G, p).items():
            dist_sum[node] += dd; reach[node] += 1
    return {node: ((reach[node]-1)/dist_sum[node] if dist_sum[node] > 0 else 0.0)
            for node in G.nodes()}


def local_centralities(H):
    out = {}
    out["degree_undir"] = dict(H.degree())
    kb = min(BETW_K, H.number_of_nodes())
    out["betweenness"] = (nx.betweenness_centrality(H, k=kb, seed=SEED, normalized=True)
                          if H.number_of_nodes() > 2 else {v: 0.0 for v in H})
    out["closeness"] = approx_closeness(H, CLOSE_K, SEED)
    try:
        out["pagerank"] = nx.pagerank(H, alpha=0.85, max_iter=200)
    except Exception:
        out["pagerank"] = {v: 1.0/max(H.number_of_nodes(), 1) for v in H}
    return out


def louvain_3(G):
    if community_louvain is None:
        raise SystemExit("Install python-louvain:  pip install python-louvain")
    part = community_louvain.best_partition(G, resolution=RESOLUTION,
                                            random_state=SEED)
    comms = collections.defaultdict(list)
    for node, c in part.items():
        comms[c].append(node)
    # pack communities into N_CLIENTS by adding each to the smallest client
    order = sorted(comms.values(), key=len, reverse=True)
    clients = [[] for _ in range(N_CLIENTS)]
    for comm in order:
        smallest = min(range(N_CLIENTS), key=lambda i: len(clients[i]))
        clients[smallest].extend(comm)
    node_to_client = {}
    for ci, members in enumerate(clients):
        for nd in members:
            node_to_client[nd] = ci
    return node_to_client, [len(c) for c in clients]


def main():
    t0 = time.time()
    print("Loading Amazon Computers ...")
    d = Amazon(root=DATA_ROOT, name="Computers")[0]
    n = d.num_nodes
    X = d.x.numpy(); y = d.y.numpy(); ei = d.edge_index.numpy()
    G = nx.Graph(); G.add_nodes_from(range(n))
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    print(f"  {n:,} nodes, {G.number_of_edges():,} edges")

    print(f"  Louvain (resolution {RESOLUTION}) -> {N_CLIENTS} clients ...")
    n2c, sizes = louvain_3(G)
    print(f"  client sizes: {sizes}")

    # count cut edges and build per-client subgraphs
    cut = 0
    for u, v in G.edges():
        if n2c[u] != n2c[v]:
            cut += 1
    kept = G.number_of_edges() - cut
    print(f"  edges kept inside clients: {kept:,} "
          f"({100*kept/G.number_of_edges():.1f}%)   cut: {cut:,}")

    # local centralities per client
    local = {c: {} for c in CENTRALITIES}
    cuts = {}
    print("  computing local centralities per client ...")
    for ci in range(N_CLIENTS):
        members = [nd for nd in range(n) if n2c[nd] == ci]
        H = G.subgraph(members).copy()
        lc = local_centralities(H)
        for c in CENTRALITIES:
            for nd in members:
                local[c][nd] = lc[c].get(nd, 0.0)
        # cut info per node (for a future correction step)
        for nd in members:
            deg_full = G.degree(nd)
            deg_local = H.degree(nd)
            cut_edges = deg_full - deg_local
            cuts[nd] = (deg_full, cut_edges,
                        cut_edges/deg_full if deg_full else 0.0)
        print(f"    client {ci}: {len(members):,} nodes")

    # assemble the feature table (features + LOCAL centralities + label)
    print("  assembling feature table ...")
    df = pd.DataFrame({f"feat_{j}": X[:, j] for j in range(X.shape[1])})
    for c in CENTRALITIES:
        df[c] = [local[c][i] for i in range(n)]
    df["label"] = y
    df["client"] = [n2c[i] for i in range(n)]
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")
    back = pd.read_csv(OUT_CSV)
    assert len(back) == n
    print(f"  verified: {OUT_CSV} holds {len(back):,} rows")

    # cache local values + cut info + client map (+ global if available)
    blob = {"local": local, "cuts": cuts, "node_to_client": n2c,
            "client_size": {c: sizes[c] for c in range(N_CLIENTS)},
            "n_global": n}
    pickle.dump(blob, open(CACHE_OUT, "wb"))
    print(f"  saved: {CACHE_OUT}")
    print(f"  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
