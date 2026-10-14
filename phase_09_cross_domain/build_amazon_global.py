"""
═══════════════════════════════════════════════════════════════════════════
 AMAZON COMPUTERS — STEP 1 (GLOBAL): build the per-NODE feature table
═══════════════════════════════════════════════════════════════════════════

 Second-dataset validation of the IDS study, in a new domain.

 Here we classify NODES, not edges:
   node   = a product on Amazon (computers & accessories)
   edge   = two products frequently bought together (real co-purchase)
   label  = product category (10 classes)  <- multi-class, like the 10 attacks

 One row per product, with:

   TARGET
     label            -> 0..9  (product category)

   PRODUCT FEATURES  (Model A — "no centralities")
     feat_0 .. feat_766   review bag-of-words (the product's own properties,
                          analogous to the traffic features in the IDS work)

   CENTRALITY FEATURES  (Model B — "with centralities")
     degree_undir, betweenness, closeness, pagerank
     computed once on the FULL global co-purchase graph.

 SPEED NOTE
   13,752 nodes is too many for exact betweenness/closeness, so we use the
   standard sampling approximations (k pivot nodes). Degree and PageRank are
   exact. This mirrors real practice at this scale.

 Output:  amazon_features_global.csv
 Run:     python build_amazon_global.py
          (first: pip install torch_geometric)
═══════════════════════════════════════════════════════════════════════════
"""

import os, time, pickle, random, collections
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.datasets import Amazon

# ── Config ──────────────────────────────────────────────────────────────
DATA_ROOT   = "amazon_data"      # where PyG caches the raw download
BETW_K      = 500                # pivot nodes for approx betweenness
CLOSE_K     = 500                # pivot nodes for approx closeness
SEED        = 42
CACHE_PATH  = "amazon_graph_centralities.pkl"
OUT_CSV     = "amazon_features_global.csv"
CENTRALITIES = ["degree_undir", "betweenness", "closeness", "pagerank"]


def approx_closeness(G, k, seed):
    """Pivot-sampled closeness: BFS from k random nodes, estimate per node."""
    random.seed(seed)
    pivots = random.sample(list(G.nodes()), min(k, G.number_of_nodes()))
    dist_sum = collections.defaultdict(float)
    reach = collections.defaultdict(int)
    for p in pivots:
        for node, dd in nx.single_source_shortest_path_length(G, p).items():
            dist_sum[node] += dd
            reach[node] += 1
    out = {}
    for node in G.nodes():
        out[node] = (reach[node] - 1) / dist_sum[node] if dist_sum[node] > 0 else 0.0
    return out


def centralities(G):
    out = {}
    t = time.time()
    out["degree_undir"] = dict(G.degree())
    print(f"    degree      {time.time()-t:.1f}s")
    t = time.time()
    out["betweenness"] = nx.betweenness_centrality(G, k=min(BETW_K, G.number_of_nodes()),
                                                    seed=SEED, normalized=True)
    print(f"    betweenness (k={BETW_K} approx)  {time.time()-t:.1f}s")
    t = time.time()
    out["closeness"] = approx_closeness(G, CLOSE_K, SEED)
    print(f"    closeness   (k={CLOSE_K} approx)  {time.time()-t:.1f}s")
    t = time.time()
    out["pagerank"] = nx.pagerank(G, alpha=0.85, max_iter=200)
    print(f"    pagerank    {time.time()-t:.1f}s")
    return out


def main():
    t0 = time.time()
    print("Loading Amazon Computers via PyTorch Geometric ...")
    d = Amazon(root=DATA_ROOT, name="Computers")[0]
    n = d.num_nodes
    X = d.x.numpy()                      # (n, 767) review word features
    y = d.y.numpy()                      # (n,) class 0..9
    ei = d.edge_index.numpy()
    print(f"  {n:,} products, {X.shape[1]} features, {int(y.max())+1} classes, "
          f"{ei.shape[1]:,} directed co-purchase edges")

    # build undirected simple graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    print(f"  graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} "
          f"undirected edges")

    # centralities with cache
    sig = ("amazon_computers", n, G.number_of_edges(), BETW_K, CLOSE_K, SEED)
    cents = None
    if os.path.exists(CACHE_PATH):
        try:
            blob = pickle.load(open(CACHE_PATH, "rb"))
            if blob.get("sig") == sig:
                cents = blob["cents"]
                print("  loaded centralities from cache")
        except Exception:
            pass
    if cents is None:
        print("  computing global centralities ...")
        cents = centralities(G)
        pickle.dump({"sig": sig, "cents": cents}, open(CACHE_PATH, "wb"))
        print("  cached centralities")

    # assemble one row per node
    print("  assembling feature table ...")
    feat_cols = {f"feat_{j}": X[:, j] for j in range(X.shape[1])}
    df = pd.DataFrame(feat_cols)
    for c in CENTRALITIES:
        df[c] = [cents[c][i] for i in range(n)]
    df["label"] = y

    df.to_csv(OUT_CSV, index=False, float_format="%.6f")
    # verify
    back = pd.read_csv(OUT_CSV)
    assert len(back) == n and "label" in back.columns
    print(f"  verified: {OUT_CSV} holds {len(back):,} rows, "
          f"{int(back['label'].nunique())} classes")
    print("  class distribution:")
    vc = df["label"].value_counts().sort_index()
    for cls, cnt in vc.items():
        print(f"      class {cls}: {cnt:>6,}  ({100*cnt/n:4.1f}%)")
    print(f"  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
