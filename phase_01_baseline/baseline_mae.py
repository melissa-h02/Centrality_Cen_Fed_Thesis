"""
Phase 1 — Baseline MAE analysis and metric selection
====================================================

Clean reproduction of the initial NF-ToN-IoT experiment reported in the thesis.

Purpose
-------
1. Build the NF-ToN-IoT communication graph.
2. Compute five global centrality measures:
   - degree
   - betweenness
   - closeness
   - PageRank
   - eigenvector
3. Partition the graph into 3 clients using:
   - Random
   - Degree-balanced
   - Stratified
   - Community (Greedy Modularity)
4. Recompute the same centralities independently inside each client.
5. Compare local and global values using Mean Absolute Error (MAE).

This is intentionally the initial Phase-1 implementation. Later thesis phases
switch to Spearman rank correlation and use the final standardized protocol.

Expected thesis table (rounded to 4 decimals):
                    Degree  Betweenness  Closeness  PageRank  Eigenvector
Random              0.0011       0.0001     0.0312    0.0014       0.0023
Degree-balanced     0.0010       0.0001     0.0313    0.0014       0.0029
Stratified          0.0011       0.0001     0.0286    0.0014       0.0023
Community           0.0041       0.0005     0.0224    0.0013       0.0058

Notes
-----
- The script preserves the initial implementation choices from the original
  experiment, including sampled betweenness.
- Small numerical differences can occur across NetworkX / NumPy versions.
"""

from __future__ import annotations

import os
import random
import warnings

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CSV_PATH = "NF-ToN-IoT.csv"
OUTPUT_DIR = "output_phase1_mae"

N_CLIENTS = 3
SEED = 42

SRC_COL = "IPV4_SRC_ADDR"
DST_COL = "IPV4_DST_ADDR"
LABEL_COL = "Label"
ATTACK_COL = "Attack"
BYTES_COL = "IN_BYTES"

METRICS = ["degree", "betweenness", "closeness", "pagerank", "eigenvector"]


# ---------------------------------------------------------------------------
# DATA + GRAPH
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Put NF-ToN-IoT.csv beside this script "
            "or change CSV_PATH."
        )

    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)

    required = [SRC_COL, DST_COL, LABEL_COL, ATTACK_COL, BYTES_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df[SRC_COL] = df[SRC_COL].astype(str)
    df[DST_COL] = df[DST_COL].astype(str)

    print(f"  {len(df):,} flows")
    return df


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build the directed IP communication graph used in the initial study."""
    print("Building graph ...")

    G = nx.DiGraph()

    attack_ips = (
        set(df.loc[df[LABEL_COL] == 1, SRC_COL].unique())
        | set(df.loc[df[LABEL_COL] == 1, DST_COL].unique())
    )

    all_ips = pd.concat([df[SRC_COL], df[DST_COL]]).unique()
    for ip in all_ips:
        G.add_node(ip, role="attack" if ip in attack_ips else "benign")

    edge_agg = (
        df.groupby([SRC_COL, DST_COL, LABEL_COL])
        .agg(
            flow_count=(LABEL_COL, "count"),
            total_bytes=(BYTES_COL, "sum"),
            attack_type=(ATTACK_COL, lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )

    for _, row in edge_agg.iterrows():
        G.add_edge(
            row[SRC_COL],
            row[DST_COL],
            weight=row["flow_count"],
            total_bytes=row["total_bytes"],
            label=int(row[LABEL_COL]),
            attack_type=row["attack_type"],
        )

    print(
        f"  graph: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} directed edges"
    )
    return G


# ---------------------------------------------------------------------------
# CENTRALITIES
# ---------------------------------------------------------------------------

def safe_eigenvector(G: nx.DiGraph) -> dict:
    """
    Initial-study eigenvector fallback:
    compute on the largest strongly connected component when possible,
    otherwise fall back to the largest weakly connected component.
    Nodes outside the selected component receive 0.
    """
    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_nodes() < 3:
        return {n: 0.0 for n in G.nodes()}

    components = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
    largest = components[0]

    if len(largest) < 3:
        components = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
        largest = components[0]

    if len(largest) < 3:
        return {n: 0.0 for n in G.nodes()}

    H = G.subgraph(largest).copy()

    try:
        eig = nx.eigenvector_centrality(
            H,
            max_iter=100000,
            tol=1e-4,
            weight="weight",
        )
    except nx.PowerIterationFailedConvergence:
        try:
            eig = nx.eigenvector_centrality_numpy(
                H.to_undirected(),
                weight="weight",
            )
        except Exception:
            return {n: 0.0 for n in G.nodes()}

    return {n: eig.get(n, 0.0) for n in G.nodes()}


def compute_global_centralities(G: nx.DiGraph) -> dict[str, dict]:
    """
    Preserve the initial experiment's computation choices.

    In the original Phase-1 code:
      - degree used NetworkX degree_centrality
      - global betweenness used k=300
      - PageRank used alpha=0.85
      - eigenvector used the safe component fallback
    """
    print("Computing GLOBAL centralities ...")

    out = {
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(
            G,
            k=min(300, G.number_of_nodes()),
            normalized=True,
            seed=SEED,
        ),
        "closeness": nx.closeness_centrality(G),
        "pagerank": nx.pagerank(G, alpha=0.85, max_iter=200),
        "eigenvector": safe_eigenvector(G),
    }
    return out


def compute_local_centralities(C: nx.DiGraph) -> dict[str, dict]:
    """
    Recompute centralities independently inside one client subgraph.

    This keeps the initial implementation's sampled local betweenness
    (k=min(150, client_size)).
    """
    n = C.number_of_nodes()

    if n == 0:
        return {m: {} for m in METRICS}

    if n == 1:
        node = next(iter(C.nodes()))
        return {m: {node: 0.0} for m in METRICS}

    try:
        pagerank = nx.pagerank(C, alpha=0.85, max_iter=200)
    except Exception:
        pagerank = nx.pagerank(C, alpha=0.85, max_iter=500, tol=1e-4)

    return {
        "degree": nx.degree_centrality(C),
        "betweenness": nx.betweenness_centrality(
            C,
            k=min(150, n),
            normalized=True,
            seed=SEED,
        ),
        "closeness": nx.closeness_centrality(C),
        "pagerank": pagerank,
        "eigenvector": safe_eigenvector(C),
    }


# ---------------------------------------------------------------------------
# PARTITIONING
# ---------------------------------------------------------------------------

def partition_random(G: nx.DiGraph, k: int) -> list[set]:
    nodes = list(G.nodes())
    rng = random.Random(SEED)
    rng.shuffle(nodes)

    parts = [set() for _ in range(k)]
    for i, node in enumerate(nodes):
        parts[i % k].add(node)
    return parts


def partition_degree_balanced(G: nx.DiGraph, k: int) -> list[set]:
    nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    parts = [set() for _ in range(k)]

    for i, node in enumerate(nodes):
        parts[i % k].add(node)

    return parts


def partition_stratified(G: nx.DiGraph, k: int) -> list[set]:
    attack_nodes = [
        n for n in G.nodes()
        if G.nodes[n].get("role") == "attack"
    ]
    benign_nodes = [
        n for n in G.nodes()
        if G.nodes[n].get("role") != "attack"
    ]

    rng = random.Random(SEED)
    rng.shuffle(attack_nodes)
    rng.shuffle(benign_nodes)

    parts = [set() for _ in range(k)]

    for i, node in enumerate(attack_nodes):
        parts[i % k].add(node)

    for i, node in enumerate(benign_nodes):
        parts[i % k].add(node)

    return parts


def partition_community(G: nx.DiGraph, k: int) -> list[set]:
    """
    Initial Phase-1 community strategy: Greedy Modularity.

    Communities are detected on the undirected graph and greedily packed
    into the currently smallest client.
    """
    U = G.to_undirected()
    communities = list(nx.community.greedy_modularity_communities(U))
    communities = sorted(communities, key=len, reverse=True)

    print(f"  Greedy Modularity found {len(communities)} communities")

    parts = [set() for _ in range(k)]
    sizes = [0] * k

    for community in communities:
        client = sizes.index(min(sizes))
        parts[client].update(community)
        sizes[client] += len(community)

    return parts


# ---------------------------------------------------------------------------
# PHASE-1 EXPERIMENT
# ---------------------------------------------------------------------------

def evaluate_strategy(
    G: nx.DiGraph,
    global_centrality: dict[str, dict],
    partitions: list[set],
) -> dict[str, float]:
    """Compute MAE(global, local) over all nodes for each centrality."""

    local_values = {m: {} for m in METRICS}

    for client_id, members in enumerate(partitions, start=1):
        C = G.subgraph(members).copy()
        centrality = compute_local_centralities(C)

        for metric in METRICS:
            local_values[metric].update(centrality[metric])

        print(
            f"    client {client_id}: "
            f"{C.number_of_nodes():,} nodes, "
            f"{C.number_of_edges():,} edges"
        )

    maes = {}

    for metric in METRICS:
        errors = [
            abs(
                global_centrality[metric].get(node, 0.0)
                - local_values[metric].get(node, 0.0)
            )
            for node in G.nodes()
        ]
        maes[metric] = float(np.mean(errors))

    return maes


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    df = load_data(CSV_PATH)
    G = build_graph(df)
    global_centrality = compute_global_centralities(G)

    strategy_functions = {
        "Random": partition_random,
        "Degree-balanced": partition_degree_balanced,
        "Stratified": partition_stratified,
        "Community": partition_community,
    }

    rows = []

    print("\n" + "=" * 90)
    print("PHASE 1 — INITIAL MAE COMPARISON")
    print("=" * 90)

    for name, fn in strategy_functions.items():
        print(f"\nStrategy: {name}")
        partitions = fn(G, N_CLIENTS)
        maes = evaluate_strategy(G, global_centrality, partitions)

        rows.append({
            "Strategy": name,
            "Degree": maes["degree"],
            "Betweenness": maes["betweenness"],
            "Closeness": maes["closeness"],
            "PageRank": maes["pagerank"],
            "Eigenvector": maes["eigenvector"],
        })

    results = pd.DataFrame(rows).set_index("Strategy")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, "phase1_mae_summary.csv")
    results.to_csv(out_csv, float_format="%.8f")

    print("\n" + "=" * 90)
    print("INITIAL MAE COMPARISON (lower = smaller numerical error)")
    print("=" * 90)
    print(results.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 90)
    print(f"\nSaved: {out_csv}")
    print(
        "\nReminder: Phase 1 was retained to document why MAE was later "
        "replaced by Spearman rank correlation as the primary metric."
    )


if __name__ == "__main__":
    main()
