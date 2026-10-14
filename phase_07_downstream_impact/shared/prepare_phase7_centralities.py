"""
Prepare the full-node centrality table used by Phase 7.

Historical/original correction path:
- Build the full NF-ToN-IoT graph.
- Louvain resolution 0.8.
- Evaluate downstream experiments on K=3 clients.
- Use K=5 as the independent correction-fitting partition.
- Degree: exact correction, global_degree = local_degree + cut_edges.
- Closeness/PageRank: RandomForestRegressor, matching the original correction study
  (200 trees, max_depth=6, random_state=42), trained on K=5 and applied to K=3.
- Betweenness: left uncorrected, consistent with the thesis conclusion.

Output columns:
node,
global_degree, local_degree, corrected_degree,
global_betweenness, local_betweenness, corrected_betweenness,
global_closeness, local_closeness, corrected_closeness,
global_pagerank, local_pagerank, corrected_pagerank

Run from repository root:
    python phase_07_downstream_impact/prepare_phase7_centralities.py NF-ToN-IoT.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

SEED = 42
LOUVAIN_RES = 0.8
METRICS = ("degree", "betweenness", "closeness", "pagerank")


def build_graph(df: pd.DataFrame, src_col: str, dst_col: str) -> nx.DiGraph:
    df = df.copy()
    df[src_col] = df[src_col].astype(str)
    df[dst_col] = df[dst_col].astype(str)

    G = nx.DiGraph()
    for ip in pd.concat([df[src_col], df[dst_col]]).unique():
        G.add_node(ip)

    # Same weighted graph convention used by the original downstream code:
    # one directed edge per (src,dst), weight = number of flows.
    agg = df.groupby([src_col, dst_col]).size().reset_index(name="weight")
    for s, d, w in agg.itertuples(index=False, name=None):
        G.add_edge(str(s), str(d), weight=int(w))
    return G


def centralities(G: nx.DiGraph) -> dict[str, dict[str, float]]:
    n = G.number_of_nodes()
    if n == 0:
        return {m: {} for m in METRICS}

    UG = G.to_undirected()
    out: dict[str, dict[str, float]] = {}
    out["degree"] = {str(v): float(d) for v, d in UG.degree()}
    out["betweenness"] = (
        {str(k): float(v) for k, v in nx.betweenness_centrality(G, normalized=False).items()}
        if n > 2
        else {str(v): 0.0 for v in G}
    )
    out["closeness"] = {
        str(k): float(v)
        for k, v in nx.closeness_centrality(G, wf_improved=False).items()
    }
    try:
        out["pagerank"] = {
            str(k): float(v)
            for k, v in nx.pagerank(G, alpha=0.85, max_iter=200, weight="weight").items()
        }
    except Exception:
        out["pagerank"] = {str(v): 1.0 / max(n, 1) for v in G}
    return out


def pack_communities(comms: list[set[str]], k: int) -> tuple[list[set[str]], dict[str, int]]:
    communities = [set(map(str, c)) for c in comms]

    # Historical fallback if fewer natural communities than requested clients:
    # recursively split the largest community structurally, not randomly.
    guard = 0
    while len(communities) < k and guard < 50:
        guard += 1
        communities.sort(key=len, reverse=True)
        big = communities.pop(0)
        if len(big) < 2:
            communities.append(big)
            break

        sub = nx.Graph()
        sub.add_nodes_from(big)
        # Caller will normally have >= k communities on this dataset; this
        # fallback is retained only to guarantee k non-empty groups.
        nodes = sorted(big)
        half = len(nodes) // 2
        communities.append(set(nodes[:half]))
        communities.append(set(nodes[half:]))

    parts = [set() for _ in range(k)]
    sizes = [0] * k
    for comm in sorted(communities, key=len, reverse=True):
        j = sizes.index(min(sizes))
        parts[j].update(comm)
        sizes[j] += len(comm)

    node_to_client = {n: cid for cid, part in enumerate(parts) for n in part}
    return parts, node_to_client


def make_clients(G: nx.DiGraph, k: int) -> tuple[list[set[str]], dict[str, int]]:
    UG = G.to_undirected()
    comms = [
        set(map(str, c))
        for c in nx.community.louvain_communities(
            UG, seed=SEED, resolution=LOUVAIN_RES, weight="weight"
        )
    ]
    parts = [set() for _ in range(k)]
    sizes = [0] * k
    for comm in sorted(comms, key=len, reverse=True):
        j = sizes.index(min(sizes))
        parts[j].update(comm)
        sizes[j] += len(comm)

    # If Louvain gives fewer than k communities, split the largest populated
    # client deterministically so the experiment remains executable.
    while sum(bool(p) for p in parts) < k:
        j = max(range(k), key=lambda z: len(parts[z]))
        empty = next(z for z in range(k) if not parts[z])
        nodes = sorted(parts[j])
        half = max(1, len(nodes) // 2)
        moved = set(nodes[:half])
        parts[j] -= moved
        parts[empty] |= moved

    node_to_client = {n: cid for cid, part in enumerate(parts) for n in part}
    return parts, node_to_client


def local_table(
    G: nx.DiGraph,
    global_c: dict[str, dict[str, float]],
    k: int,
) -> pd.DataFrame:
    parts, n2c = make_clients(G, k)
    UG = G.to_undirected()
    N = G.number_of_nodes()

    local_by_client: dict[int, dict[str, dict[str, float]]] = {}
    for cid, part in enumerate(parts):
        local_by_client[cid] = centralities(G.subgraph(part).copy())

    rows = []
    for node in map(str, G.nodes()):
        cid = n2c[node]
        nbrs = [str(x) for x in UG.neighbors(node)]
        cut_edges = sum(1 for nb in nbrs if n2c.get(nb) != cid)
        degree_full = len(nbrs)
        cut_ratio = cut_edges / degree_full if degree_full else 0.0
        n_client = len(parts[cid])

        row = {
            "node": node,
            "client": cid,
            "n_client": n_client,
            "n_global": N,
            "client_size_frac": n_client / N,
            "cut_edges": cut_edges,
            "cut_ratio": cut_ratio,
            "degree_full_undir": degree_full,
        }

        loc = local_by_client[cid]
        for m in METRICS:
            row[f"global_{m}"] = global_c[m].get(node, 0.0)
            row[f"local_{m}"] = loc[m].get(node, 0.0)
        rows.append(row)

    return pd.DataFrame(rows)


def fit_rf(train: pd.DataFrame, metric: str) -> RandomForestRegressor:
    cols = [f"local_{metric}", "cut_ratio", "client_size_frac"]
    tr = train[cols + [f"global_{metric}"]].replace([np.inf, -np.inf], np.nan).dropna()

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(tr[cols].to_numpy(float), tr[f"global_{metric}"].to_numpy(float))
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Phase 7 centrality table")
    p.add_argument("dataset")
    p.add_argument("--src-col", default="IPV4_SRC_ADDR")
    p.add_argument("--dst-col", default="IPV4_DST_ADDR")
    p.add_argument("--output", default="phase7_centralities.csv")
    args = p.parse_args()

    print(f"Loading {args.dataset} ...")
    df = pd.read_csv(args.dataset, low_memory=False)
    G = build_graph(df, args.src_col, args.dst_col)
    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} directed edges")

    print("Computing global centralities ...")
    gc = centralities(G)

    print("Building K=3 downstream/evaluation partition ...")
    apply_tbl = local_table(G, gc, 3)

    print("Building K=5 correction-fitting partition ...")
    fit_tbl = local_table(G, gc, 5)

    out = apply_tbl[[
        "node",
        "global_degree", "local_degree",
        "global_betweenness", "local_betweenness",
        "global_closeness", "local_closeness",
        "global_pagerank", "local_pagerank",
    ]].copy()

    # Exact degree correction.
    out["corrected_degree"] = (
        apply_tbl["local_degree"].to_numpy(float)
        + apply_tbl["cut_edges"].to_numpy(float)
    )

    # Betweenness remains uncorrected in the final thesis protocol.
    out["corrected_betweenness"] = out["local_betweenness"].to_numpy(float)

    # Historical learned correction model retained by the user's original
    # Phase 6 scripts: RF trained on an independent client-count split.
    for metric in ("closeness", "pagerank"):
        model = fit_rf(fit_tbl, metric)
        cols = [f"local_{metric}", "cut_ratio", "client_size_frac"]
        X = apply_tbl[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        out[f"corrected_{metric}"] = model.predict(X)

    # Arrange columns exactly how current Phase 7 scripts expect them.
    ordered = ["node"]
    for m in METRICS:
        ordered += [f"global_{m}", f"local_{m}", f"corrected_{m}"]
    out = out[ordered]

    out.to_csv(args.output, index=False)
    print(f"\nSaved {len(out):,} full-node rows to {args.output}")
    print("Columns:")
    print("  " + ", ".join(out.columns))


if __name__ == "__main__":
    main()
