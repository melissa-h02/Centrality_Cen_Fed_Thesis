"""
DETAILED COMMUNITY-DETECTION ALGORITHM COMPARISON
--------------------------------------------------
Runs exactly the nine Phase-3 configurations reported in the thesis:

  • Louvain (resolution = 0.8)
  • Louvain (resolution = 1.0)
  • Louvain (resolution = 1.5)
  • Leiden  (resolution = 1.0)
  • Leiden  (resolution = 2.0)
  • Label Propagation
  • Infomap
  • Greedy Modularity
  • Girvan-Newman

For each configuration and K = 3, 5, 7 clients, the script reports
centrality rank/value preservation, edge and attack-edge preservation,
community structure, and runtime. The early Phase-3 comparison uses the
same seven centrality metrics as the historical algorithm-comparison run.

Dependencies for the complete nine-configuration run:
    pip install igraph leidenalg
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import time
import warnings
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

# ── Phase-3 graph-community dependencies ─────────────────────
# Infomap uses python-igraph; Leiden uses leidenalg on the same igraph graph.
# They are required here because the thesis comparison contains all 9 rows.
try:
    import igraph as ig
    import leidenalg as la
except ImportError as exc:
    raise ImportError(
        "Phase 3 requires igraph and leidenalg. Install them with: "
        "pip install igraph leidenalg"
    ) from exc


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
CSV_PATH      = r"NF-ToN-IoT.csv"
OUTPUT_DIR    = "output_community_algorithms_detailed"
CLIENT_COUNTS = [3, 5, 7]
MAIN_N_CLIENTS = 3       # used for the deep per-client visualisations

# Resolutions used in the thesis Phase-3 table only.
LOUVAIN_RESOLUTIONS = [0.8, 1.0, 1.5]
LEIDEN_RESOLUTIONS  = [1.0, 2.0]
# ══════════════════════════════════════════════════════════════

METRICS = ["degree", "out_deg", "in_deg", "betweenness",
           "closeness", "pagerank", "eigenvector"]

METRIC_LABELS = {
    "degree": "Degree", "out_deg": "Out-degree", "in_deg": "In-degree",
    "betweenness": "Betweenness", "closeness": "Closeness",
    "pagerank": "PageRank", "eigenvector": "Eigenvector",
}

# Colour per centrality (matches the per-metric stacked breakdown chart)
METRIC_COLORS = {
    "degree":      "#1976D2",  # blue
    "out_deg":     "#D32F2F",  # red
    "in_deg":      "#F48FB1",  # pink
    "betweenness": "#388E3C",  # green
    "closeness":   "#FFA000",  # amber
    "pagerank":    "#6D4C41",  # brown
    "eigenvector": "#90A4AE",  # blue-gray
}

# ── LIGHT Color palette (same as your Two-Level Analysis) ─────
BG     = "#FFFFFF"
PANEL  = "#F5F5F5"
GRID   = "#E0E0E0"
TEXT   = "#1A1A2E"
MUTED  = "#555555"
ATTACK = "#D32F2F"
BENIGN = "#2E7D32"

# One distinct colour per algorithm (kept consistent across every chart).
# Louvain variants use a blue gradient: lighter = lower resolution (fewer
# communities), darker = higher resolution (more communities).
LOUVAIN_BLUE_SHADES = {
    0.8: "#42A5F5",
    1.0: "#1976D2",
    1.5: "#0D47A1",
}

ALGO_COLORS = {
    "Greedy Modularity": "#F57F17",
    "Girvan-Newman":     "#D32F2F",
    "Infomap":           "#7B1FA2",
    "Label Propagation": "#00897B",
    "Leiden (res=1.0)":  "#43A047",
    "Leiden (res=2.0)":  "#7CB342",
}
for _res in LOUVAIN_RESOLUTIONS:
    ALGO_COLORS[f"Louvain (res={_res})"] = LOUVAIN_BLUE_SHADES[_res]


# ══════════════════════════════════════════════════════════════
# DATA LOADING & GRAPH BUILDING
# ══════════════════════════════════════════════════════════════

def load_csv(path):
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    print(f"   {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def build_graph(df):
    print("\nBuilding network graph...")
    G = nx.DiGraph()
    attack_ips = set(df[df["Label"] == 1]["IPV4_SRC_ADDR"].unique()) | \
                 set(df[df["Label"] == 1]["IPV4_DST_ADDR"].unique())
    all_ips = pd.concat([df["IPV4_SRC_ADDR"], df["IPV4_DST_ADDR"]]).unique()
    for ip in all_ips:
        G.add_node(ip, role="attack" if ip in attack_ips else "benign")
    edge_agg = (
        df.groupby(["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label"])
        .agg(flow_count=("Label", "count"),
             total_bytes=("IN_BYTES", "sum"),
             attack_type=("Attack", lambda x: x.mode()[0]))
        .reset_index()
    )
    for _, row in edge_agg.iterrows():
        G.add_edge(row["IPV4_SRC_ADDR"], row["IPV4_DST_ADDR"],
                   weight=row["flow_count"], total_bytes=row["total_bytes"],
                   label=int(row["Label"]), attack_type=row["attack_type"])
    print(f"   Nodes: {G.number_of_nodes():,}  |  Edges: {G.number_of_edges():,}")
    return G


# ══════════════════════════════════════════════════════════════
# CENTRALITY COMPUTATION
# ══════════════════════════════════════════════════════════════

def safe_eigenvector(G):
    if G.number_of_nodes() < 3:
        return {n: 0.0 for n in G.nodes()}
    components = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
    if not components:
        return {n: 0.0 for n in G.nodes()}
    largest = components[0]
    if len(largest) < 3:
        components = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
        if not components:
            return {n: 0.0 for n in G.nodes()}
        largest = components[0]
    if len(largest) < 3:
        return {n: 0.0 for n in G.nodes()}
    subG = G.subgraph(largest).copy()
    try:
        eig_sub = nx.eigenvector_centrality(subG, max_iter=1000, tol=1e-03,
                                            weight="weight")
    except nx.PowerIterationFailedConvergence:
        try:
            eig_sub = nx.eigenvector_centrality_numpy(subG.to_undirected(),
                                                      weight="weight")
        except Exception:
            return {n: 0.0 for n in G.nodes()}
    return {n: eig_sub.get(n, 0.0) for n in G.nodes()}


def compute_centralities_timed(G, label=""):
    timings, results = {}, {}
    nodes = list(G.nodes()); n = len(nodes)
    if n == 0:
        for m in METRICS: timings[m] = 0.0; results[m] = {}
        return results, timings
    if n == 1:
        for m in METRICS: timings[m] = 0.0; results[m] = {nodes[0]: 0.0}
        return results, timings

    t0 = time.time(); results["degree"]   = nx.degree_centrality(G);    timings["degree"]      = time.time() - t0
    t0 = time.time(); results["out_deg"]  = nx.out_degree_centrality(G); timings["out_deg"]    = time.time() - t0
    t0 = time.time(); results["in_deg"]   = nx.in_degree_centrality(G);  timings["in_deg"]     = time.time() - t0
    t0 = time.time(); results["betweenness"] = nx.betweenness_centrality(G, k=min(500, n),
                                                                          normalized=True, seed=42)
    timings["betweenness"] = time.time() - t0
    t0 = time.time(); results["closeness"] = nx.closeness_centrality(G);  timings["closeness"] = time.time() - t0
    t0 = time.time()
    try:
        results["pagerank"] = nx.pagerank(G, alpha=0.85, max_iter=200)
    except Exception:
        try:
            results["pagerank"] = nx.pagerank(G, alpha=0.85, max_iter=500, tol=1e-4)
        except Exception:
            results["pagerank"] = {nd: 1.0/n for nd in nodes}
    timings["pagerank"] = time.time() - t0
    t0 = time.time(); results["eigenvector"] = safe_eigenvector(G); timings["eigenvector"] = time.time() - t0

    total = sum(timings.values())
    if label:
        print(f"   [{label}] {n} nodes | {total:.2f}s")
    return results, timings


# ══════════════════════════════════════════════════════════════
# COMMUNITY-DETECTION ALGORITHMS
# ══════════════════════════════════════════════════════════════

def detect_louvain(G, resolution=1.0):
    U = G.to_undirected()
    return list(nx.community.louvain_communities(
        U, seed=42, resolution=resolution))

def detect_greedy_modularity(G):
    U = G.to_undirected()
    return list(nx.community.greedy_modularity_communities(U))

def detect_girvan_newman(G, max_level=None):
    """Classic GN — takes the first split of the hierarchy (like your previous code)."""
    U = G.to_undirected()
    comp_gen = nx.community.girvan_newman(U)
    try:
        first_split = next(comp_gen)
        return [set(c) for c in first_split]
    except StopIteration:
        return [set(U.nodes())]

def detect_label_propagation(G):
    """
    Asynchronous Label Propagation (Raghavan et al. 2007).
    Near-linear time O(E), no parameter tuning, but stochastic — we
    seed it for reproducibility.
    """
    U = G.to_undirected()
    # asyn_lpa_communities returns a generator of frozensets
    return [set(c) for c in nx.community.asyn_lpa_communities(
        U, weight="weight", seed=42)]

def detect_leiden(G, resolution=1.0):
    """Weighted Leiden on the undirected host graph, matching leiden_matched.py."""
    U = G.to_undirected()
    nodes = list(U.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in U.edges()]
    weights = [float(d.get("weight", 1.0)) for _, _, d in U.edges(data=True)]

    ig_G = ig.Graph(n=len(nodes), edges=edges, directed=False)
    if weights:
        ig_G.es["weight"] = weights

    partition = la.find_partition(
        ig_G,
        la.RBConfigurationVertexPartition,
        weights="weight" if weights else None,
        resolution_parameter=resolution,
        seed=42,
    )
    return [set(nodes[i] for i in community) for community in partition]


def detect_infomap(G):
    """
    Infomap (Rosvall & Bergstrom 2008) — minimises the map equation
    (description length of a random walker on the network). We use the
    implementation bundled inside python-igraph, which works on the
    DIRECTED, weighted graph natively — a real advantage for flow
    networks like ours.
    """
        # Map node IDs to integer indices for igraph
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Build a directed igraph from our networkx DiGraph, preserving weights
    edges = []
    weights = []
    for u, v, d in G.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(float(d.get("weight", 1.0)))

    ig_G = ig.Graph(n=len(nodes), edges=edges, directed=True)
    ig_G.es["weight"] = weights

    # Run Infomap (igraph parameter is `edge_weights`, not `weights`)
    clustering = ig_G.community_infomap(edge_weights="weight")

    # Convert back: each community is a set of original node IDs
    communities = []
    for comm in clustering:
        communities.append(set(nodes[i] for i in comm))
    return communities


# Exactly the 9 configurations reported in the thesis Phase-3 table.
# Dict insertion order is also the display/run order.
ALGORITHMS = {
    "Louvain (res=0.8)":  lambda G: detect_louvain(G, resolution=0.8),
    "Louvain (res=1.0)":  lambda G: detect_louvain(G, resolution=1.0),
    "Louvain (res=1.5)":  lambda G: detect_louvain(G, resolution=1.5),
    "Leiden (res=1.0)":   lambda G: detect_leiden(G, resolution=1.0),
    "Leiden (res=2.0)":   lambda G: detect_leiden(G, resolution=2.0),
    "Label Propagation":  detect_label_propagation,
    "Infomap":            detect_infomap,
    "Greedy Modularity":  detect_greedy_modularity,
    "Girvan-Newman":      detect_girvan_newman,
}


# ══════════════════════════════════════════════════════════════
# PARTITIONING: assign detected communities to n clients
# ══════════════════════════════════════════════════════════════

def communities_to_clients(communities, n_clients):
    """Pack communities into n clients with greedy-smallest-first balancing."""
    sorted_comms = sorted(communities, key=len, reverse=True)
    parts = [set() for _ in range(n_clients)]
    sizes = [0] * n_clients
    for comm in sorted_comms:
        j = sizes.index(min(sizes))
        parts[j].update(comm)
        sizes[j] += len(comm)
    return parts


def modularity_score(G, communities):
    try:
        return nx.community.modularity(G.to_undirected(), communities)
    except Exception:
        return np.nan


# ══════════════════════════════════════════════════════════════
# STYLING HELPERS
# ══════════════════════════════════════════════════════════════

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=11, pad=10, fontweight="bold")
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID); sp.set_linewidth(0.7)

def force_integer_yaxis(ax):
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))


# ══════════════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════════════

def run_all(G, output_dir, client_counts):
    os.makedirs(output_dir, exist_ok=True)
    nodes = list(G.nodes())

    total_edges   = G.number_of_edges()
    attack_edges  = [(u, v) for u, v, d in G.edges(data=True) if d.get("label") == 1]
    benign_edges  = [(u, v) for u, v, d in G.edges(data=True) if d.get("label") == 0]
    n_attack_e    = len(attack_edges)
    n_benign_e    = len(benign_edges)

    # ── 1. Global centralities (reference) ──────────────────
    print("\n" + "="*70)
    print("  STEP 1: CENTRALIZED (GLOBAL) CENTRALITIES")
    print("="*70)
    global_cents, global_timings = compute_centralities_timed(G, "Global")
    for m in METRICS:
        print(f"      {m:<14} {global_timings[m]:.4f}s")
    global_total_time = sum(global_timings.values())
    print(f"   TOTAL GLOBAL TIME: {global_total_time:.2f}s")

    # ── 2. Run each algorithm ONCE, reuse communities across 3/5/7 clients ─
    print("\n" + "="*70)
    print(f"  STEP 2: DETECTING COMMUNITIES ({len(ALGORITHMS)} algorithms, each runs once)")
    print("="*70)
    algo_result = {}
    for name, fn in ALGORITHMS.items():
        print(f"\n   Running {name}...")
        try:
            t0 = time.time()
            comms = fn(G)
            dt = time.time() - t0
            mod = modularity_score(G, comms)
            sizes = sorted([len(c) for c in comms], reverse=True)
            print(f"      time={dt:.2f}s | #comm={len(comms)} | mod={mod:.4f}")
            print(f"      all community sizes: {sizes}")
            algo_result[name] = {
                "communities":    comms,
                "detection_time": dt,
                "modularity":     mod,
                "n_communities":  len(comms),
                "sizes":          sizes,
            }
        except Exception as e:
            print(f"      ⚠  {name} failed: {e}")
            print(f"         skipping this algorithm.")
            continue

    if not algo_result:
        print("\n⚠  No algorithm succeeded — aborting.")
        return None

    # ── 3. Per-client analysis for every (algo, n_clients) ──
    all_timing_rows = [{"algorithm": "Centralized", "n_clients": 0, "client": "Global",
                        "n_nodes": len(nodes), "n_edges": total_edges,
                        **{f"time_{m}": global_timings[m] for m in METRICS},
                        "time_total": global_total_time}]
    all_change_rows, all_corr_rows = [], []
    all_summary   = []
    all_edge_rows, all_attack_type_rows = [], []
    algo_meta_rows = []

    for n_clients in client_counts:
        print("\n" + "="*70)
        print(f"  STEP 3: PER-CLIENT ANALYSIS  ({n_clients} clients)")
        print("="*70)

        for algo_name, meta in algo_result.items():
            print(f"\n   --- {algo_name} ({n_clients} clients) ---")
            partitions = communities_to_clients(meta["communities"], n_clients)

            node_to_client = {}
            for i, part in enumerate(partitions):
                for nd in part:
                    node_to_client[nd] = i + 1

            for i, part in enumerate(partitions):
                internal = sum(1 for u, v in G.edges() if u in part and v in part)
                cross    = sum(1 for u, v in G.edges()
                               if (u in part) != (v in part))
                atk_n    = sum(1 for nd in part if G.nodes[nd].get("role") == "attack")
                print(f"      Client {i+1}: {len(part):>4} nodes | "
                      f"{internal:>5} internal | {cross:>5} cross | "
                      f"{atk_n:>3} attack nodes")

            # ── centralities per client (timed) ─────────────
            client_cents, client_timings = {}, {}
            for i, part in enumerate(partitions):
                cid = i + 1
                sub_edges = [(u, v, d) for u, v, d in G.edges(data=True)
                             if u in part and v in part]
                C = nx.DiGraph()
                C.add_nodes_from([(nd, G.nodes[nd]) for nd in part])
                C.add_edges_from(sub_edges)
                cents_i, t_i = compute_centralities_timed(
                    C, label=f"{algo_name}/c{cid}")
                client_cents[cid], client_timings[cid] = cents_i, t_i

                row = {"algorithm": algo_name, "n_clients": n_clients,
                       "client": f"Client {cid}", "n_nodes": len(part),
                       "n_edges": C.number_of_edges(),
                       **{f"time_{m}": t_i[m] for m in METRICS},
                       "time_total": sum(t_i.values())}
                all_timing_rows.append(row)

            # ── change detection & correlation per client ──
            THRESHOLD = 0.001
            for cid in range(1, n_clients + 1):
                cl_nodes = [nd for nd in nodes if node_to_client.get(nd) == cid]
                if not cl_nodes:
                    continue

                ch_row = {"algorithm": algo_name, "n_clients": n_clients,
                          "client": f"Client {cid}", "n_nodes": len(cl_nodes)}
                co_row = {"algorithm": algo_name, "n_clients": n_clients,
                          "client": f"Client {cid}"}

                for m in METRICS:
                    diffs, g_vals, l_vals = [], [], []
                    for nd in cl_nodes:
                        g = global_cents[m].get(nd, 0.0)
                        l = client_cents[cid][m].get(nd, 0.0)
                        diffs.append(abs(l - g)); g_vals.append(g); l_vals.append(l)
                    diffs = np.array(diffs)
                    n_ch = int((diffs > THRESHOLD).sum())
                    ch_row[f"{m}_changed"]     = n_ch
                    ch_row[f"{m}_pct_changed"] = n_ch / len(cl_nodes) * 100
                    ch_row[f"{m}_mae"]         = diffs.mean()
                    ch_row[f"{m}_max_diff"]    = diffs.max()

                    if len(cl_nodes) >= 3 and np.std(g_vals) > 0 and np.std(l_vals) > 0:
                        co_row[f"{m}_pearson"]  = np.corrcoef(g_vals, l_vals)[0, 1]
                        co_row[f"{m}_spearman"] = spearmanr(g_vals, l_vals).correlation
                    else:
                        co_row[f"{m}_pearson"]  = np.nan
                        co_row[f"{m}_spearman"] = np.nan

                all_change_rows.append(ch_row)
                all_corr_rows.append(co_row)

            # ── edge preservation / attack type ─────────────
            internal_total = 0; internal_atk = 0; internal_ben = 0
            cross_total    = 0; cross_atk    = 0; cross_ben    = 0
            for u, v, d in G.edges(data=True):
                same = node_to_client.get(u) == node_to_client.get(v)
                is_atk = d.get("label") == 1
                if same:
                    internal_total += 1
                    if is_atk: internal_atk += 1
                    else:      internal_ben += 1
                else:
                    cross_total += 1
                    if is_atk: cross_atk += 1
                    else:      cross_ben += 1

            all_edge_rows.append({
                "algorithm": algo_name, "n_clients": n_clients,
                "total_edges": total_edges,
                "internal_total": internal_total, "cross_total": cross_total,
                "pct_preserved": internal_total / total_edges * 100,
                "attack_preserved": internal_atk, "attack_lost": cross_atk,
                "attack_pct_preserved": internal_atk / max(n_attack_e, 1) * 100,
                "benign_preserved": internal_ben, "benign_lost": cross_ben,
                "benign_pct_preserved": internal_ben / max(n_benign_e, 1) * 100,
            })

            # per attack-type preservation
            for atype in sorted(set(d.get("attack_type", "Benign")
                                     for _, _, d in G.edges(data=True))):
                t_edges = [(u, v) for u, v, d in G.edges(data=True)
                           if d.get("attack_type") == atype]
                preserved = sum(1 for (u, v) in t_edges
                                if node_to_client.get(u) == node_to_client.get(v))
                total = len(t_edges)
                all_attack_type_rows.append({
                    "algorithm": algo_name, "n_clients": n_clients,
                    "attack_type": atype, "total_edges": total,
                    "preserved": preserved, "lost": total - preserved,
                    "pct_preserved": preserved / max(total, 1) * 100,
                })

            # ── summary row for strategy-level comparison ──
            sp_vals, pe_vals, mae_vals = [], [], []
            for co in all_corr_rows[-n_clients:]:
                for m in METRICS:
                    if not np.isnan(co.get(f"{m}_spearman", np.nan)):
                        sp_vals.append(co[f"{m}_spearman"])
                    if not np.isnan(co.get(f"{m}_pearson", np.nan)):
                        pe_vals.append(co[f"{m}_pearson"])
            for ch in all_change_rows[-n_clients:]:
                for m in METRICS:
                    mae_vals.append(ch[f"{m}_mae"])
            fed_time = sum(sum(client_timings[cid].values())
                           for cid in client_timings)
            summary_entry = {
                "algorithm": algo_name, "n_clients": n_clients,
                "avg_spearman": np.mean(sp_vals) if sp_vals else 0,
                "avg_pearson":  np.mean(pe_vals) if pe_vals else 0,
                "avg_mae":      np.mean(mae_vals) if mae_vals else 0,
                "edge_pct_preserved":   internal_total / total_edges * 100,
                "attack_pct_preserved": internal_atk / max(n_attack_e, 1) * 100,
                "fed_time":       fed_time,
                "global_time":    global_total_time,
                "detection_time": meta["detection_time"],
                "modularity":     meta["modularity"],
                "n_communities":  meta["n_communities"],
            }
            all_summary.append(summary_entry)

    # Algorithm metadata (global, independent of client count)
    for name, meta in algo_result.items():
        algo_meta_rows.append({
            "algorithm": name,
            "detection_time": meta["detection_time"],
            "modularity": meta["modularity"],
            "n_communities": meta["n_communities"],
            "largest_community": meta["sizes"][0] if meta["sizes"] else 0,
            "smallest_community": meta["sizes"][-1] if meta["sizes"] else 0,
            "median_community": int(np.median(meta["sizes"])) if meta["sizes"] else 0,
        })

    # ── DataFrames ─────────────────────────────────────────
    timing_df    = pd.DataFrame(all_timing_rows)
    change_df    = pd.DataFrame(all_change_rows)
    corr_df      = pd.DataFrame(all_corr_rows)
    summary_df   = pd.DataFrame(all_summary)
    edge_df      = pd.DataFrame(all_edge_rows)
    attack_df    = pd.DataFrame(all_attack_type_rows)
    algo_meta_df = pd.DataFrame(algo_meta_rows)

    # ── Save CSVs ──────────────────────────────────────────
    def save(df, name):
        p = os.path.join(output_dir, name)
        df.to_csv(p, index=False, float_format="%.6f")
        print(f"   saved: {name}")
    print("\nSaving CSVs...")
    save(timing_df,    "timing_comparison.csv")
    save(change_df,    "change_detection.csv")
    save(corr_df,      "correlation_per_client.csv")
    save(summary_df,   "summary_by_algo_nclients.csv")
    save(edge_df,      "edge_preservation.csv")
    save(attack_df,    "attack_type_preservation.csv")
    save(algo_meta_df, "algorithm_metadata.csv")

    # ── Tables ─────────────────────────────────────────────
    print_summary_table(summary_df)
    print_algo_meta_table(algo_meta_df)

    # ── Visualisations ─────────────────────────────────────
    draw_algo_overview(algo_meta_df, output_dir)
    draw_spearman_heatmap(corr_df, output_dir, n_clients=MAIN_N_CLIENTS)
    draw_pearson_heatmap(corr_df, output_dir, n_clients=MAIN_N_CLIENTS)
    draw_change_heatmap(change_df, output_dir, n_clients=MAIN_N_CLIENTS)
    draw_timing_chart(timing_df, output_dir, n_clients=MAIN_N_CLIENTS)
    draw_per_metric_timing_breakdown(timing_df, output_dir, n_clients=MAIN_N_CLIENTS)
    draw_edge_preservation(edge_df, output_dir, n_clients=MAIN_N_CLIENTS)
    draw_attack_type_preservation(attack_df, output_dir, n_clients=MAIN_N_CLIENTS)
    draw_scaling_spearman(summary_df, output_dir)
    draw_scaling_edge(summary_df, output_dir)
    draw_tradeoff_summary(summary_df, edge_df, output_dir, n_clients=MAIN_N_CLIENTS)
    draw_edge_cut_vs_distortion(edge_df, change_df, output_dir)


    return {"timing": timing_df, "change": change_df, "corr": corr_df,
            "summary": summary_df, "edges": edge_df,
            "attacks": attack_df, "meta": algo_meta_df}


# ══════════════════════════════════════════════════════════════
# PRINT TABLES
# ══════════════════════════════════════════════════════════════

def print_summary_table(df):
    print(f"\n{'='*110}")
    print("  SUMMARY: algorithm × #clients")
    print(f"{'='*110}")
    print(f"{'Algorithm':<20} {'#Cli':>5} {'Spearman':>10} {'Pearson':>9} "
          f"{'MAE':>10} {'Edge%':>8} {'Atk%':>8} {'FedT':>8} {'GlobalT':>9} {'Mod':>8}")
    print("-" * 110)
    for _, r in df.iterrows():
        print(f"{r['algorithm']:<20} {int(r['n_clients']):>5} "
              f"{r['avg_spearman']:>10.3f} {r['avg_pearson']:>9.3f} "
              f"{r['avg_mae']:>10.5f} "
              f"{r['edge_pct_preserved']:>7.1f}% "
              f"{r['attack_pct_preserved']:>7.1f}% "
              f"{r['fed_time']:>7.2f}s {r['global_time']:>8.2f}s "
              f"{r['modularity']:>8.3f}")
    print("=" * 110)


def print_algo_meta_table(df):
    print(f"\n{'='*90}")
    print("  ALGORITHM METADATA (runs once per algorithm)")
    print(f"{'='*90}")
    print(f"{'Algorithm':<20} {'DetectTime':>11} {'#Comm':>7} "
          f"{'Modularity':>11} {'Largest':>8} {'Median':>7} {'Smallest':>9}")
    print("-" * 90)
    for _, r in df.iterrows():
        print(f"{r['algorithm']:<20} {r['detection_time']:>10.2f}s "
              f"{int(r['n_communities']):>7} {r['modularity']:>11.4f} "
              f"{int(r['largest_community']):>8} {int(r['median_community']):>7} "
              f"{int(r['smallest_community']):>9}")
    print("=" * 90)


# ══════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════

def _algo_list_from(df):
    """Get the set of algorithms actually present in the data, in canonical order."""
    canonical = list(ALGO_COLORS.keys())
    present = set(df["algorithm"].unique())
    return [a for a in canonical if a in present]


def draw_algo_overview(meta_df, output_dir):
    print("\nDrawing algorithm overview (modularity, #comm, detection time)...")
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor=BG)
    fig.suptitle("Community-detection algorithms — structural overview",
                 color=TEXT, fontsize=16, fontweight="bold", y=1.02)

    algos = meta_df["algorithm"].tolist()
    colors = [ALGO_COLORS[a] for a in algos]

    # 1) modularity
    ax = axes[0]; style_ax(ax, "Modularity  (higher = stronger community structure)")
    bars = ax.bar(algos, meta_df["modularity"], color=colors, alpha=0.9)
    for b, v in zip(bars, meta_df["modularity"]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f"{v:.3f}", ha="center", va="bottom", color=TEXT,
                fontsize=10, fontweight="bold")
    ax.set_ylabel("Modularity", color=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=TEXT, labelsize=9, rotation=15)

    # 2) number of communities
    ax = axes[1]; style_ax(ax, "Number of communities detected")
    bars = ax.bar(algos, meta_df["n_communities"], color=colors, alpha=0.9)
    for b, v in zip(bars, meta_df["n_communities"]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                f"{int(v)}", ha="center", va="bottom", color=TEXT,
                fontsize=10, fontweight="bold")
    ax.set_ylabel("# communities", color=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    force_integer_yaxis(ax)
    ax.tick_params(axis="x", colors=TEXT, labelsize=9, rotation=15)

    # 3) detection time
    ax = axes[2]; style_ax(ax, "Community-detection time")
    bars = ax.bar(algos, meta_df["detection_time"], color=colors, alpha=0.9)
    for b, v in zip(bars, meta_df["detection_time"]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                f"{v:.2f}s", ha="center", va="bottom", color=TEXT,
                fontsize=10, fontweight="bold")
    ax.set_ylabel("Seconds", color=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=TEXT, labelsize=9, rotation=15)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(output_dir, "1_algorithm_overview.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"   saved: {out}")


def _heatmap_panel(ax, matrix, row_labels, col_labels, title,
                   cmap, vmin, vmax, cell_text=None, fmt="{:.3f}"):
    """
    cell_text: optional 2D list of strings to display instead of numeric values.
               If None, `fmt` is applied to `matrix` values.
    """
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, color=TEXT, fontsize=9,
                       rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, color=TEXT, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if cell_text is not None:
                txt = cell_text[i][j]
            else:
                txt = "N/A" if np.isnan(v) else fmt.format(v)
            ax.text(j, i, txt, ha="center", va="center",
                    color="#000000", fontsize=8, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.65,
                              edgecolor="none", pad=1))
    ax.set_facecolor(PANEL)
    return im


def draw_spearman_heatmap(corr_df, output_dir, n_clients):
    print(f"\nDrawing Spearman heatmap (n={n_clients})...")
    sub_all = corr_df[corr_df["n_clients"] == n_clients]
    algos = _algo_list_from(sub_all)
    n_algos = len(algos)

    fig, axes = plt.subplots(1, n_algos, figsize=(7 * n_algos, 7), facecolor=BG)
    if n_algos == 1:
        axes = [axes]
    fig.suptitle(f"Spearman rank correlation — per client × centrality  (n_clients={n_clients})",
                 color=TEXT, fontsize=15, fontweight="bold", y=1.02)

    im = None
    for idx, algo in enumerate(algos):
        ax = axes[idx]
        sub = sub_all[sub_all["algorithm"] == algo]
        rows, labels = [], []
        for _, r in sub.iterrows():
            labels.append(r["client"])
            rows.append([r.get(f"{m}_spearman", np.nan) for m in METRICS])
        mat = np.array(rows) if rows else np.zeros((1, len(METRICS)))
        if not rows:
            labels = ["(empty)"]
        im = _heatmap_panel(ax, mat, labels,
                            [METRIC_LABELS[m] for m in METRICS],
                            algo, cmap="RdYlGn", vmin=-0.4, vmax=1.0)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.22, wspace=0.3)
    cax = fig.add_axes([0.3, 0.05, 0.4, 0.025])
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal", label="Spearman ρ")
    cbar.ax.tick_params(colors=MUTED)
    out = os.path.join(output_dir, f"2_spearman_heatmap_n{n_clients}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"   saved: {out}")


def draw_pearson_heatmap(corr_df, output_dir, n_clients):
    print(f"\nDrawing Pearson heatmap (n={n_clients})...")
    sub_all = corr_df[corr_df["n_clients"] == n_clients]
    algos = _algo_list_from(sub_all)
    n_algos = len(algos)

    fig, axes = plt.subplots(1, n_algos, figsize=(7 * n_algos, 7), facecolor=BG)
    if n_algos == 1:
        axes = [axes]
    fig.suptitle(f"Pearson correlation — per client × centrality  (n_clients={n_clients})",
                 color=TEXT, fontsize=15, fontweight="bold", y=1.02)

    im = None
    for idx, algo in enumerate(algos):
        ax = axes[idx]
        sub = sub_all[sub_all["algorithm"] == algo]
        rows, labels = [], []
        for _, r in sub.iterrows():
            labels.append(r["client"])
            rows.append([r.get(f"{m}_pearson", np.nan) for m in METRICS])
        mat = np.array(rows) if rows else np.zeros((1, len(METRICS)))
        if not rows:
            labels = ["(empty)"]
        im = _heatmap_panel(ax, mat, labels,
                            [METRIC_LABELS[m] for m in METRICS],
                            algo, cmap="RdYlGn", vmin=-0.2, vmax=1.0)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.22, wspace=0.3)
    cax = fig.add_axes([0.3, 0.05, 0.4, 0.025])
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal", label="Pearson r")
    cbar.ax.tick_params(colors=MUTED)
    out = os.path.join(output_dir, f"3_pearson_heatmap_n{n_clients}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"   saved: {out}")


def draw_change_heatmap(change_df, output_dir, n_clients):
    """Each cell shows '<n_changed>  (<pct>%)' — node count AND percentage."""
    print(f"\nDrawing change-detection heatmap (n={n_clients})...")
    sub_all = change_df[change_df["n_clients"] == n_clients]
    algos = _algo_list_from(sub_all)
    n_algos = len(algos)

    fig, axes = plt.subplots(1, n_algos, figsize=(7 * n_algos, 7), facecolor=BG)
    if n_algos == 1:
        axes = [axes]
    fig.suptitle(f"Nodes whose centrality changed  (|diff| > 0.001)  —  n_clients={n_clients}",
                 color=TEXT, fontsize=15, fontweight="bold", y=1.02)

    im = None
    for idx, algo in enumerate(algos):
        ax = axes[idx]
        sub = sub_all[sub_all["algorithm"] == algo]
        rows, labels, text_rows = [], [], []
        for _, r in sub.iterrows():
            labels.append(f"{r['client']}\n(n={int(r['n_nodes'])})")
            pct_row = [r[f"{m}_pct_changed"] for m in METRICS]
            txt_row = [f"{int(r[f'{m}_changed'])}\n({r[f'{m}_pct_changed']:.1f}%)"
                       for m in METRICS]
            rows.append(pct_row); text_rows.append(txt_row)

        mat = np.array(rows) if rows else np.zeros((1, len(METRICS)))
        if not rows:
            labels = ["(empty)"]; text_rows = [["" for _ in METRICS]]

        im = _heatmap_panel(ax, mat, labels,
                            [METRIC_LABELS[m] for m in METRICS],
                            algo, cmap="RdYlBu_r", vmin=0, vmax=100,
                            cell_text=text_rows)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.22, wspace=0.3)
    cax = fig.add_axes([0.3, 0.05, 0.4, 0.025])
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal", label="% of nodes changed")
    cbar.ax.tick_params(colors=MUTED)
    out = os.path.join(output_dir, f"4_change_heatmap_n{n_clients}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"   saved: {out}")


def draw_timing_chart(timing_df, output_dir, n_clients):
    print(f"\nDrawing timing chart (n={n_clients})...")
    fig, ax = plt.subplots(1, 1, figsize=(20, 8), facecolor=BG)
    fig.suptitle(f"Computation time — centralized vs per-client  (n_clients={n_clients})",
                 color=TEXT, fontsize=16, fontweight="bold", y=0.98)

    sub = timing_df[((timing_df["client"] == "Global") |
                     (timing_df["n_clients"] == n_clients))].copy()

    style_ax(ax, "Time to compute the 7 centralities on each subgraph")
    labels, vals, colors = [], [], []
    for _, r in sub.iterrows():
        if r["client"] == "Global":
            labels.append("Centralized\nGlobal")
            colors.append("#333333")
        else:
            labels.append(f"{r['algorithm']}\n{r['client']}")
            colors.append(ALGO_COLORS.get(r["algorithm"], "#888"))
        vals.append(r["time_total"])

    bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, color=TEXT, rotation=45, ha="right")
    ax.set_ylabel("Time (seconds)", color=MUTED, fontsize=11)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.01,
                f"{v:.3f}s", ha="center", va="bottom", color=TEXT,
                fontsize=8, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(output_dir, f"5_timing_n{n_clients}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"   saved: {out}")


def draw_per_metric_timing_breakdown(timing_df, output_dir, n_clients):
    """
    Stacked bar chart, one bar per (algorithm × client), each segment is
    the time spent on one centrality. Includes a dashed line for the
    centralized (global) total and a small per-metric breakdown of global
    on the side for reference.
    """
    print(f"\nDrawing per-metric timing breakdown (n={n_clients})...")

    sub = timing_df[timing_df["n_clients"] == n_clients].copy()
    g_row = timing_df[timing_df["client"] == "Global"]
    if g_row.empty:
        print("   no global row found, skipping")
        return
    g_total = float(g_row["time_total"].iloc[0])
    g_per_metric = {m: float(g_row[f"time_{m}"].iloc[0]) for m in METRICS}

    if sub.empty:
        print("   no per-client rows for this n_clients, skipping")
        return

    # Order: by algorithm (canonical order), then by client number
    algo_order = [a for a in ALGO_COLORS.keys()
                  if a in sub["algorithm"].unique()]
    sub["algo_idx"] = sub["algorithm"].map({a: i for i, a in enumerate(algo_order)})
    sub["client_num"] = sub["client"].str.extract(r"(\d+)").astype(int)
    sub = sub.sort_values(["algo_idx", "client_num"]).reset_index(drop=True)

    n_bars = len(sub)
    # Width scales with number of bars; min 14, max 32
    fig_w = max(14, min(32, 1.1 * n_bars + 4))
    fig, ax = plt.subplots(figsize=(fig_w, 8), facecolor=BG)
    fig.suptitle(
        f"Per-metric time breakdown — community-detection algorithms  "
        f"(n_clients={n_clients})",
        color=TEXT, fontsize=15, fontweight="bold", y=0.99,
    )
    style_ax(ax, "Stacked: time per centrality, per (algorithm × client)")

    x = np.arange(n_bars)
    bar_w = 0.7

    # Stack each metric on top of the previous
    bottoms = np.zeros(n_bars)
    for m in METRICS:
        heights = sub[f"time_{m}"].values
        ax.bar(x, heights, bar_w, bottom=bottoms,
               color=METRIC_COLORS[m], alpha=0.92,
               edgecolor="white", linewidth=0.5,
               label=METRIC_LABELS[m])
        bottoms = bottoms + heights

    # Total label on top of each bar
    totals = sub["time_total"].values
    y_max = max(totals.max(), g_total) * 1.18
    for i, t in enumerate(totals):
        ax.text(x[i], t + y_max * 0.008, f"{t:.2f}s",
                ha="center", va="bottom", color=TEXT,
                fontsize=8, fontweight="bold")

    # Algorithm group labels and separators
    # Find segment boundaries between algorithms
    algo_segments = []
    cur_algo = sub.iloc[0]["algorithm"]
    seg_start = 0
    for i in range(1, n_bars):
        if sub.iloc[i]["algorithm"] != cur_algo:
            algo_segments.append((cur_algo, seg_start, i - 1))
            seg_start = i
            cur_algo = sub.iloc[i]["algorithm"]
    algo_segments.append((cur_algo, seg_start, n_bars - 1))

    # Vertical separators between algorithm groups
    for i in range(len(algo_segments) - 1):
        sep = algo_segments[i][2] + 0.5
        ax.axvline(x=sep, color="#BBBBBB", linewidth=0.8,
                   linestyle=":", alpha=0.7, zorder=0)

    # Algorithm names below the x-axis tick labels
    for algo, start, end in algo_segments:
        center = (start + end) / 2
        ax.text(center, -y_max * 0.18, algo,
                ha="center", va="top",
                color=ALGO_COLORS[algo], fontsize=11, fontweight="bold",
                transform=ax.transData,
                clip_on=False)

    # X-tick labels: just the client portion ("Client 1", etc.)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["client"].tolist(),
                       fontsize=8, color=TEXT, rotation=45, ha="right")
    ax.set_ylabel("Time (seconds)", color=MUTED, fontsize=11)
    ax.set_xlim(-0.6, n_bars - 0.4)
    ax.set_ylim(0, y_max)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)

    # Centralized baseline reference line
    ax.axhline(y=g_total, color="#333333", linestyle="--",
               alpha=0.7, linewidth=1.5, zorder=0)
    ax.text(n_bars - 0.5, g_total * 1.02,
            f"Centralized (Global): {g_total:.2f}s",
            color="#333333", fontsize=9, ha="right", fontweight="bold")

    # Legend (centralities)
    ax.legend(loc="upper left", fontsize=9, ncol=1,
              facecolor=BG, labelcolor=TEXT, edgecolor=GRID,
              title="Centrality", title_fontsize=10,
              framealpha=0.95)

    # Footer line documenting the "Global" reference
    metric_breakdown = "  ".join(
        f"{METRIC_LABELS[m]}: {g_per_metric[m]:.2f}s" for m in METRICS
    )
    fig.text(0.5, 0.01,
             f"Centralized per-metric breakdown — {metric_breakdown}",
             ha="center", va="bottom", color=MUTED,
             fontsize=8.5, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = os.path.join(output_dir, f"6_timing_per_metric_n{n_clients}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"   saved: {out}")


def draw_edge_preservation(edge_df, output_dir, n_clients):
    print(f"\nDrawing edge preservation (n={n_clients})...")
    sub = edge_df[edge_df["n_clients"] == n_clients].reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor=BG)
    fig.suptitle(f"Edge preservation by algorithm  (n_clients={n_clients})",
                 color=TEXT, fontsize=16, fontweight="bold", y=1.02)

    strats = sub["algorithm"].tolist()
    x = np.arange(len(strats)); w = 0.35

    ax = axes[0]; style_ax(ax, "Total edges: preserved vs lost")
    ax.bar(x, sub["internal_total"], w, color=BENIGN, alpha=0.85, label="Preserved (internal)")
    ax.bar(x, sub["cross_total"],    w, color=ATTACK, alpha=0.85,
           bottom=sub["internal_total"], label="Lost (cross)")
    ax.set_xticks(x); ax.set_xticklabels(strats, color=TEXT, fontsize=9, rotation=15)
    ax.set_ylabel("Edge count", color=MUTED)
    ax.legend(fontsize=9, facecolor=BG, labelcolor=TEXT, edgecolor=GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.text(i, r["internal_total"]+r["cross_total"]+10,
                f"{r['pct_preserved']:.1f}%\nkept",
                ha="center", va="bottom", color=BENIGN,
                fontsize=10, fontweight="bold")

    ax2 = axes[1]; style_ax(ax2, "Attack edges: preserved vs lost")
    ax2.bar(x, sub["attack_preserved"], w, color="#F57F17", alpha=0.85, label="Attack preserved")
    ax2.bar(x, sub["attack_lost"],      w, color=ATTACK,   alpha=0.85,
            bottom=sub["attack_preserved"], label="Attack lost")
    ax2.set_xticks(x); ax2.set_xticklabels(strats, color=TEXT, fontsize=9, rotation=15)
    ax2.set_ylabel("Attack edge count", color=MUTED)
    ax2.legend(fontsize=9, facecolor=BG, labelcolor=TEXT, edgecolor=GRID)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.6); ax2.set_axisbelow(True)
    for i, (_, r) in enumerate(sub.iterrows()):
        total_atk = r["attack_preserved"] + r["attack_lost"]
        ax2.text(i, total_atk+5, f"{r['attack_pct_preserved']:.1f}%",
                 ha="center", va="bottom", color="#F57F17",
                 fontsize=11, fontweight="bold")

    ax3 = axes[2]; style_ax(ax3, "Preservation rate: attack vs benign")
    ax3.bar(x-w/2, sub["attack_pct_preserved"], w, color=ATTACK, alpha=0.85, label="Attack %")
    ax3.bar(x+w/2, sub["benign_pct_preserved"], w, color=BENIGN, alpha=0.85, label="Benign %")
    ax3.set_xticks(x); ax3.set_xticklabels(strats, color=TEXT, fontsize=9, rotation=15)
    ax3.set_ylabel("% preserved", color=MUTED); ax3.set_ylim(0, 110)
    ax3.legend(fontsize=9, facecolor=BG, labelcolor=TEXT, edgecolor=GRID)
    ax3.yaxis.grid(True, color=GRID, linewidth=0.6); ax3.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(output_dir, f"8_edge_preservation_n{n_clients}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"   saved: {out}")


def draw_attack_type_preservation(attack_df, output_dir, n_clients):
    print(f"\nDrawing per-attack-type preservation (n={n_clients})...")
    sub_all = attack_df[(attack_df["n_clients"] == n_clients) &
                         (attack_df["attack_type"] != "Benign")]
    algos = _algo_list_from(sub_all)
    n_algos = len(algos)

    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 8), facecolor=BG)
    if n_algos == 1:
        axes = [axes]
    fig.suptitle(f"Attack-type preservation by algorithm  (n_clients={n_clients})",
                 color=TEXT, fontsize=15, fontweight="bold", y=1.02)

    for idx, algo in enumerate(algos):
        ax = axes[idx]; style_ax(ax, algo)
        sub = sub_all[sub_all["algorithm"] == algo].sort_values("total_edges",
                                                                 ascending=True)
        y = np.arange(len(sub))
        ax.barh(y, sub["pct_preserved"], color=ALGO_COLORS[algo], alpha=0.85)
        ax.set_yticks(y); ax.set_yticklabels(sub["attack_type"],
                                              color=TEXT, fontsize=9)
        ax.set_xlabel("% edges preserved", color=MUTED)
        ax.set_xlim(0, 110)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        for i, (_, r) in enumerate(sub.iterrows()):
            ax.text(r["pct_preserved"]+1, i,
                    f"{int(r['preserved'])}/{int(r['total_edges'])}",
                    va="center", color=TEXT, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(output_dir, f"9_attack_type_n{n_clients}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"   saved: {out}")


def draw_scaling_spearman(summary_df, output_dir):
    print("\nDrawing scaling chart (Spearman & Pearson vs #clients)...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), facecolor=BG)
    fig.suptitle("Scalability: correlation vs number of clients",
                 color=TEXT, fontsize=16, fontweight="bold", y=1.02)

    ncs = sorted(summary_df["n_clients"].unique())
    algos = _algo_list_from(summary_df)

    for ax_idx, (col, title, ylabel) in enumerate([
        ("avg_spearman", "Average Spearman rank correlation",
         "Spearman ρ  (higher = better)"),
        ("avg_pearson", "Average Pearson correlation",
         "Pearson r  (higher = better)"),
    ]):
        ax = axes[ax_idx]; style_ax(ax, title)
        for algo in algos:
            vals = []
            for nc in ncs:
                row = summary_df[(summary_df["algorithm"] == algo) &
                                  (summary_df["n_clients"] == nc)]
                vals.append(row[col].values[0] if len(row) else np.nan)
            ax.plot(ncs, vals, marker="o", linewidth=2.5, markersize=10,
                    color=ALGO_COLORS[algo], label=algo, alpha=0.9)
            for nc, v in zip(ncs, vals):
                if not np.isnan(v):
                    ax.text(nc, v + 0.02, f"{v:.3f}", ha="center",
                            color=TEXT, fontsize=8, fontweight="bold")
        ax.set_xlabel("Number of clients", color=MUTED, fontsize=11)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=11)
        ax.set_xticks(ncs); ax.set_ylim(-0.1, 1.15)
        ax.axhline(y=0, color="#999999", linewidth=0.5)
        ax.legend(fontsize=9, facecolor=BG, labelcolor=TEXT, edgecolor=GRID)
        ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(output_dir, "10_scaling_correlation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"   saved: {out}")


def draw_scaling_edge(summary_df, output_dir):
    print("\nDrawing scaling chart (edge preservation vs #clients)...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), facecolor=BG)
    fig.suptitle("Scalability: edge preservation vs number of clients",
                 color=TEXT, fontsize=16, fontweight="bold", y=1.02)

    ncs = sorted(summary_df["n_clients"].unique())
    algos = _algo_list_from(summary_df)

    for ax_idx, (col, title, ylabel) in enumerate([
        ("edge_pct_preserved",  "Total edge preservation",    "% edges preserved"),
        ("attack_pct_preserved","Attack edge preservation",   "% attack edges preserved"),
    ]):
        ax = axes[ax_idx]; style_ax(ax, title)
        for algo in algos:
            vals = []
            for nc in ncs:
                row = summary_df[(summary_df["algorithm"] == algo) &
                                  (summary_df["n_clients"] == nc)]
                vals.append(row[col].values[0] if len(row) else np.nan)
            ax.plot(ncs, vals, marker="s", linewidth=2.5, markersize=10,
                    color=ALGO_COLORS[algo], label=algo, alpha=0.9)
            for nc, v in zip(ncs, vals):
                if not np.isnan(v):
                    ax.text(nc, v + 1.5, f"{v:.1f}%", ha="center",
                            color=TEXT, fontsize=8, fontweight="bold")
        ax.set_xlabel("Number of clients", color=MUTED, fontsize=11)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=11)
        ax.set_xticks(ncs); ax.set_ylim(0, 110)
        ax.legend(fontsize=9, facecolor=BG, labelcolor=TEXT, edgecolor=GRID)
        ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(output_dir, "11_scaling_edges.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"   saved: {out}")


def draw_tradeoff_summary(summary_df, edge_df, output_dir, n_clients):
    print(f"\nDrawing trade-off summary (n={n_clients})...")
    fig, axes = plt.subplots(1, 3, figsize=(26, 8), facecolor=BG)
    fig.suptitle(f"Algorithm trade-off summary  (n_clients={n_clients})",
                 color=TEXT, fontsize=16, fontweight="bold", y=1.02)

    sub = summary_df[summary_df["n_clients"] == n_clients]
    algos = sub["algorithm"].tolist()
    colors = [ALGO_COLORS[a] for a in algos]

    # 1) Spearman
    ax = axes[0]; style_ax(ax, "Rank correlation (Spearman)")
    bars = ax.bar(algos, sub["avg_spearman"], color=colors, alpha=0.85)
    for b, v in zip(bars, sub["avg_spearman"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", color=TEXT,
                fontsize=10, fontweight="bold")
    ax.set_ylabel("Spearman ρ (higher = better)", color=MUTED)
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", labelsize=9, colors=TEXT, rotation=20)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)

    # 2) Attack-edge preservation
    ax = axes[1]; style_ax(ax, "Attack-edge preservation")
    bars = ax.bar(algos, sub["attack_pct_preserved"], color=colors, alpha=0.85)
    for b, v in zip(bars, sub["attack_pct_preserved"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{v:.1f}%", ha="center", va="bottom", color=TEXT,
                fontsize=10, fontweight="bold")
    ax.set_ylabel("% attack edges preserved", color=MUTED)
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", labelsize=9, colors=TEXT, rotation=20)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)

    # 3) TOTAL COST — side-by-side grouped bars
    ax = axes[2]; style_ax(ax, "Time cost:  detection  vs  federated centralities")

    detect = sub["detection_time"].values
    fed    = sub["fed_time"].values
    g_time = sub["global_time"].iloc[0]

    x = np.arange(len(algos))
    w = 0.35

    bars_d = ax.bar(x - w / 2, detect, w, color=colors, alpha=0.50,
                    label="Community detection",
                    edgecolor=[ALGO_COLORS[a] for a in algos], linewidth=1.2)
    bars_f = ax.bar(x + w / 2, fed,    w, color=colors, alpha=0.95,
                    label="Federated centralities")

    ax.axhline(y=g_time, color="#333333", linestyle="--",
               alpha=0.7, linewidth=1.5, zorder=0)
    ax.text(len(algos) - 0.5, g_time * 1.05,
            f"Centralized (baseline): {g_time:.2f}s",
            color="#333333", fontsize=9, ha="right", fontweight="bold")

    max_val = max(max(detect), max(fed), g_time) * 1.15
    for i, (d, f) in enumerate(zip(detect, fed)):
        ax.text(x[i] - w / 2, d + max_val * 0.008,
                f"{d:.2f}s", ha="center", va="bottom",
                color=TEXT, fontsize=8, fontweight="bold")
        ax.text(x[i] + w / 2, f + max_val * 0.008,
                f"{f:.2f}s", ha="center", va="bottom",
                color=TEXT, fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(algos, color=TEXT, fontsize=9, rotation=20)
    ax.set_ylabel("Time (seconds)", color=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=9, facecolor=BG, labelcolor=TEXT, edgecolor=GRID,
              loc="upper left")

    ax.text(0.5, -0.22,
            "Left bar = one-time community-detection cost.  "
            "Right bar = cost of computing all 7 centralities across the clients.",
            transform=ax.transAxes, ha="center", va="top",
            color=MUTED, fontsize=9, style="italic")

    plt.tight_layout()
    out = os.path.join(output_dir, f"12_tradeoff_summary_n{n_clients}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"   saved: {out}")

def draw_edge_cut_vs_distortion(edge_df, change_df, output_dir):
    """
    Scatter plot per centrality: x = % of edges cut, y = MAE distortion.
    One dot per (algorithm × n_clients) experiment.
    Fits a linear regression and prints slope + R^2 in the corner.
    """
    print("\nDrawing edge-cuts vs centrality-distortion correlation...")

    # ── 1. Aggregate per-client MAE up to (algorithm × n_clients) ──
    agg_rows = []
    for (algo, ncl), grp in change_df.groupby(["algorithm", "n_clients"]):
        row = {"algorithm": algo, "n_clients": ncl}
        for m in METRICS:
            row[f"{m}_mae"] = grp[f"{m}_mae"].mean()
        agg_rows.append(row)
    mae_agg = pd.DataFrame(agg_rows)

    # ── 2. Compute % edges cut from the edge_df ──────────────────
    edge_cut = edge_df[["algorithm", "n_clients", "pct_preserved"]].copy()
    edge_cut["pct_cut"] = 100.0 - edge_cut["pct_preserved"]

    # ── 3. Merge the two on (algorithm, n_clients) ───────────────
    merged = edge_cut.merge(mae_agg, on=["algorithm", "n_clients"])

    # ── 4. Save the merged table for the report ──────────────────
    merged.to_csv(os.path.join(output_dir, "edge_cut_vs_distortion.csv"),
                  index=False, float_format="%.6f")
    print(f"   saved: edge_cut_vs_distortion.csv")

    # ── 5. Draw one scatter per centrality ───────────────────────
    n_metrics = len(METRICS)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows),
                             facecolor=BG)
    fig.suptitle("Edge cuts vs. centrality distortion  "
                 "(one dot = one algorithm × #clients experiment)",
                 color=TEXT, fontsize=15, fontweight="bold", y=1.00)

    axes_flat = axes.flatten() if n_rows > 1 else axes

    for idx, m in enumerate(METRICS):
        ax = axes_flat[idx]
        style_ax(ax, METRIC_LABELS[m])

        x = merged["pct_cut"].values
        y = merged[f"{m}_mae"].values

        # colour each dot by algorithm
        colors = [ALGO_COLORS.get(a, "#888") for a in merged["algorithm"]]
        ax.scatter(x, y, c=colors, s=60, alpha=0.85,
                   edgecolors="white", linewidths=1.0, zorder=3)

        # fit a line — only if there is variation on x
        if len(x) >= 2 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 50)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, "--", color="#333333",
                    linewidth=1.5, alpha=0.7, zorder=2)

            # R^2 — how well the line fits the dots
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            ax.text(0.04, 0.96,
                    f"slope = {slope:.2e}\nR² = {r2:.3f}",
                    transform=ax.transAxes, ha="left", va="top",
                    color=TEXT, fontsize=9, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor=GRID,
                              alpha=0.9, pad=4))

        ax.set_xlabel("% of edges cut", color=MUTED, fontsize=9)
        ax.set_ylabel("MAE distortion", color=MUTED, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)

    # hide any unused subplot cells
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(output_dir, "13_edge_cut_vs_distortion.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"   saved: {out}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = load_csv(CSV_PATH)
    G  = build_graph(df)
    results = run_all(G, OUTPUT_DIR, CLIENT_COUNTS)
    print(f"\nDetailed algorithm comparison complete! Check: {OUTPUT_DIR}")