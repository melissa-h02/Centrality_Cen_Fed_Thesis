"""
---------------------------------------------------------------
PARTITIONING STRATEGY COMPARISON — multi-client × multi-dataset
---------------------------------------------------------------
  Compares 4 client-partitioning strategies:
      • Random          — shuffle nodes, split evenly
      • Degree-balanced — round-robin nodes by descending degree
      • Stratified      — keep attack/benign ratio balanced per client
      • Community       — using louvain

  For EACH strategy at 3 / 5 / 7 clients:
      • 7 centralities computed centrally (once) and per-client
      • Spearman & Pearson correlations vs centralized values
      • Change detection (# nodes whose centrality shifted)
      • Edge preservation (total + per attack-type)
      • Computation timing per metric

  Output schema is harmonised with community_algorithms_detailed.py so the
  same dashboard can read both, the only difference is that the
  "algorithm" column here contains strategy names instead.

  Switch datasets via ACTIVE_DATASET. Outputs go to
  output_strategies_<dataset>/ — one folder per dataset.

  ── SAMPLING ─────────────────────────────────────────────────────
  For large datasets like CIC-ToN-IoT (3 GB), we sample. Two modes:
    • SAMPLE_SIZE = None       → load the full file (no sampling) for ex for the NF-ton dataset
    • SAMPLE_SIZE = 200_000     → stratified sample preserving the full
                                  file's attack-class distribution
  Stratified sampling avoids the first-N-rows trap, where the first
  200,000 rows of CIC-ToN-IoT happen to be 99% benign that would
  bias every downstream result.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import time
import random
import warnings
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# DATASET CONFIGURATION  (mirrors community_algorithms_detailed.py)
# ══════════════════════════════════════════════════════════════
DATASET_CONFIGS = {
    "NF-ToN-IoT": {
        "path":            r"NF-ToN-IoT.csv",
        "src_col":         "IPV4_SRC_ADDR",
        "dst_col":         "IPV4_DST_ADDR",
        "label_col":       "Label",
        "attack_type_col": "Attack",
        "bytes_col":       "IN_BYTES",
    },
    "CIC-ToN-IoT": {
        "path":            r"CIC-ToN-IoT.csv",
        "src_col":         "Src IP",
        "dst_col":         "Dst IP",
        "label_col":       "Label",
        "attack_type_col": "Attack",
        "bytes_col":       "TotLen Fwd Pkts",
    },
}

ACTIVE_DATASET = "NF-ToN-IoT"      # Phase 2 primary dataset

# ── Sampling configuration ────────────────────────────────────
# Set to None to load the full file. For CIC-ToN-IoT, the full
# file is 3 GB and won't fit in memory on most workstations — use
# a stratified sample. 
SAMPLE_SIZE     = None

# Chunk size for reading the CSV when SAMPLE_SIZE is set. 100,000
# rows × 85 columns of strings is roughly 60 MB per chunk in RAM,
# which is fine on any modern machine.
CHUNK_SIZE      = 100_000

# Random seed for reproducibility — same as the partitioning seed
# so results are bit-identical across re-runs.
SAMPLE_SEED     = 42

# ══════════════════════════════════════════════════════════════
# CONFIG (everything else)
# ══════════════════════════════════════════════════════════════
_cfg          = DATASET_CONFIGS[ACTIVE_DATASET]
CSV_PATH      = _cfg["path"]
OUTPUT_DIR    = f"output_strategies_{ACTIVE_DATASET.replace('-', '_')}"
CLIENT_COUNTS = [3, 5, 7]
MAIN_N_CLIENTS = 3

METRICS = ["degree", "out_deg", "in_deg", "betweenness", "closeness", "pagerank", "eigenvector"]

METRIC_LABELS = {
    "degree": "Degree", "out_deg": "Out-degree", "in_deg": "In-degree",
    "betweenness": "Betweenness", "closeness": "Closeness",
    "pagerank": "PageRank", "eigenvector": "Eigenvector",
}

METRIC_COLORS = {
    "degree":      "#1976D2",
    "out_deg":     "#D32F2F",
    "in_deg":      "#F48FB1",
    "betweenness": "#388E3C",
    "closeness":   "#FFA000",
    "pagerank":    "#6D4C41",
    "eigenvector": "#90A4AE",
}

# ── LIGHT Color palette ───────────────────────────────────────
BG     = "#FFFFFF"
PANEL  = "#F5F5F5"
GRID   = "#E0E0E0"
TEXT   = "#1A1A2E"
MUTED  = "#555555"
ATTACK = "#D32F2F"
BENIGN = "#2E7D32"

# Strategy colours — kept distinct from algorithm colours so the dashboard
# can show both views side-by-side without confusion.
STRAT_COLORS = {
    "Random":          "#E53935",   # red
    "Degree-balanced": "#0097A7",   # teal
    "Stratified":      "#7B1FA2",   # purple
    "Community":       "#F57F17",   # orange
}


# ══════════════════════════════════════════════════════════════
# STRATIFIED CSV SAMPLING
# ══════════════════════════════════════════════════════════════
# Two-pass approach:
#   Pass 1 — stream the file once, count rows per attack class.
#   Pass 2 — work out how many rows to take from each class so the
#            sample's class distribution mirrors the full file's, then
#            stream the file again and collect those rows.
# ══════════════════════════════════════════════════════════════

def _count_classes(path, attack_col, chunk_size):
    """Pass 1: count rows per attack class without loading the full file."""
    print(f"   Pass 1/2: counting attack classes (chunk = {chunk_size:,})...")
    counts = {}
    t0 = time.time()
    n_total = 0
    for chunk in pd.read_csv(path, chunksize=chunk_size,
                              usecols=[attack_col], low_memory=False):
        vc = chunk[attack_col].fillna("__MISSING__").value_counts()
        for cls, n in vc.items():
            counts[cls] = counts.get(cls, 0) + int(n)
            n_total += int(n)
    elapsed = time.time() - t0
    print(f"   Pass 1 done: {n_total:,} rows across {len(counts)} classes "
          f"in {elapsed:.1f}s")
    return counts, n_total


def _compute_per_class_targets(class_counts, total_rows, sample_size):
    """Compute how many rows to draw from each attack class so the
    sample preserves the full file's class proportions.

    Uses largest-remainder rounding so the integer per-class counts
    sum exactly to sample_size, even when sample_size * pct is fractional.
    """
    if total_rows <= sample_size:
        # Nothing to sample — use the whole file.
        return {cls: n for cls, n in class_counts.items()}

    # Step 1: proportional floats
    floats = {cls: sample_size * (n / total_rows)
              for cls, n in class_counts.items()}

    # Step 2: floor each, track the fractional remainders
    base = {cls: int(np.floor(v)) for cls, v in floats.items()}
    remainder = sample_size - sum(base.values())

    # Step 3: distribute the remainder to classes with the largest
    # fractional parts. This is the largest-remainder method.
    fracs = sorted(((cls, floats[cls] - base[cls]) for cls in floats),
                   key=lambda x: -x[1])
    for i in range(remainder):
        base[fracs[i % len(fracs)][0]] += 1

    # Step 4: never ask for more than the class actually has
    for cls in base:
        base[cls] = min(base[cls], class_counts[cls])

    return base


def _draw_stratified_sample(path, attack_col, per_class_target,
                             chunk_size, seed):
    """Pass 2: stream the file again, this time collecting rows.

    For each chunk we sample the right number from each class. The
    seed-based RNG is reset for each chunk so the sample is
    reproducible regardless of chunk boundaries.
    """
    print(f"   Pass 2/2: drawing stratified sample (seed = {seed})...")
    targets = dict(per_class_target)        # local mutable copy
    rng = np.random.default_rng(seed)
    collected = []
    t0 = time.time()

    raise NotImplementedError(
        "internal: this function needs class_counts; "
        "call _draw_stratified_sample_v2 instead"
    )


def _draw_stratified_sample_v2(path, attack_col, class_counts,
                                per_class_target, chunk_size, seed):
    """Pass 2: stream the file and draw rows so that
    each class contributes exactly per_class_target[cls] rows in total.

    Method: for each chunk we know what fraction of each class's total
    appears in the chunk; we sample that fraction of the per-class
    target from the chunk. Rounding errors are corrected at the end.
    """
    print(f"   Pass 2/2: drawing stratified sample (seed = {seed})...")
    rng = np.random.default_rng(seed)
    taken_so_far = {cls: 0 for cls in class_counts}
    seen_so_far  = {cls: 0 for cls in class_counts}
    collected = []
    t0 = time.time()
    n_chunks = 0

    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        n_chunks += 1
        # mark the class column to make grouping fast
        cls_series = chunk[attack_col].fillna("__MISSING__")

        for cls, total in class_counts.items():
            target = per_class_target.get(cls, 0)
            if target <= 0:
                continue
            mask = (cls_series == cls).values
            n_in_chunk = int(mask.sum())
            if n_in_chunk == 0:
                continue
            seen_so_far[cls] += n_in_chunk
            # How many of this class should we take from this chunk?
            # Target fraction = (target so far we should have taken) -
            #                   (target already taken)
            # where "target so far we should have taken" is
            # target * (seen_so_far / total). This converges to `target`
            # by the end of the file.
            ideal_taken = int(round(target * seen_so_far[cls] / total))
            need_now = ideal_taken - taken_so_far[cls]
            if need_now <= 0:
                continue
            need_now = min(need_now, n_in_chunk)
            # Pick `need_now` random rows from the matching subset
            chunk_subset = chunk.loc[mask]
            picked = chunk_subset.sample(n=need_now, random_state=int(
                rng.integers(0, 2**31 - 1)))
            collected.append(picked)
            taken_so_far[cls] += need_now

        if n_chunks % 10 == 0:
            total_taken = sum(taken_so_far.values())
            print(f"      ... chunk {n_chunks}, {total_taken:,} rows drawn so far")

    elapsed = time.time() - t0
    df = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()
    # Final shuffle so the order isn't a class-grouped block
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"   Pass 2 done: {len(df):,} rows drawn in {elapsed:.1f}s")
    print("   Per-class breakdown of the sample:")
    for cls in sorted(class_counts.keys(), key=lambda c: -class_counts[c]):
        target = per_class_target.get(cls, 0)
        actual = taken_so_far.get(cls, 0)
        print(f"      {cls:<20} target={target:>6,}  actual={actual:>6,}  "
              f"(full file: {class_counts[cls]:>9,})")
    return df


def load_csv(path, sample_size=None):
    """Load the CSV. If sample_size is set, return a stratified sample
    that preserves the full file's attack-class distribution."""
    print(f"Loading [{ACTIVE_DATASET}]: {path}")

    if sample_size is None:
        df = pd.read_csv(path, low_memory=False)
        print(f"   {df.shape[0]:,} rows x {df.shape[1]} columns (full file)")
    else:
        atk_col = _cfg["attack_type_col"]
        print(f"   Target sample size: {sample_size:,} rows "
              f"(stratified by '{atk_col}')")
        class_counts, total_rows = _count_classes(path, atk_col, CHUNK_SIZE)

        if total_rows <= sample_size:
            print(f"   Full file ({total_rows:,} rows) is already smaller "
                  f"than the target — loading entire file.")
            df = pd.read_csv(path, low_memory=False)
        else:
            print(f"   Full file has {total_rows:,} rows; computing per-class "
                  f"targets to preserve class proportions...")
            targets = _compute_per_class_targets(
                class_counts, total_rows, sample_size)
            df = _draw_stratified_sample_v2(
                path, atk_col, class_counts, targets, CHUNK_SIZE, SAMPLE_SEED)
        print(f"   {df.shape[0]:,} rows x {df.shape[1]} columns (sampled)")

    required = ["src_col", "dst_col", "label_col", "attack_type_col", "bytes_col"]
    missing = [c for c in required if _cfg[c] not in df.columns]
    if missing:
        raise KeyError(
            f"Dataset '{ACTIVE_DATASET}' is missing expected columns: "
            f"{[_cfg[c] for c in missing]}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    return df


def build_graph(df):
    print("\nBuilding network graph...")

    src = _cfg["src_col"]; dst = _cfg["dst_col"]
    lab = _cfg["label_col"]; atk = _cfg["attack_type_col"]
    byt = _cfg["bytes_col"]

    df[src] = df[src].astype(str)
    df[dst] = df[dst].astype(str)

    G = nx.DiGraph()
    attack_ips = set(df[df[lab] == 1][src].unique()) | \
                 set(df[df[lab] == 1][dst].unique())
    all_ips = pd.concat([df[src], df[dst]]).unique()
    for ip in all_ips:
        G.add_node(ip, role="attack" if ip in attack_ips else "benign")

    edge_agg = (
        df.groupby([src, dst, lab])
        .agg(flow_count=(lab, "count"),
             total_bytes=(byt, "sum"),
             attack_type=(atk, lambda x: x.mode()[0]))
        .reset_index()
    )
    for _, row in edge_agg.iterrows():
        G.add_edge(row[src], row[dst],
                   weight=row["flow_count"],
                   total_bytes=row["total_bytes"],
                   label=int(row[lab]),
                   attack_type=row["attack_type"])
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
# PARTITIONING STRATEGIES
# ══════════════════════════════════════════════════════════════

def partition_random(nodes, n, G):
    ns = nodes.copy()
    random.seed(42)
    random.shuffle(ns)
    chunk = len(ns) // n
    parts = []
    for i in range(n):
        start = i * chunk
        end = start + chunk if i < n - 1 else len(ns)
        parts.append(set(ns[start:end]))
    return parts


def partition_degree_balanced(nodes, n, G):
    sorted_nodes = sorted(nodes, key=lambda x: G.degree(x), reverse=True)
    parts = [set() for _ in range(n)]
    for i, node in enumerate(sorted_nodes):
        parts[i % n].add(node)
    return parts


def partition_stratified(nodes, n, G):
    attack = [nd for nd in nodes if G.nodes[nd].get("role") == "attack"]
    benign = [nd for nd in nodes if G.nodes[nd].get("role") != "attack"]
    random.seed(42)
    random.shuffle(attack)
    random.shuffle(benign)
    parts = [set() for _ in range(n)]
    for i, nd in enumerate(attack):
        parts[i % n].add(nd)
    for i, nd in enumerate(benign):
        parts[i % n].add(nd)
    return parts


def partition_community(nodes, n, G, _cache={}):
    """
    Initial Phase-2 community strategy used before the detailed Phase-3
    algorithm comparison: Greedy Modularity on the undirected graph.

    Detected communities are greedily packed into the currently smallest
    client, preserving the early imbalance documented in the thesis.
    """
    cache_key = id(G)
    if cache_key not in _cache:
        U = G.to_undirected()
        comms = sorted(
            nx.community.greedy_modularity_communities(U),
            key=len,
            reverse=True,
        )
        _cache[cache_key] = comms
        print(f"      Found {len(comms)} communities (Greedy Modularity)")
    communities = _cache[cache_key]

    parts = [set() for _ in range(n)]
    sizes = [0] * n
    for comm in communities:
        j = sizes.index(min(sizes))
        parts[j].update(comm)
        sizes[j] += len(comm)
    return parts


STRATEGIES = {
    "Random":          partition_random,
    "Degree-balanced": partition_degree_balanced,
    "Stratified":      partition_stratified,
    "Community":       partition_community,
}


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

    total_edges  = G.number_of_edges()
    n_attack_e   = sum(1 for _, _, d in G.edges(data=True) if d.get("label") == 1)
    n_benign_e   = sum(1 for _, _, d in G.edges(data=True) if d.get("label") == 0)

    # ── 1. Global centralities ─────────────────────────────
    print("\n" + "="*70)
    print("  STEP 1: CENTRALIZED (GLOBAL) CENTRALITIES")
    print("="*70)
    global_cents, global_timings = compute_centralities_timed(G, "Global")
    for m in METRICS:
        print(f"      {m:<14} {global_timings[m]:.4f}s")
    global_total_time = sum(global_timings.values())
    print(f"   TOTAL GLOBAL TIME: {global_total_time:.2f}s")

    # ── 2. Run each (strategy, n_clients) combination ──────
    all_timing_rows = [{"algorithm": "Centralized", "n_clients": 0, "client": "Global",
                        "n_nodes": len(nodes), "n_edges": total_edges,
                        **{f"time_{m}": global_timings[m] for m in METRICS},
                        "time_total": global_total_time}]
    all_change_rows, all_corr_rows = [], []
    all_summary, all_edge_rows, all_attack_type_rows = [], [], []

    for n_clients in client_counts:
        print("\n" + "="*70)
        print(f"  STEP 2: PER-CLIENT ANALYSIS  ({n_clients} clients)")
        print("="*70)

        for strat_name, strat_fn in STRATEGIES.items():
            print(f"\n   --- {strat_name} ({n_clients} clients) ---")
            partitions = strat_fn(nodes, n_clients, G)

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

            # ── per-client centralities ─────────────────────
            client_cents, client_timings = {}, {}
            for i, part in enumerate(partitions):
                cid = i + 1
                sub_edges = [(u, v, d) for u, v, d in G.edges(data=True)
                             if u in part and v in part]
                C = nx.DiGraph()
                C.add_nodes_from([(nd, G.nodes[nd]) for nd in part])
                C.add_edges_from(sub_edges)
                cents_i, t_i = compute_centralities_timed(
                    C, label=f"{strat_name}/c{cid}")
                client_cents[cid], client_timings[cid] = cents_i, t_i

                row = {"algorithm": strat_name, "n_clients": n_clients,
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

                ch_row = {"algorithm": strat_name, "n_clients": n_clients,
                          "client": f"Client {cid}", "n_nodes": len(cl_nodes)}
                co_row = {"algorithm": strat_name, "n_clients": n_clients,
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
                "algorithm": strat_name, "n_clients": n_clients,
                "total_edges": total_edges,
                "internal_total": internal_total, "cross_total": cross_total,
                "pct_preserved": internal_total / total_edges * 100,
                "attack_preserved": internal_atk, "attack_lost": cross_atk,
                "attack_pct_preserved": internal_atk / max(n_attack_e, 1) * 100,
                "benign_preserved": internal_ben, "benign_lost": cross_ben,
                "benign_pct_preserved": internal_ben / max(n_benign_e, 1) * 100,
            })

            for atype in sorted(set(d.get("attack_type", "Benign")
                                     for _, _, d in G.edges(data=True))):
                t_edges = [(u, v) for u, v, d in G.edges(data=True)
                           if d.get("attack_type") == atype]
                preserved = sum(1 for (u, v) in t_edges
                                if node_to_client.get(u) == node_to_client.get(v))
                total = len(t_edges)
                all_attack_type_rows.append({
                    "algorithm": strat_name, "n_clients": n_clients,
                    "attack_type": atype, "total_edges": total,
                    "preserved": preserved, "lost": total - preserved,
                    "pct_preserved": preserved / max(total, 1) * 100,
                })

            # ── summary row ─────────────────────────────────
            # Filter explicitly by the CURRENT strategy and client count.
            # This is important for community partitions with empty clients:
            # slicing the last n_clients rows can accidentally include rows
            # from the previous strategy when no row was created for an empty
            # client.
            sp_vals, pe_vals, mae_vals = [], [], []

            current_corr = [
                co for co in all_corr_rows
                if co.get("algorithm") == strat_name
                and co.get("n_clients") == n_clients
            ]
            current_change = [
                ch for ch in all_change_rows
                if ch.get("algorithm") == strat_name
                and ch.get("n_clients") == n_clients
            ]

            for co in current_corr:
                for m in METRICS:
                    if not np.isnan(co.get(f"{m}_spearman", np.nan)):
                        sp_vals.append(co[f"{m}_spearman"])
                    if not np.isnan(co.get(f"{m}_pearson", np.nan)):
                        pe_vals.append(co[f"{m}_pearson"])

            for ch in current_change:
                for m in METRICS:
                    mae_vals.append(ch[f"{m}_mae"])
            fed_time = sum(sum(client_timings[cid].values())
                           for cid in client_timings)

            summary_entry = {
                "algorithm": strat_name, "n_clients": n_clients,
                "avg_spearman": np.mean(sp_vals) if sp_vals else 0,
                "avg_pearson":  np.mean(pe_vals) if pe_vals else 0,
                "avg_mae":      np.mean(mae_vals) if mae_vals else 0,
                "edge_pct_preserved":   internal_total / total_edges * 100,
                "attack_pct_preserved": internal_atk / max(n_attack_e, 1) * 100,
                "fed_time":       fed_time,
                "global_time":    global_total_time,
            }
            all_summary.append(summary_entry)

    # ── DataFrames ─────────────────────────────────────────
    timing_df  = pd.DataFrame(all_timing_rows)
    change_df  = pd.DataFrame(all_change_rows)
    corr_df    = pd.DataFrame(all_corr_rows)
    summary_df = pd.DataFrame(all_summary)
    edge_df    = pd.DataFrame(all_edge_rows)
    attack_df  = pd.DataFrame(all_attack_type_rows)

    # ── Save CSVs ──────────────────────────────────────────
    def save(df, name):
        p = os.path.join(output_dir, name)
        df.to_csv(p, index=False, float_format="%.6f")
        print(f"   saved: {name}")
    print("\nSaving CSVs...")
    save(timing_df,  "timing_comparison.csv")
    save(change_df,  "change_detection.csv")
    save(corr_df,    "correlation_per_client.csv")
    save(summary_df, "summary_by_algo_nclients.csv")
    save(edge_df,    "edge_preservation.csv")
    save(attack_df,  "attack_type_preservation.csv")

    # ── Tables ─────────────────────────────────────────────
    print_summary_table(summary_df)

    # ── Visualisations ─────────────────────────────────────
    draw_spearman_heatmap(corr_df, output_dir, n_clients=MAIN_N_CLIENTS)
    draw_pearson_heatmap(corr_df, output_dir, n_clients=MAIN_N_CLIENTS)
    draw_change_heatmap(change_df, output_dir, n_clients=MAIN_N_CLIENTS)
    draw_edge_preservation(edge_df, output_dir, n_clients=MAIN_N_CLIENTS)
    draw_attack_type_preservation(attack_df, output_dir, n_clients=MAIN_N_CLIENTS)
    draw_scaling_correlation(summary_df, output_dir)
    draw_scaling_edge(summary_df, output_dir)
    draw_tradeoff_summary(summary_df, output_dir, n_clients=MAIN_N_CLIENTS)

    return {"timing": timing_df, "change": change_df, "corr": corr_df,
            "summary": summary_df, "edges": edge_df, "attacks": attack_df}


# ══════════════════════════════════════════════════════════════
# PRINT TABLES
# ══════════════════════════════════════════════════════════════

def print_summary_table(df):
    print(f"\n{'='*100}")
    print("  SUMMARY: strategy × #clients")
    print(f"{'='*100}")
    print(f"{'Strategy':<20} {'#Cli':>5} {'Spearman':>10} {'Pearson':>9} "
          f"{'MAE':>10} {'Edge%':>8} {'Atk%':>8} {'FedT':>8} {'GlobalT':>9}")
    print("-" * 100)
    for _, r in df.iterrows():
        print(f"{r['algorithm']:<20} {int(r['n_clients']):>5} "
              f"{r['avg_spearman']:>10.3f} {r['avg_pearson']:>9.3f} "
              f"{r['avg_mae']:>10.5f} "
              f"{r['edge_pct_preserved']:>7.1f}% "
              f"{r['attack_pct_preserved']:>7.1f}% "
              f"{r['fed_time']:>7.2f}s {r['global_time']:>8.2f}s")
    print("=" * 100)


# ══════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════

def _strat_list_from(df):
    canonical = list(STRAT_COLORS.keys())
    present = set(df["algorithm"].unique())
    return [s for s in canonical if s in present]


def _heatmap_panel(ax, matrix, row_labels, col_labels, title,
                   cmap, vmin, vmax, cell_text=None, fmt="{:.3f}"):
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
    strats = _strat_list_from(sub_all)
    n_strats = len(strats)
    if n_strats == 0:
        return

    fig, axes = plt.subplots(1, n_strats, figsize=(6 * n_strats, 7), facecolor=BG)
    if n_strats == 1:
        axes = [axes]
    fig.suptitle(f"Spearman rank correlation — per client × centrality  (n_clients={n_clients})",
                 color=TEXT, fontsize=15, fontweight="bold", y=1.02)

    im = None
    for idx, strat in enumerate(strats):
        ax = axes[idx]
        sub = sub_all[sub_all["algorithm"] == strat]
        rows, labels = [], []
        for _, r in sub.iterrows():
            labels.append(r["client"])
            rows.append([r.get(f"{m}_spearman", np.nan) for m in METRICS])
        mat = np.array(rows) if rows else np.zeros((1, len(METRICS)))
        if not rows:
            labels = ["(empty)"]
        im = _heatmap_panel(ax, mat, labels,
                            [METRIC_LABELS[m] for m in METRICS],
                            strat, cmap="RdYlGn", vmin=-0.4, vmax=1.0)

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
    strats = _strat_list_from(sub_all)
    n_strats = len(strats)
    if n_strats == 0:
        return

    fig, axes = plt.subplots(1, n_strats, figsize=(6 * n_strats, 7), facecolor=BG)
    if n_strats == 1:
        axes = [axes]
    fig.suptitle(f"Pearson correlation — per client × centrality  (n_clients={n_clients})",
                 color=TEXT, fontsize=15, fontweight="bold", y=1.02)

    im = None
    for idx, strat in enumerate(strats):
        ax = axes[idx]
        sub = sub_all[sub_all["algorithm"] == strat]
        rows, labels = [], []
        for _, r in sub.iterrows():
            labels.append(r["client"])
            rows.append([r.get(f"{m}_pearson", np.nan) for m in METRICS])
        mat = np.array(rows) if rows else np.zeros((1, len(METRICS)))
        if not rows:
            labels = ["(empty)"]
        im = _heatmap_panel(ax, mat, labels,
                            [METRIC_LABELS[m] for m in METRICS],
                            strat, cmap="RdYlGn", vmin=-0.2, vmax=1.0)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.22, wspace=0.3)
    cax = fig.add_axes([0.3, 0.05, 0.4, 0.025])
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal", label="Pearson r")
    cbar.ax.tick_params(colors=MUTED)
    out = os.path.join(output_dir, f"3_pearson_heatmap_n{n_clients}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"   saved: {out}")


def draw_change_heatmap(change_df, output_dir, n_clients):
    print(f"\nDrawing change-detection heatmap (n={n_clients})...")
    sub_all = change_df[change_df["n_clients"] == n_clients]
    strats = _strat_list_from(sub_all)
    n_strats = len(strats)
    if n_strats == 0:
        return

    fig, axes = plt.subplots(1, n_strats, figsize=(6 * n_strats, 7), facecolor=BG)
    if n_strats == 1:
        axes = [axes]
    fig.suptitle(f"Nodes whose centrality changed  (|diff| > 0.001)  —  n_clients={n_clients}",
                 color=TEXT, fontsize=15, fontweight="bold", y=1.02)

    im = None
    for idx, strat in enumerate(strats):
        ax = axes[idx]
        sub = sub_all[sub_all["algorithm"] == strat]
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
                            strat, cmap="RdYlBu_r", vmin=0, vmax=100,
                            cell_text=text_rows)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.22, wspace=0.3)
    cax = fig.add_axes([0.3, 0.05, 0.4, 0.025])
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal", label="% of nodes changed")
    cbar.ax.tick_params(colors=MUTED)
    out = os.path.join(output_dir, f"4_change_heatmap_n{n_clients}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"   saved: {out}")


def draw_edge_preservation(edge_df, output_dir, n_clients):
    print(f"\nDrawing edge preservation (n={n_clients})...")
    sub = edge_df[edge_df["n_clients"] == n_clients].reset_index(drop=True)
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor=BG)
    fig.suptitle(f"Edge preservation by strategy  (n_clients={n_clients})",
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
    strats = _strat_list_from(sub_all)
    n_strats = len(strats)
    if n_strats == 0:
        return

    fig, axes = plt.subplots(1, n_strats, figsize=(6 * n_strats, 8), facecolor=BG)
    if n_strats == 1:
        axes = [axes]
    fig.suptitle(f"Attack-type preservation by strategy  (n_clients={n_clients})",
                 color=TEXT, fontsize=15, fontweight="bold", y=1.02)

    for idx, strat in enumerate(strats):
        ax = axes[idx]; style_ax(ax, strat)
        sub = sub_all[sub_all["algorithm"] == strat].sort_values("total_edges",
                                                                  ascending=True)
        y = np.arange(len(sub))
        ax.barh(y, sub["pct_preserved"], color=STRAT_COLORS[strat], alpha=0.85)
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


def draw_scaling_correlation(summary_df, output_dir):
    print("\nDrawing scaling chart (Spearman & Pearson vs #clients)...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), facecolor=BG)
    fig.suptitle("Scalability: correlation vs number of clients",
                 color=TEXT, fontsize=16, fontweight="bold", y=1.02)

    ncs = sorted(summary_df["n_clients"].unique())
    strats = _strat_list_from(summary_df)

    for ax_idx, (col, title, ylabel) in enumerate([
        ("avg_spearman", "Average Spearman rank correlation",
         "Spearman ρ  (higher = better)"),
        ("avg_pearson", "Average Pearson correlation",
         "Pearson r  (higher = better)"),
    ]):
        ax = axes[ax_idx]; style_ax(ax, title)
        for strat in strats:
            vals = []
            for nc in ncs:
                row = summary_df[(summary_df["algorithm"] == strat) &
                                  (summary_df["n_clients"] == nc)]
                vals.append(row[col].values[0] if len(row) else np.nan)
            ax.plot(ncs, vals, marker="o", linewidth=2.5, markersize=10,
                    color=STRAT_COLORS[strat], label=strat, alpha=0.9)
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
    strats = _strat_list_from(summary_df)

    for ax_idx, (col, title, ylabel) in enumerate([
        ("edge_pct_preserved",  "Total edge preservation",    "% edges preserved"),
        ("attack_pct_preserved","Attack edge preservation",   "% attack edges preserved"),
    ]):
        ax = axes[ax_idx]; style_ax(ax, title)
        for strat in strats:
            vals = []
            for nc in ncs:
                row = summary_df[(summary_df["algorithm"] == strat) &
                                  (summary_df["n_clients"] == nc)]
                vals.append(row[col].values[0] if len(row) else np.nan)
            ax.plot(ncs, vals, marker="s", linewidth=2.5, markersize=10,
                    color=STRAT_COLORS[strat], label=strat, alpha=0.9)
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


def draw_tradeoff_summary(summary_df, output_dir, n_clients):
    print(f"\nDrawing trade-off summary (n={n_clients})...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor=BG)
    fig.suptitle(f"Strategy trade-off summary  (n_clients={n_clients})",
                 color=TEXT, fontsize=16, fontweight="bold", y=1.02)

    sub = summary_df[summary_df["n_clients"] == n_clients]
    strats = sub["algorithm"].tolist()
    colors = [STRAT_COLORS[s] for s in strats]

    ax = axes[0]; style_ax(ax, "Rank correlation (Spearman)")
    bars = ax.bar(strats, sub["avg_spearman"], color=colors, alpha=0.85)
    for b, v in zip(bars, sub["avg_spearman"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", color=TEXT,
                fontsize=10, fontweight="bold")
    ax.set_ylabel("Spearman ρ (higher = better)", color=MUTED)
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis="x", labelsize=9, colors=TEXT, rotation=15)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)

    ax = axes[1]; style_ax(ax, "Attack-edge preservation")
    bars = ax.bar(strats, sub["attack_pct_preserved"], color=colors, alpha=0.85)
    for b, v in zip(bars, sub["attack_pct_preserved"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{v:.1f}%", ha="center", va="bottom", color=TEXT,
                fontsize=10, fontweight="bold")
    ax.set_ylabel("% attack edges preserved", color=MUTED)
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", labelsize=9, colors=TEXT, rotation=15)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)

    ax = axes[2]; style_ax(ax, "Federated centrality time")
    fed = sub["fed_time"].values
    g_time = sub["global_time"].iloc[0]
    bars = ax.bar(strats, fed, color=colors, alpha=0.85)
    for b, v in zip(bars, fed):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(fed) * 0.01,
                f"{v:.2f}s", ha="center", va="bottom", color=TEXT,
                fontsize=10, fontweight="bold")
    ax.axhline(y=g_time, color="#333333", linestyle="--",
               alpha=0.7, linewidth=1.5)
    ax.text(len(strats) - 0.5, g_time * 1.05,
            f"Centralized: {g_time:.2f}s",
            color="#333333", fontsize=9, ha="right", fontweight="bold")
    ax.set_ylabel("Time (seconds)", color=MUTED)
    ax.tick_params(axis="x", labelsize=9, colors=TEXT, rotation=15)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(output_dir, f"12_tradeoff_summary_n{n_clients}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"   saved: {out}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print(f"  STRATEGY COMPARISON")
    print(f"  ACTIVE DATASET: {ACTIVE_DATASET}")
    print(f"  CSV PATH:       {CSV_PATH}")
    print(f"  OUTPUT DIR:     {OUTPUT_DIR}")
    if SAMPLE_SIZE is not None:
        print(f"  SAMPLE SIZE:    {SAMPLE_SIZE:,} (stratified by attack class)")
        print(f"  SAMPLE SEED:    {SAMPLE_SEED}")
    else:
        print(f"  SAMPLE SIZE:    None (full file)")
    print("="*70)

    df = load_csv(CSV_PATH, sample_size=SAMPLE_SIZE)
    G  = build_graph(df)
    results = run_all(G, OUTPUT_DIR, CLIENT_COUNTS)
    print(f"\nStrategy comparison complete! Check: {OUTPUT_DIR}")
