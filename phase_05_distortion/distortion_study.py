"""
═══════════════════════════════════════════════════════════════════════════
 STEP 1 — Per-Node Edge Cut vs. Centrality Distortion
          (degree ≥ 2 filter  +  PERCENTILE-normalized rank shift)
          ***  reports BOTH Pearson and Spearman ***
═══════════════════════════════════════════════════════════════════════════

WHAT THIS SCRIPT DOES
---------------------
For every node v it computes:

  (a) cut_ratio(v)  = (edges of v that cross a partition boundary) / degree(v)
                       0 = nothing cut, 1 = node fully isolated in its client

  (b) pct_shift(v, m) = | global_percentile(v, m) − local_percentile(v, m) |
                        for each centrality metric m.

Then it correlates cut_ratio against pct_shift, separately for each metric
and each algorithm, and produces for BOTH Pearson and Spearman:

  • a correlation table  (metrics as rows, algorithms as columns)
  • a Pearson-vs-Spearman side-by-side comparison
  • one scatter plot per centrality metric (title shows both coefficients)

WHY BOTH?
  Pearson measures a STRAIGHT LINE relationship (does shift rise in constant
  proportion to cut_ratio?).  Spearman measures a MONOTONIC relationship (do
  nodes with more cut simply tend to rank higher in shift, even if the rise
  is curved?).  If the two agree, the relationship is roughly linear.  If
  Spearman is higher than Pearson, the relationship is real but curved.  If
  Spearman is lower, a few outliers were inflating Pearson.

TWO KEY CHANGES vs. the old one
----------------------------------
1. DEGREE FILTER.  Only nodes with undirected degree ≥ 2 enter the
   correlation. Degree-1 (leaf) nodes are structurally uninformative.

2. PERCENTILE NORMALIZED RANK SHIFT.  rank / N puts the global (large) and
   local (small) rankings on the same 0–1 scale before comparing.

Output folder: output_distortion/
═══════════════════════════════════════════════════════════════════════════
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
DATASET_PATH = r"NF-ToN-IoT.csv"
SRC_COL      = "IPV4_SRC_ADDR"
DST_COL      = "IPV4_DST_ADDR"
LABEL_COL    = "Label"
ATTACK_COL   = "Attack"
BYTES_COL    = "IN_BYTES"

OUTPUT_DIR   = "output_distortion"
N_CLIENTS    = 3
LOUVAIN_RES  = 0.8
SEED         = 42

# ── The two settings that implement the fix ───────────────
MIN_DEGREE       = 2      # include only nodes with undirected degree ≥ this
DROP_GLOBAL_ZERO = False  # OPTIONAL extra filter. Keep False to match the
                          # spec exactly (degree filter only).

ALGORITHMS   = ["Louvain (res=0.8)", "Girvan-Newman", "Label Propagation"]
PRIMARY_ALGO = "Louvain (res=0.8)"

METRICS = ["degree", "betweenness", "closeness", "pagerank"]

METRIC_LABELS = {
    "degree":      "Degree",
    "betweenness": "Betweenness",
    "closeness":   "Closeness",
    "pagerank":    "PageRank",
}

# Which correlation coefficients to report as deliverable tables.
COEFS = ["pearson", "spearman"]
COEF_LABELS = {"pearson": "Pearson r", "spearman": "Spearman \u03c1"}

# Plot styling
BG, PANEL, GRID = "#FFFFFF", "#F5F5F5", "#E0E0E0"
TEXT, MUTED     = "#1A1A2E", "#555555"


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING & GRAPH BUILDING
# ═══════════════════════════════════════════════════════════════════════
def load_csv(path):
    print(f"\nLoading: {path}")
    t0 = time.time()
    df = pd.read_csv(path, low_memory=False)
    print(f"   {df.shape[0]:,} rows × {df.shape[1]} columns "
          f"({time.time()-t0:.1f}s)")
    return df


def build_graph(df):
    print("\nBuilding network graph...")
    t0 = time.time()
    df[SRC_COL] = df[SRC_COL].astype(str)
    df[DST_COL] = df[DST_COL].astype(str)

    G = nx.DiGraph()
    attack_ips = (set(df[df[LABEL_COL] == 1][SRC_COL].unique()) |
                  set(df[df[LABEL_COL] == 1][DST_COL].unique()))
    all_ips = pd.concat([df[SRC_COL], df[DST_COL]]).unique()
    for ip in all_ips:
        G.add_node(ip, role="attack" if ip in attack_ips else "benign")

    edge_agg = (df.groupby([SRC_COL, DST_COL, LABEL_COL])
                .agg(flow_count=(LABEL_COL, "count"),
                     total_bytes=(BYTES_COL, "sum"),
                     attack_type=(ATTACK_COL, lambda x: x.mode()[0]))
                .reset_index())
    for _, row in edge_agg.iterrows():
        G.add_edge(row[SRC_COL], row[DST_COL],
                   weight=row["flow_count"],
                   total_bytes=row["total_bytes"],
                   label=int(row[LABEL_COL]),
                   attack_type=row["attack_type"])

    UG = G.to_undirected()
    degrees = [UG.degree(n) for n in UG.nodes()]
    n_deg_1   = sum(1 for d in degrees if d == 1)
    n_deg_ge2 = sum(1 for d in degrees if d >= 2)
    print(f"   Nodes:                  {G.number_of_nodes():>8,}")
    print(f"   Edges (directed):       {G.number_of_edges():>8,}")
    print(f"   Degree-1 nodes:         {n_deg_1:>8,}  "
          f"({n_deg_1 / G.number_of_nodes() * 100:.1f}%)  ← dropped by filter")
    print(f"   Degree-≥2 nodes:        {n_deg_ge2:>8,}  "
          f"({n_deg_ge2 / G.number_of_nodes() * 100:.1f}%)  ← kept")
    print(f"   Build time: {time.time()-t0:.1f}s")
    return G


# ═══════════════════════════════════════════════════════════════════════
# CENTRALITY COMPUTATION
# ═══════════════════════════════════════════════════════════════════════
def compute_centralities(G, label=""):
    """All 4 centralities, returned as RAW (un-normalized) values."""
    t0 = time.time()
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {m: {} for m in METRICS}
    if n == 1:
        return {m: {nodes[0]: 0.0} for m in METRICS}

    results = {}
    results["degree"] = dict(G.degree())          # raw integer degree
    if n <= 2:
        results["betweenness"] = {nd: 0.0 for nd in nodes}
    else:
        results["betweenness"] = nx.betweenness_centrality(
            G, k=None, normalized=False)          # raw path counts
    results["closeness"] = nx.closeness_centrality(G, wf_improved=False)
    try:
        results["pagerank"] = nx.pagerank(G, alpha=0.85, max_iter=200)
    except Exception:
        try:
            results["pagerank"] = nx.pagerank(G, alpha=0.85, max_iter=500,
                                              tol=1e-4)
        except Exception:
            results["pagerank"] = {nd: 1.0 / n for nd in nodes}
    if label:
        print(f"   [{label}] {n} nodes | {time.time()-t0:.2f}s")
    return results


# ═══════════════════════════════════════════════════════════════════════
# COMMUNITY DETECTION  → CLIENTS
# ═══════════════════════════════════════════════════════════════════════
def detect_louvain(G, resolution=LOUVAIN_RES):
    U = G.to_undirected()
    return list(nx.community.louvain_communities(
        U, seed=SEED, resolution=resolution))


def detect_girvan_newman(G):
    # NOTE: takes only the FIRST split (2 communities).
    U = G.to_undirected()
    gen = nx.community.girvan_newman(U)
    try:
        return [set(c) for c in next(gen)]
    except StopIteration:
        return [set(U.nodes())]


def detect_label_propagation(G):
    U = G.to_undirected()
    return [set(c) for c in nx.community.asyn_lpa_communities(
        U, weight="weight", seed=SEED)]


def run_algorithm(name, G):
    print(f"\nRunning {name}...")
    t0 = time.time()
    if name == "Louvain (res=0.8)":
        comms = detect_louvain(G, resolution=LOUVAIN_RES)
    elif name == "Girvan-Newman":
        comms = detect_girvan_newman(G)
    elif name == "Label Propagation":
        comms = detect_label_propagation(G)
    else:
        raise ValueError(f"Unknown algorithm: {name}")
    print(f"   Found {len(comms)} communities in {time.time()-t0:.2f}s")
    return comms


def communities_to_clients(communities, n_clients):
    """Greedily pack communities into n_clients (balances client sizes)."""
    sorted_comms = sorted(communities, key=len, reverse=True)
    parts = [set() for _ in range(n_clients)]
    sizes = [0] * n_clients
    for comm in sorted_comms:
        j = sizes.index(min(sizes))
        parts[j].update(comm)
        sizes[j] += len(comm)
    return parts


# ═══════════════════════════════════════════════════════════════════════
# CUT RATIO
# ═══════════════════════════════════════════════════════════════════════
def compute_cut_ratios(G, node_to_client):
    UG = G.to_undirected()
    cut_data = {}
    for v in UG.nodes():
        my_client = node_to_client.get(v)
        neighbours = list(UG.neighbors(v))
        deg = len(neighbours)
        if deg == 0:
            cut_data[v] = {"degree_undir": 0, "cut_edges": 0,
                           "cut_ratio": np.nan}
            continue
        cut = sum(1 for nb in neighbours
                  if node_to_client.get(nb) != my_client)
        cut_data[v] = {"degree_undir": deg, "cut_edges": cut,
                       "cut_ratio": cut / deg}
    return cut_data


# ═══════════════════════════════════════════════════════════════════════
# PERCENTILE RANKS
# ═══════════════════════════════════════════════════════════════════════
def percentile_ranks(centrality_dict, n_pop):
    """Turn {node: value} into ({node: rank}, {node: percentile})."""
    if not centrality_dict or n_pop < 1:
        return {}, {}
    s = pd.Series(centrality_dict, dtype=float)
    ranks = s.rank(ascending=False, method="average")
    pct = ranks / n_pop
    return ranks.to_dict(), pct.to_dict()


def compute_global_percentiles(global_cents, n_global):
    out = {}
    for m in METRICS:
        ranks, pct = percentile_ranks(global_cents[m], n_global)
        out[m] = {"rank": ranks, "pct": pct}
    return out


def compute_local_percentiles(local_cents_per_client, partitions):
    out = {m: {"rank": {}, "pct": {}} for m in METRICS}
    for cid, part in enumerate(partitions):
        n_local = len(part)
        if n_local < 2:
            for m in METRICS:
                for nd in part:
                    out[m]["rank"][nd] = np.nan
                    out[m]["pct"][nd] = np.nan
            continue
        cents = local_cents_per_client[cid]
        for m in METRICS:
            ranks, pct = percentile_ranks(cents[m], n_local)
            out[m]["rank"].update(ranks)
            out[m]["pct"].update(pct)
    return out


def compute_pct_shifts(global_pct, local_pct, nodes):
    shifts = {nd: {} for nd in nodes}
    for nd in nodes:
        for m in METRICS:
            g = global_pct[m]["pct"].get(nd, np.nan)
            l = local_pct[m]["pct"].get(nd, np.nan)
            gr = global_pct[m]["rank"].get(nd, np.nan)
            lr = local_pct[m]["rank"].get(nd, np.nan)
            shifts[nd][f"global_rank_{m}"] = gr
            shifts[nd][f"local_rank_{m}"]  = lr
            shifts[nd][f"global_pct_{m}"]  = g
            shifts[nd][f"local_pct_{m}"]   = l
            shifts[nd][f"pct_shift_{m}"]   = (abs(g - l)
                                              if pd.notna(g) and pd.notna(l)
                                              else np.nan)
    return shifts


# ═══════════════════════════════════════════════════════════════════════
# NODE TABLE
# ═══════════════════════════════════════════════════════════════════════
def build_node_table(G, algo_name, cut_data, shifts, global_cents,
                     local_cents_per_client, node_to_client):
    rows = []
    for nd in G.nodes():
        row = {
            "algorithm":    algo_name,
            "node":         nd,
            "role":         G.nodes[nd].get("role"),
            "client":       node_to_client.get(nd),
            "degree_undir": cut_data[nd]["degree_undir"],
            "cut_edges":    cut_data[nd]["cut_edges"],
            "cut_ratio":    cut_data[nd]["cut_ratio"],
            **shifts[nd],
        }
        cid = node_to_client.get(nd)
        for m in METRICS:
            row[f"global_val_{m}"] = global_cents[m].get(nd, np.nan)
            row[f"local_val_{m}"]  = (local_cents_per_client[cid][m].get(nd, np.nan)
                                      if cid is not None else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# CORRELATIONS  (Pearson AND Spearman)  : apply degree ≥ 2 filter here
# ═══════════════════════════════════════════════════════════════════════
def strength_label(r):
    a = abs(r)
    if   a >= 0.5: s = "strong"
    elif a >= 0.3: s = "moderate"
    elif a >= 0.1: s = "weak"
    else:          s = "near zero"
    return f"{s} {'positive' if r >= 0 else 'negative'}"


def compute_correlations(node_df, algo_name):
    rows = []
    for m in METRICS:
        shift_col = f"pct_shift_{m}"
        val_col   = f"global_val_{m}"

        # ── THE FILTER: degree ≥ MIN_DEGREE, and optionally drop zeros ──
        mask = node_df["degree_undir"] >= MIN_DEGREE
        if DROP_GLOBAL_ZERO and val_col in node_df.columns:
            mask &= node_df[val_col] > 0

        sub = node_df.loc[mask, ["cut_ratio", shift_col]].dropna()
        sub = sub.rename(columns={shift_col: "shift"})
        n = len(sub)

        if n < 3 or sub["cut_ratio"].std() == 0 or sub["shift"].std() == 0:
            rows.append({
                "algorithm": algo_name, "metric": m,
                "metric_label": METRIC_LABELS[m], "n_nodes": n,
                "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_rho": np.nan, "spearman_p": np.nan,
                "verdict_pearson": "insufficient data",
                "verdict_spearman": "insufficient data",
            })
            continue

        pr, pp = pearsonr(sub["cut_ratio"], sub["shift"])
        sr, sp = spearmanr(sub["cut_ratio"], sub["shift"])
        rows.append({
            "algorithm": algo_name, "metric": m,
            "metric_label": METRIC_LABELS[m], "n_nodes": n,
            "pearson_r": pr, "pearson_p": pp,
            "spearman_rho": sr, "spearman_p": sp,
            "verdict_pearson": strength_label(pr),
            "verdict_spearman": strength_label(sr),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# PER-ALGORITHM PIPELINE
# ═══════════════════════════════════════════════════════════════════════
def run_one_algorithm(algo_name, G, global_cents, global_pct):
    print("\n" + "═" * 70)
    print(f"  ALGORITHM: {algo_name}")
    print("═" * 70)

    communities = run_algorithm(algo_name, G)
    partitions  = communities_to_clients(communities, N_CLIENTS)

    node_to_client = {}
    for cid, part in enumerate(partitions):
        for nd in part:
            node_to_client[nd] = cid
        atk = sum(1 for nd in part if G.nodes[nd].get("role") == "attack")
        print(f"   Client {cid}: {len(part):>5,} nodes  |  "
              f"{atk:>4} attack nodes")

    print("\n   Computing per-client centralities...")
    local_cents_per_client = {}
    for cid, part in enumerate(partitions):
        sub_edges = [(u, v, d) for u, v, d in G.edges(data=True)
                     if u in part and v in part]
        C = nx.DiGraph()
        C.add_nodes_from([(nd, G.nodes[nd]) for nd in part])
        C.add_edges_from(sub_edges)
        local_cents_per_client[cid] = compute_centralities(C, f"Client {cid}")

    print("\n   Computing cut ratios and percentile rank shifts...")
    cut_data  = compute_cut_ratios(G, node_to_client)
    local_pct = compute_local_percentiles(local_cents_per_client, partitions)
    shifts    = compute_pct_shifts(global_pct, local_pct, list(G.nodes()))

    node_df = build_node_table(G, algo_name, cut_data, shifts,
                               global_cents, local_cents_per_client,
                               node_to_client)
    corr_df = compute_correlations(node_df, algo_name)
    return node_df, corr_df


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT TABLES  (generic: works for Pearson OR Spearman)
# ═══════════════════════════════════════════════════════════════════════
def make_pivots(full_corr_df, coef):
    """Return (value_pivot, p_pivot, n_pivot) for the chosen coefficient."""
    val_col = "pearson_r" if coef == "pearson" else "spearman_rho"
    p_col   = "pearson_p" if coef == "pearson" else "spearman_p"
    piv_v = full_corr_df.pivot(index="metric", columns="algorithm",
                               values=val_col).reindex(METRICS)
    piv_p = full_corr_df.pivot(index="metric", columns="algorithm",
                               values=p_col).reindex(METRICS)
    piv_n = full_corr_df.pivot(index="metric", columns="algorithm",
                               values="n_nodes").reindex(METRICS)
    return piv_v, piv_p, piv_n


def print_correlation_table(full_corr_df, coef="pearson"):
    piv_v, piv_p, piv_n = make_pivots(full_corr_df, coef)
    algos = [a for a in ALGORITHMS if a in piv_v.columns]
    label = COEF_LABELS[coef]

    print("\n" + "=" * 112)
    print(f"  STEP 1 DELIVERABLE — {label}  (cut_ratio  vs.  percentile rank shift)")
    print(f"  Filter: undirected degree ≥ {MIN_DEGREE}"
          + ("  +  global value > 0" if DROP_GLOBAL_ZERO else "")
          + f"   |   n_clients = {N_CLIENTS}")
    print("=" * 112)
    header = f"{'Metric':<14} " + " ".join(f"{a:>26}" for a in algos)
    print(header)
    print("-" * len(header))
    for m in METRICS:
        cells = []
        for a in algos:
            r = piv_v.loc[m, a]; p = piv_p.loc[m, a]; n = piv_n.loc[m, a]
            if pd.isna(r):
                cells.append(f"{'n/a':>26}")
            else:
                p_str = "<0.001" if pd.notna(p) and p < 0.001 else f"p={p:.3f}"
                cells.append(f"{r:+.3f} ({p_str}, n={int(n)})".rjust(26))
        print(f"{METRIC_LABELS[m]:<14} " + " ".join(cells))
    print("=" * 112)
    return piv_v, piv_p, piv_n


def print_pearson_vs_spearman(full_corr_df):
    """Side-by-side so you can instantly see if the story changed."""
    print("\n" + "=" * 100)
    print("  PEARSON vs. SPEARMAN  —  did the result change?")
    print("  (Δ = Spearman − Pearson.  Big +Δ = relationship is monotonic but"
          " curved; big −Δ = a few outliers inflated Pearson.)")
    print("=" * 100)
    for algo in ALGORITHMS:
        sub = full_corr_df[full_corr_df["algorithm"] == algo]
        if sub.empty:
            continue
        print(f"\n  ── {algo} ──")
        print(f"  {'Metric':<14}{'Pearson r':>12}{'Spearman ρ':>13}{'Δ':>10}")
        print(f"  {'-'*14}{'-'*12}{'-'*13}{'-'*10}")
        for m in METRICS:
            row = sub[sub["metric"] == m]
            if row.empty:
                continue
            pr = row.iloc[0]["pearson_r"]
            sr = row.iloc[0]["spearman_rho"]
            pr_s = "n/a" if pd.isna(pr) else f"{pr:+.3f}"
            sr_s = "n/a" if pd.isna(sr) else f"{sr:+.3f}"
            d_s  = "—" if (pd.isna(pr) or pd.isna(sr)) else f"{sr - pr:+.3f}"
            print(f"  {METRIC_LABELS[m]:<14}{pr_s:>12}{sr_s:>13}{d_s:>10}")
    print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════
# SCATTER PLOTS — one panel per centrality metric (shows BOTH coefficients)
# ═══════════════════════════════════════════════════════════════════════
def _style(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=10.5, pad=8, fontweight="bold")
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID); sp.set_linewidth(0.7)


def draw_scatter_grid(node_df, corr_df, algo_name, output_dir):
    print(f"\nDrawing scatter grid for {algo_name}...")
    n_cols = 3
    n_rows = (len(METRICS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4.5 * n_rows),
                             facecolor=BG)
    fig.suptitle(f"Edge cut ratio  vs.  percentile rank shift   "
                 f"({algo_name}, {N_CLIENTS} clients, degree ≥ {MIN_DEGREE})",
                 color=TEXT, fontsize=15, fontweight="bold", y=1.00)
    axes_flat = axes.flatten()

    for idx, m in enumerate(METRICS):
        ax = axes_flat[idx]
        cr = corr_df[corr_df["metric"] == m]
        r_val   = cr.iloc[0]["pearson_r"]    if not cr.empty else np.nan
        rho_val = cr.iloc[0]["spearman_rho"] if not cr.empty else np.nan
        sp_val  = cr.iloc[0]["spearman_p"]   if not cr.empty else np.nan
        n_val   = cr.iloc[0]["n_nodes"]      if not cr.empty else 0

        # Title now shows BOTH coefficients
        if pd.notna(r_val):
            title = (f"{METRIC_LABELS[m]}  —  Pearson {r_val:+.3f}  ·  "
                     f"Spearman {rho_val:+.3f}  (n={int(n_val):,})")
        else:
            title = f"{METRIC_LABELS[m]}  —  n/a"
        _style(ax, title)

        mask = node_df["degree_undir"] >= MIN_DEGREE
        if DROP_GLOBAL_ZERO and f"global_val_{m}" in node_df.columns:
            mask &= node_df[f"global_val_{m}"] > 0
        sub = node_df.loc[mask, ["cut_ratio", f"pct_shift_{m}", "role"]].dropna()

        for role, c in [("benign", "#2E7D32"), ("attack", "#D32F2F")]:
            r_sub = sub[sub["role"] == role]
            if not r_sub.empty:
                ax.scatter(r_sub["cut_ratio"], r_sub[f"pct_shift_{m}"],
                           s=18, alpha=0.55, color=c, edgecolors="none",
                           label=role)

        if not cr.empty and pd.notna(r_val) and len(sub) >= 2:
            x = sub["cut_ratio"].values
            y = sub[f"pct_shift_{m}"].values
            if np.std(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, slope * xs + intercept, "--",
                        color="#333", linewidth=1.5, alpha=0.8)
            p_str = ("<0.001" if pd.notna(sp_val) and sp_val < 0.001
                     else (f"={sp_val:.3f}" if pd.notna(sp_val) else "=n/a"))
            ax.text(0.04, 0.96,
                    f"Spearman ρ = {rho_val:+.3f}\np {p_str}",
                    transform=ax.transAxes, ha="left", va="top",
                    color=TEXT, fontsize=8,
                    bbox=dict(facecolor="white", edgecolor=GRID,
                              alpha=0.92, pad=4))

        ax.set_xlabel("cut ratio", color=MUTED, fontsize=9)
        ax.set_ylabel("percentile rank shift", color=MUTED, fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right", facecolor=BG,
                      edgecolor=GRID)

    for idx in range(len(METRICS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    safe = algo_name.replace(" ", "_").replace("(", "").replace(")", "") \
                    .replace("=", "").replace(".", "")
    out = os.path.join(output_dir, f"scatter_grid_{safe}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"   saved: {out}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  STEP 1 — edge cut vs. centrality distortion")
    print("           degree ≥ {} filter + percentile rank shift"
          .format(MIN_DEGREE))
    print("           reporting BOTH Pearson and Spearman")
    print(f"  Dataset:    {DATASET_PATH}")
    print(f"  Algorithms: {', '.join(ALGORITHMS)}")
    print(f"  n_clients:  {N_CLIENTS}")
    print(f"  Output:     {OUTPUT_DIR}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_csv(DATASET_PATH)
    G  = build_graph(df)

    print("\n" + "=" * 70)
    print("  Global centralities (computed once on the FULL graph)")
    print("=" * 70)
    global_cents = compute_centralities(G, "Global")
    global_pct   = compute_global_percentiles(global_cents, G.number_of_nodes())

    all_node_dfs, all_corr_dfs = [], []
    for algo in ALGORITHMS:
        try:
            t0 = time.time()
            node_df, corr_df = run_one_algorithm(algo, G, global_cents,
                                                 global_pct)
            print(f"\n   [{algo}] total time: {time.time()-t0:.1f}s")
            all_node_dfs.append(node_df)
            all_corr_dfs.append(corr_df)
        except Exception as e:
            print(f"\n   ⚠  {algo} failed: {e}")

    if not all_corr_dfs:
        print("\n❌ No algorithm succeeded — aborting.")
        return

    full_node_df = pd.concat(all_node_dfs, ignore_index=True)
    full_corr_df = pd.concat(all_corr_dfs, ignore_index=True)

    # Save per-node + long correlations (already contains BOTH coefficients)
    print("\n" + "=" * 70)
    print("  Saving outputs")
    print("=" * 70)
    node_path = os.path.join(OUTPUT_DIR, "node_distortion.csv")
    corr_path = os.path.join(OUTPUT_DIR, "correlations.csv")
    full_node_df.to_csv(node_path, index=False, float_format="%.6f")
    full_corr_df.to_csv(corr_path, index=False, float_format="%.6f")
    print(f"   saved: {node_path}  ({len(full_node_df):,} rows)")
    print(f"   saved: {corr_path}")

    # Deliverable tables — one per coefficient
    for coef in COEFS:
        piv_v, _, _ = print_correlation_table(full_corr_df, coef)
        out_csv = os.path.join(OUTPUT_DIR, f"{coef}_table.csv")
        piv_v.reindex(METRICS).to_csv(out_csv, float_format="%.6f")
        print(f"   saved: {out_csv}")

    # Side-by-side comparison
    print_pearson_vs_spearman(full_corr_df)

    # Scatter grids (title shows both coefficients)
    for algo in ALGORITHMS:
        ndf = full_node_df[full_node_df["algorithm"] == algo]
        cdf = full_corr_df[full_corr_df["algorithm"] == algo]
        if not ndf.empty:
            draw_scatter_grid(ndf, cdf, algo, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print(f"  Done. Files in: {OUTPUT_DIR}/")
    print("     - node_distortion.csv     (per-node detail)")
    print("     - correlations.csv        (Pearson + Spearman, long format)")
    print("     - pearson_table.csv       (deliverable — Pearson)")
    print("     - spearman_table.csv      (deliverable — Spearman)")
    print("     - scatter_grid_*.png      (one grid per algorithm; both coefs)")
    print("=" * 70)


if __name__ == "__main__":
    main()