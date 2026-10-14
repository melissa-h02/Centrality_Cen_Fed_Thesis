"""
═══════════════════════════════════════════════════════════════════════════
 GRAPH NEURAL NETWORKS — GraphSAGE and GAT   (traffic vs traffic+centralities)
═══════════════════════════════════════════════════════════════════════════

 The same A-vs-B comparison as run_all_models.py, but with graph neural
 networks. Point FEATURES_CSV at any step's feature file:

   edge_features_global.csv           -> Step 1 (global centralities)
   edge_features_local_3client.csv    -> Step 2 (local centralities)
   edge_features_corrected_3client.csv-> Step 3 (corrected centralities)

 Same rows, same stratified 70/30 split (seed 42), same six metrics, so the
 numbers sit directly beside your RF / LSTM / CNN results.

 ── HOW A GNN CLASSIFIES A FLOW ───────────────────────────────────────────
   1. Build the network graph: every IP is a node, every observed
      connection is an edge.
   2. MESSAGE PASSING. Each node starts with a small feature vector and then
      repeatedly mixes in a summary of its neighbours' vectors. After two
      rounds every node's vector describes its own neighbourhood -- who it
      talks to, and who they talk to. That vector is its "embedding".
        GraphSAGE averages the neighbours.
        GAT learns how much attention to pay to each neighbour.
   3. CLASSIFY EACH FLOW from three things joined together:
        [ embedding of the source IP | embedding of the destination IP |
          the flow's own traffic features ]
      A small feed-forward network turns that into the attack class.

 ── WHAT MODEL A AND MODEL B MEAN HERE ────────────────────────────────────
   MODEL A  node features = a constant. The GNN still learns structure
            through message passing, so this is NOT a structure-free
            baseline -- it is "graph structure, but no centrality numbers".
   MODEL B  node features = the centralities from the feature file.

   Because a GNN already extracts structure on its own, the gain from adding
   centralities is expected to be SMALLER here than it was for LSTM/CNN.
   That is itself a finding worth reporting, not a failure.

 ── TIME ─────────────────────────────────────────────────────────────────
   The graph is small (about 1,500 nodes), so message passing is cheap and
   this runs FASTER than the LSTM: roughly 4-8 minutes per model on CPU.

 Output: <OUTPUT_ROOT>/GraphSAGE/, <OUTPUT_ROOT>/GAT/,
         all_gnn_summary.csv, gnn_centrality_effect.csv, all_gnn_comparison.png
 Run:    python run_gnn_models.py
         first:  pip install torch torch_geometric
═══════════════════════════════════════════════════════════════════════════
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv

# ── Config ──────────────────────────────────────────────────────────────
FEATURES_CSV = "edge_features_global.csv"      # <- change per step
TAG          = "step1_global"                   # <- change per step
OUTPUT_ROOT  = "results_step1_global_gnn"       # <- change per step

TARGET = "Attack"
SRC_COL, DST_COL, LABEL_COL = "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label"
SEED = 42
TEST_SIZE, VAL_SIZE = 0.30, 0.15
SAMPLE_N = None                 # None = full data

EPOCHS, BATCH, PATIENCE, LR = 30, 8192, 5, 1e-3
HIDDEN, HEADS, DROPOUT = 64, 4, 0.3

SELECTED_TRAFFIC = ['L4_DST_PORT', 'OUT_BYTES', 'IN_BYTES', 'TCP_FLAGS',
                    'L4_SRC_PORT', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS',
                    'IN_PKTS', 'L7_PROTO']
CENTRALITIES = ["degree_undir", "betweenness", "closeness", "pagerank"]
MODEL_B_RAW = ["L4_DST_PORT", "OUT_BYTES", "IN_BYTES", "TCP_FLAGS", "L4_SRC_PORT"]

RUN_MODELS = ["GraphSAGE", "GAT"]
METRICS = ["Accuracy", "Macro Accuracy", "Macro F1", "Weighted F1", "FNR", "FPR"]
LOWER_BETTER = {"FNR", "FPR"}
TEXT, MUTED, GRID, A_C, B_C = "#1A2330", "#5B6B7A", "#E0E6EC", "#90A4AE", "#2E7D32"

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── metrics (identical definitions to run_all_models.py) ────────────────
def per_class_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    P, R, F1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    rows = []
    for i, lab in enumerate(labels):
        TP = cm[i, i]; FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP; TN = total - TP - FN - FP
        rows.append({"class": lab, "precision": P[i], "recall": R[i], "f1": F1[i],
                     "FNR": FN / (FN + TP) if (FN + TP) else 0.0,
                     "FPR": FP / (FP + TN) if (FP + TN) else 0.0,
                     "support": int(sup[i])})
    return pd.DataFrame(rows), cm


def summarize(pc, y_true, y_pred):
    acc = (np.asarray(y_true) == np.asarray(y_pred)).mean()
    w = pc["support"] / pc["support"].sum()
    return {"Accuracy": acc,
            "Macro Accuracy": pc["recall"].mean(),
            "Macro F1": pc["f1"].mean(),
            "Weighted F1": (pc["f1"] * w).sum(),
            "FNR": (pc["FNR"] * w).sum(),
            "FPR": (pc["FPR"] * w).sum()}


def report(name, pc, elapsed):
    print(f"\n  {name} — per-class metrics   ({elapsed:.0f}s)")
    print(f"  {'class':<12}{'prec':>8}{'recall':>8}{'f1':>8}"
          f"{'FNR':>8}{'FPR':>8}{'support':>10}")
    for _, r in pc.iterrows():
        print(f"  {r['class']:<12}{r['precision']:>8.3f}{r['recall']:>8.3f}"
              f"{r['f1']:>8.3f}{r['FNR']:>8.3f}{r['FPR']:>8.4f}"
              f"{int(r['support']):>10,}")


# ── the model ───────────────────────────────────────────────────────────
class EdgeGNN(nn.Module):
    """Two rounds of message passing to build node embeddings, then classify
    each flow from [src embedding | dst embedding | flow traffic features]."""

    def __init__(self, kind, n_node_feat, n_flow_feat, n_classes):
        super().__init__()
        if kind == "GraphSAGE":
            self.conv1 = SAGEConv(n_node_feat, HIDDEN)
            self.conv2 = SAGEConv(HIDDEN, HIDDEN)
        else:                                    # GAT
            self.conv1 = GATConv(n_node_feat, HIDDEN, heads=HEADS, concat=False)
            self.conv2 = GATConv(HIDDEN, HIDDEN, heads=HEADS, concat=False)
        self.head = nn.Sequential(
            nn.Linear(2 * HIDDEN + n_flow_feat, 128), nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes))

    def embed(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        return torch.relu(self.conv2(h, edge_index))

    def forward(self, x, edge_index, src, dst, flow_feat):
        h = self.embed(x, edge_index)
        return self.head(torch.cat([h[src], h[dst], flow_feat], dim=1))


def build_graph_tensors(d):
    """Map every IP to an index and build the undirected edge list."""
    nodes = pd.unique(pd.concat([d[SRC_COL], d[DST_COL]]).astype(str))
    idx = {ip: i for i, ip in enumerate(nodes)}
    src = d[SRC_COL].astype(str).map(idx).to_numpy()
    dst = d[DST_COL].astype(str).map(idx).to_numpy()

    pairs = np.unique(np.stack([src, dst], axis=1), axis=0)   # unique edges
    both = np.concatenate([pairs, pairs[:, ::-1]], axis=0)    # make undirected
    edge_index = torch.tensor(both.T, dtype=torch.long)
    print(f"  graph: {len(nodes):,} nodes, {len(pairs):,} unique connections "
          f"({edge_index.shape[1]:,} directed message paths)")
    return idx, src, dst, edge_index, len(nodes)


def node_feature_matrix(d, n_nodes, src_i, dst_i, cent_cols):
    """One row per node. A node's own centralities come from the src_* columns
    (never zeroed). Nodes that only ever appear as a destination fall back to
    dst_* values from same-client flows, where they are visible."""
    if not cent_cols:
        return torch.ones(n_nodes, 1)             # Model A: constant feature

    X = np.zeros((n_nodes, len(cent_cols)), dtype=np.float32)
    seen = np.zeros(n_nodes, dtype=bool)

    for j, c in enumerate(cent_cols):
        s = pd.DataFrame({"n": src_i, "v": d[f"src_{c}"].to_numpy(float)})
        agg = s.groupby("n")["v"].max()
        X[agg.index.to_numpy(), j] = agg.to_numpy()
        seen[agg.index.to_numpy()] = True

    missing = np.where(~seen)[0]
    if len(missing):
        for j, c in enumerate(cent_cols):
            t = pd.DataFrame({"n": dst_i, "v": d[f"dst_{c}"].to_numpy(float)})
            agg = t.groupby("n")["v"].max()
            hit = np.intersect1d(agg.index.to_numpy(), missing)
            if len(hit):
                X[hit, j] = agg.loc[hit].to_numpy()
        print(f"    {len(missing):,} nodes appear only as destinations "
              f"(filled from visible dst values)")
    return torch.tensor(X, dtype=torch.float32)


def train_eval(kind, variant, d, y_int, labels, le, edge_index, n_nodes,
               src_i, dst_i, tr_idx, val_idx, te_idx, cent_cols, flow_cols):
    name = f"{kind}-{variant}"
    t0 = time.time()
    print(f"\n  === {name}: node feats = "
          f"{len(cent_cols) if cent_cols else 'constant'}, "
          f"flow feats = {len(flow_cols)} ===")

    x = node_feature_matrix(d, n_nodes, src_i, dst_i, cent_cols)
    if cent_cols:                                  # scale node features
        sc_n = StandardScaler().fit(x.numpy())
        x = torch.tensor(sc_n.transform(x.numpy()), dtype=torch.float32)

    F = d[flow_cols].to_numpy(np.float32)
    sc_f = StandardScaler().fit(F[tr_idx])
    F = torch.tensor(sc_f.transform(F), dtype=torch.float32)

    src_t = torch.tensor(src_i, dtype=torch.long)
    dst_t = torch.tensor(dst_i, dtype=torch.long)
    y_t = torch.tensor(y_int, dtype=torch.long)

    model = EdgeGNN(kind, x.shape[1], len(flow_cols), len(labels))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()

    best_val, best_state, bad = np.inf, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = np.random.permutation(tr_idx)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), BATCH):
            b = torch.tensor(perm[i:i + BATCH], dtype=torch.long)
            opt.zero_grad()
            out = model(x, edge_index, src_t[b], dst_t[b], F[b])
            loss = lossf(out, y_t[b])
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1

        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for i in range(0, len(val_idx), BATCH):
                b = torch.tensor(val_idx[i:i + BATCH], dtype=torch.long)
                out = model(x, edge_index, src_t[b], dst_t[b], F[b])
                vl += lossf(out, y_t[b]).item(); vn += 1
            vl /= max(vn, 1)
        print(f"    epoch {ep:>2}/{EPOCHS}  train_loss {tot/max(nb,1):.4f}  "
              f"val_loss {vl:.4f}")

        if vl < best_val - 1e-5:
            best_val, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"    early stop at epoch {ep}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te_idx), BATCH):
            b = torch.tensor(te_idx[i:i + BATCH], dtype=torch.long)
            out = model(x, edge_index, src_t[b], dst_t[b], F[b])
            preds.append(out.argmax(1).numpy())
    pred = le.inverse_transform(np.concatenate(preds))
    y_te_lab = le.inverse_transform(y_int[te_idx])

    pc, cm = per_class_metrics(y_te_lab, pred, labels)
    pc.insert(0, "model", name)
    report(name, pc, time.time() - t0)
    return pc, cm, summarize(pc, y_te_lab, pred)


def save_family(folder, family, summaries, per_class, confusions, labels):
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame(summaries).T[METRICS].to_csv(
        os.path.join(folder, f"{family}_summary.csv"), float_format="%.6f")
    pd.concat(per_class).to_csv(os.path.join(folder, f"{family}_perclass.csv"),
                                index=False, float_format="%.6f")
    for v, cm in confusions.items():
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            os.path.join(folder, f"{family}_confusion_{v}.csv"))

    a, b = f"{family}-A", f"{family}-B"
    if a in summaries and b in summaries:
        av = [summaries[a][m] for m in METRICS]
        bv = [summaries[b][m] for m in METRICS]
        x = np.arange(len(METRICS)); w = 0.36
        fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
        fig.suptitle(f"{family} — traffic only vs + centralities   "
                     f"[FNR/FPR: lower is better]",
                     fontsize=14, fontweight="bold", color=TEXT, y=0.97)
        ax.bar(x - w/2, av, w, color=A_C, edgecolor="white", linewidth=1.2,
               zorder=3, label="A: traffic only")
        ax.bar(x + w/2, bv, w, color=B_C, edgecolor="white", linewidth=1.2,
               zorder=3, label="B: + centralities")
        for xi, (p, q) in enumerate(zip(av, bv)):
            ax.text(xi - w/2, p + 0.012, f"{p:.3f}", ha="center", va="bottom",
                    fontsize=9.5, color=TEXT, fontweight="bold")
            ax.text(xi + w/2, q + 0.012, f"{q:.3f}", ha="center", va="bottom",
                    fontsize=9.5, color=TEXT, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(METRICS, fontsize=10.5, color=TEXT)
        ax.set_ylim(0, max(0.9, max(max(av), max(bv)) * 1.25))
        ax.set_ylabel("score", color=MUTED, fontsize=11)
        ax.set_facecolor("#F7F9FB"); ax.tick_params(colors=MUTED)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]: ax.spines[s].set_color(GRID)
        ax.legend(frameon=False, fontsize=11, loc="upper right")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(folder, f"{family}_comparison.png"),
                    dpi=160, bbox_inches="tight", facecolor="white")
        plt.close()
    print(f"    -> saved {family} outputs to {folder}{os.sep}")


def main():
    if not os.path.exists(FEATURES_CSV):
        print(f"Could not find {FEATURES_CSV}. Build the features first.")
        return
    t_start = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    d = pd.read_csv(FEATURES_CSV)
    vc = d[TARGET].value_counts()
    small = vc[vc < 10].index.tolist()
    if small:
        print(f"  dropping classes with < 10 samples: {small}")
        d = d[~d[TARGET].isin(small)].reset_index(drop=True)
    if SAMPLE_N is not None and len(d) > SAMPLE_N:
        frac = SAMPLE_N / len(d)
        d = (pd.concat([g.sample(max(1, int(round(len(g)*frac))), random_state=SEED)
                        for _, g in d.groupby(TARGET)])
               .sample(frac=1, random_state=SEED).reset_index(drop=True))
        print(f"  stratified sample -> {len(d):,} rows")

    le = LabelEncoder()
    y_int = le.fit_transform(d[TARGET].astype(str))
    labels = list(le.classes_)

    _, src_i, dst_i, edge_index, n_nodes = build_graph_tensors(d)

    cent_cols = [c for c in CENTRALITIES if f"src_{c}" in d.columns]
    flow_A = [c for c in SELECTED_TRAFFIC if c in d.columns]
    flow_B = [c for c in MODEL_B_RAW if c in d.columns]

    idx = np.arange(len(d))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE,
                                      stratify=y_int, random_state=SEED)
    tr2_idx, val_idx = train_test_split(tr_idx, test_size=VAL_SIZE,
                                        stratify=y_int[tr_idx], random_state=SEED)

    print("=" * 90)
    print(f"  {TAG.upper()} — GNN  |  {FEATURES_CSV}")
    print(f"  n = {len(d):,} flows, {len(labels)} classes   "
          f"train {len(tr2_idx):,} / val {len(val_idx):,} / test {len(te_idx):,}")
    print(f"  Model A: constant node features + {len(flow_A)} traffic feats")
    print(f"  Model B: {len(cent_cols)} centrality node features "
          f"+ {len(flow_B)} traffic feats")
    print(f"  outputs -> {OUTPUT_ROOT}{os.sep}<MODEL>{os.sep}")
    print("=" * 90)

    all_summaries = {}
    for family in RUN_MODELS:
        fam_sum, fam_pc, fam_cm = {}, [], {}
        for variant, cc, fc in (("A", [], flow_A), ("B", cent_cols, flow_B)):
            torch.manual_seed(SEED); np.random.seed(SEED)
            pc, cm, s = train_eval(family, variant, d, y_int, labels, le,
                                   edge_index, n_nodes, src_i, dst_i,
                                   tr2_idx, val_idx, te_idx, cc, fc)
            fam_pc.append(pc); fam_cm[variant] = cm
            fam_sum[f"{family}-{variant}"] = s
            all_summaries[f"{family}-{variant}"] = s
        save_family(os.path.join(OUTPUT_ROOT, family), family,
                    fam_sum, fam_pc, fam_cm, labels)

    order = [f"{m}-{v}" for m in RUN_MODELS for v in ("A", "B")
             if f"{m}-{v}" in all_summaries]
    print("\n" + "=" * 90)
    print(f"  {TAG.upper()} — GNN RESULTS   (FNR/FPR: LOWER is better)")
    print("=" * 90)
    print(f"  {'model':<14}" + "".join(f"{m:>16}" for m in METRICS))
    print(f"  {'-'*14}" + "".join("-" * 16 for _ in METRICS))
    for nm in order:
        print(f"  {nm:<14}" + "".join(f"{all_summaries[nm][m]:>16.4f}"
                                      for m in METRICS))
    print("=" * 90)

    print("\n  EFFECT OF ADDING CENTRALITIES  (B − A)")
    print(f"  {'family':<14}" + "".join(f"{m:>16}" for m in METRICS))
    print(f"  {'-'*14}" + "".join("-" * 16 for _ in METRICS))
    eff = []
    for fam in RUN_MODELS:
        if f"{fam}-A" in all_summaries and f"{fam}-B" in all_summaries:
            dd = {m: all_summaries[f"{fam}-B"][m] - all_summaries[f"{fam}-A"][m]
                  for m in METRICS}
            print(f"  {fam:<14}" + "".join(f"{dd[m]:>+16.4f}" for m in METRICS))
            eff.append({"family": fam, **dd})
    print("=" * 90)

    pd.DataFrame(all_summaries).T[METRICS].to_csv(
        os.path.join(OUTPUT_ROOT, "all_gnn_summary.csv"), float_format="%.6f")
    pd.DataFrame(eff).to_csv(os.path.join(OUTPUT_ROOT,
                             "gnn_centrality_effect.csv"),
                             index=False, float_format="%.6f")

    fams = [f for f in RUN_MODELS if f"{f}-A" in all_summaries]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="white")
    fig.suptitle(f"{TAG} — GNN: traffic features vs traffic + centralities",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.99)
    x = np.arange(len(fams)); w = 0.36
    for ax, metric in zip(axes.flat, METRICS):
        av = [all_summaries[f"{f}-A"][metric] for f in fams]
        bv = [all_summaries[f"{f}-B"][metric] for f in fams]
        ax.bar(x - w/2, av, w, color=A_C, edgecolor="white", linewidth=1.2,
               zorder=3, label="traffic only")
        ax.bar(x + w/2, bv, w, color=B_C, edgecolor="white", linewidth=1.2,
               zorder=3, label="+ centralities")
        for xi, (p, q) in enumerate(zip(av, bv)):
            ax.text(xi - w/2, p + 0.012, f"{p:.3f}", ha="center", va="bottom",
                    fontsize=9, color=TEXT, fontweight="bold")
            ax.text(xi + w/2, q + 0.012, f"{q:.3f}", ha="center", va="bottom",
                    fontsize=9, color=TEXT, fontweight="bold")
        ax.set_title(metric + (" ↓ lower is better" if metric in LOWER_BETTER else ""),
                     fontsize=12.5, fontweight="bold", color=TEXT, pad=8)
        ax.set_xticks(x); ax.set_xticklabels(fams, fontsize=11, color=TEXT)
        ax.set_ylim(0, max(0.82, max(max(av), max(bv)) * 1.30))
        ax.set_facecolor("#F7F9FB"); ax.tick_params(colors=MUTED, labelsize=10)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]: ax.spines[s].set_color(GRID)
    axes.flat[0].legend(frameon=False, fontsize=10, loc="upper right")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(OUTPUT_ROOT, "all_gnn_comparison.png"),
                dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved: {OUTPUT_ROOT}{os.sep}all_gnn_summary.csv")
    print(f"  saved: {OUTPUT_ROOT}{os.sep}gnn_centrality_effect.csv")
    print(f"  saved: {OUTPUT_ROOT}{os.sep}all_gnn_comparison.png")
    print(f"\n  total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
