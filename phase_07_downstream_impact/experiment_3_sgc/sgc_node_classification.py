"""Phase 7 Experiment 3: SGC node classification on NF-ToN-IoT.

Implements Simple Graph Convolution without PyTorch Geometric:
    X_prop = A_hat^k X
    A_hat = D^(-1/2) (A + I) D^(-1/2)
then trains a linear LogisticRegression classifier with five-fold stratified
cross-validation.

Conditions: global, local, corrected and zero-centrality. The script also
reports the no-propagation baseline, which is useful for the Phase 8 DAP
comparison described in the thesis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEED = 42
METRICS = ("degree", "betweenness", "closeness", "pagerank")


def normalized_adjacency(G: nx.Graph, nodes: list[str]) -> sparse.csr_matrix:
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
    A = sparse.csr_matrix(A)
    A = A + sparse.eye(A.shape[0], format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(deg, dtype=float)
    nz = deg > 0
    inv[nz] = 1.0 / np.sqrt(deg[nz])
    D = sparse.diags(inv)
    return (D @ A @ D).tocsr()


def propagate(Ahat: sparse.csr_matrix, X: np.ndarray, hops: int) -> np.ndarray:
    Z = X.astype(float)
    for _ in range(hops):
        Z = Ahat @ Z
    return np.asarray(Z)


def cv_score(X: np.ndarray, y: np.ndarray, folds: int = 5) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    f1s, p10s = [], []
    for train, test in cv.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train])
        Xte = scaler.transform(X[test])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        clf.fit(Xtr, y[train])
        pred = clf.predict(Xte)
        prob = clf.predict_proba(Xte)[:, 1]
        f1s.append(f1_score(y[test], pred, average="macro"))
        k = min(10, len(test))
        order = np.argsort(-prob)[:k]
        p10s.append(float(y[test][order].mean()) if k else np.nan)
    return float(np.mean(f1s)), float(np.nanmean(p10s))


def main() -> None:
    p = argparse.ArgumentParser(description="SGC node classification")
    p.add_argument("flows", help="Raw flow CSV used to build graph and node labels")
    p.add_argument("centralities", help="Node centrality table")
    p.add_argument("--src-col", default="IPV4_SRC_ADDR")
    p.add_argument("--dst-col", default="IPV4_DST_ADDR")
    p.add_argument("--label-col", default="Label")
    p.add_argument("--hops", type=int, default=1)
    p.add_argument("--output", default="sgc_results.csv")
    args = p.parse_args()

    flows = pd.read_csv(args.flows, low_memory=False)
    flows[args.src_col] = flows[args.src_col].astype(str)
    flows[args.dst_col] = flows[args.dst_col].astype(str)
    cent = pd.read_csv(args.centralities)
    cent["node"] = cent["node"].astype(str)

    G = nx.Graph()
    edges = flows[[args.src_col, args.dst_col]].drop_duplicates()
    G.add_edges_from(edges.itertuples(index=False, name=None))

    atk = flows[flows[args.label_col] == 1]
    attack_nodes = set(atk[args.src_col].astype(str)) | set(atk[args.dst_col].astype(str))

    common = sorted(set(G.nodes()) & set(cent["node"]), key=str)
    cent = cent.set_index("node").loc[common]
    y = np.array([1 if n in attack_nodes else 0 for n in common], dtype=int)
    Ahat = normalized_adjacency(G.subgraph(common).copy(), common)

    rows = []
    for condition in ("global", "local", "corrected", "zero"):
        if condition == "zero":
            X = np.zeros((len(common), len(METRICS)), dtype=float)
        else:
            cols = [f"{condition}_{m}" for m in METRICS]
            missing = [c for c in cols if c not in cent.columns]
            if missing:
                print(f"Skipping {condition}: missing {missing}")
                continue
            X = cent[cols].fillna(0.0).to_numpy(float)

        f1_raw, p10_raw = cv_score(X, y)
        Xp = propagate(Ahat, X, args.hops)
        f1_prop, p10_prop = cv_score(Xp, y)
        rows.append({
            "condition": condition,
            "macro_f1_no_propagation": f1_raw,
            "macro_f1_with_propagation": f1_prop,
            "change": f1_prop - f1_raw,
            "precision_at_10_no_propagation": p10_raw,
            "precision_at_10_with_propagation": p10_prop,
            "hops": args.hops,
        })

    res = pd.DataFrame(rows)
    res.to_csv(args.output, index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
