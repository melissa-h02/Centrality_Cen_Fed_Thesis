"""Phase 7 Experiment 2: top-K threat-node ranking and attack-flow coverage.

The ranking itself uses centrality only; labels are used afterwards solely to
measure operational coverage.

Metrics:
- Precision@K: overlap of a local/corrected top-K with the true global top-K.
- Attack coverage@K: percentage of attack flows caught when monitoring only
  the selected K monitored nodes (source OR destination endpoint).

The report's reference result for NF-ToN-IoT PageRank at K=10 is 66.0% local
versus 92.3% corrected. This script computes that quantity from the supplied
centrality table and raw flow data; it does not hard-code the numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Set

import numpy as np
import pandas as pd

METRICS = ("degree", "closeness", "pagerank")


def topk(df: pd.DataFrame, value_col: str, k: int) -> Set[str]:
    x = df[["node", value_col]].dropna().copy()
    x["node"] = x["node"].astype(str)
    x = x.sort_values([value_col, "node"], ascending=[False, True])
    return set(x.head(k)["node"])


def attack_coverage(df: pd.DataFrame, nodes: Set[str], src_col: str,
                    label_col: str, count_receiver: bool = True,
                    dst_col: str | None = None) -> float:
    atk = df[df[label_col] == 1]
    if len(atk) == 0:
        return 0.0
    hit = atk[src_col].astype(str).isin(nodes)
    if count_receiver:
        if not dst_col:
            raise ValueError("--dst-col is required with --count-receiver")
        hit |= atk[dst_col].astype(str).isin(nodes)
    return 100.0 * float(hit.mean())


def main() -> None:
    p = argparse.ArgumentParser(description="Top-K threat ranking and attack-flow coverage")
    p.add_argument("flows", help="Raw flow CSV")
    p.add_argument("centralities", help="Node centrality CSV")
    p.add_argument("--src-col", default="IPV4_SRC_ADDR")
    p.add_argument("--dst-col", default="IPV4_DST_ADDR")
    p.add_argument("--label-col", default="Label")
    p.add_argument("--k", default="5,10,20,50")
    p.add_argument("--count-receiver", dest="count_receiver", action="store_true", default=True,
                   help="Count an attack flow if either endpoint is monitored (default; thesis definition)")
    p.add_argument("--source-only", dest="count_receiver", action="store_false",
                   help="Supplementary mode: count only attack flows whose source is monitored")
    p.add_argument("--output", default="topk_attack_coverage.csv")
    args = p.parse_args()

    flows = pd.read_csv(args.flows, low_memory=False)
    cent = pd.read_csv(args.centralities)
    flows[args.src_col] = flows[args.src_col].astype(str)
    if args.dst_col in flows.columns:
        flows[args.dst_col] = flows[args.dst_col].astype(str)
    cent["node"] = cent["node"].astype(str)

    ks = [int(v) for v in args.k.split(",") if v.strip()]
    rows = []
    for metric in METRICS:
        cols = {c: f"{c}_{metric}" for c in ("global", "local", "corrected")}
        if not all(v in cent.columns for v in cols.values()):
            print(f"Skipping {metric}: need {list(cols.values())}")
            continue
        for k in ks:
            global_set = topk(cent, cols["global"], k)
            for condition in ("global", "local", "corrected"):
                picked = topk(cent, cols[condition], k)
                overlap = len(picked & global_set) / max(k, 1)
                coverage = attack_coverage(
                    flows, picked, args.src_col, args.label_col,
                    args.count_receiver, args.dst_col
                )
                rows.append({
                    "metric": metric,
                    "K": k,
                    "condition": condition,
                    "precision_at_k_vs_global": overlap,
                    "attack_coverage_pct": coverage,
                    "n_selected": len(picked),
                })

    res = pd.DataFrame(rows)
    res.to_csv(args.output, index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
