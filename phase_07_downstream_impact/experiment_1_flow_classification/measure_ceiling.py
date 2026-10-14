"""
═══════════════════════════════════════════════════════════════════════════
 HOW GOOD COULD ANY MODEL POSSIBLY BE?  (the dataset's hard ceiling)
═══════════════════════════════════════════════════════════════════════════

 WHY THIS SCRIPT EXISTS
   Our classifiers reached about 0.70 accuracy and stalled, and several
   attack types stayed at 0.000 F1 no matter what we did. Before blaming the
   models or the features, we should ask a simpler question:

       Given these features, what is the BEST score ANY model could get?

 THE IDEA (no machine learning involved)
   Group the flows by their feature values. If two flows have EXACTLY the
   same features but different attack labels, no classifier on earth can
   tell them apart — it must output one answer for both, so it is guaranteed
   to get some of them wrong.
   The best any model can do for such a group is answer with that group's
   most common label. Adding this up over every group gives the maximum
   achievable accuracy: the CEILING.

 WHAT IT REPORTS
   • how many flows sit in "conflicting" groups (same features, >1 label)
   • the ceiling accuracy for MODEL A features (traffic only)
   • the ceiling accuracy for MODEL B features (traffic + centralities)
   • the macro F1 of that best-possible predictor
   • per class: the best recall any model could reach

   Compare the ceiling to what our models actually scored. If they are close,
   the models are already near-perfect and the DATASET is the limitation.

 Run:  python measure_ceiling.py
       (point FEATURES_CSV at any of the step feature files)
═══════════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# ── Config ──────────────────────────────────────────────────────────────
FEATURES_CSV = "edge_features_global.csv"
TARGET = "Attack"

SELECTED_TRAFFIC = ['L4_DST_PORT', 'OUT_BYTES', 'IN_BYTES', 'TCP_FLAGS',
                    'L4_SRC_PORT', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS',
                    'IN_PKTS', 'L7_PROTO']
CENTRALITIES = ["degree_undir", "betweenness", "closeness", "pagerank"]
MODEL_B_RAW = ["L4_DST_PORT", "OUT_BYTES", "IN_BYTES", "TCP_FLAGS", "L4_SRC_PORT"]

OUT_CSV = "ceiling_analysis.csv"


def ceiling_for(d, feats, name):
    """Best accuracy any deterministic model could reach with these features."""
    n = len(d)
    grp = d.groupby(feats, sort=False)[TARGET]

    # rows whose feature-tuple carries more than one distinct label
    n_labels = grp.transform("nunique")
    conflicting = int((n_labels > 1).sum())

    # best possible prediction for each group = its most common label
    best_pred = grp.transform(lambda s: s.mode().iat[0])
    ceiling_acc = float((best_pred == d[TARGET]).mean())

    P, R, F1, sup = precision_recall_fscore_support(
        d[TARGET], best_pred, zero_division=0)
    labels = sorted(d[TARGET].unique())
    macro_f1 = float(np.mean(F1))

    print(f"\n  {name}")
    print(f"     features used            : {len(feats)}")
    print(f"     distinct feature-tuples  : {grp.ngroups:,}")
    print(f"     flows in conflicting sets: {conflicting:,} "
          f"({100*conflicting/n:.1f}% of all flows)")
    print(f"     -> CEILING ACCURACY      : {ceiling_acc:.4f}")
    print(f"     -> ceiling macro F1      : {macro_f1:.4f}")

    per_class = pd.DataFrame({"class": labels, "best_possible_recall": R,
                              "best_possible_f1": F1, "support": sup})
    return {"feature_set": name, "n_features": len(feats),
            "distinct_tuples": grp.ngroups,
            "conflicting_flows": conflicting,
            "conflicting_pct": round(100 * conflicting / n, 2),
            "ceiling_accuracy": round(ceiling_acc, 4),
            "ceiling_macro_f1": round(macro_f1, 4)}, per_class


def main():
    if not os.path.exists(FEATURES_CSV):
        print(f"Could not find {FEATURES_CSV} in this folder.")
        return

    print(f"Loading {FEATURES_CSV} ...")
    d = pd.read_csv(FEATURES_CSV)
    print(f"  {len(d):,} flows, {d[TARGET].nunique()} attack classes")

    feat_A = [c for c in SELECTED_TRAFFIC if c in d.columns]
    cent = [f"src_{c}" for c in CENTRALITIES if f"src_{c}" in d.columns] + \
           [f"dst_{c}" for c in CENTRALITIES if f"dst_{c}" in d.columns]
    feat_B = cent + [c for c in MODEL_B_RAW if c in d.columns]

    print("\n" + "=" * 78)
    print("  THE BEST SCORE ANY MODEL COULD POSSIBLY REACH")
    print("  (no learning — just: answer each identical-feature group with")
    print("   its most common label, which is the best that is possible)")
    print("=" * 78)

    rows, per_class_all = [], []
    r, pc = ceiling_for(d, feat_A, "MODEL A  — traffic features only")
    rows.append(r); pc.insert(0, "feature_set", "A"); per_class_all.append(pc)

    if cent:
        r, pc = ceiling_for(d, feat_B, "MODEL B  — traffic + centralities")
        rows.append(r); pc.insert(0, "feature_set", "B"); per_class_all.append(pc)

    print("\n" + "=" * 78)
    print("  PER-CLASS: the best recall ANY model could reach")
    print("=" * 78)
    print(f"  {'class':<14}{'A: best recall':>16}{'B: best recall':>18}"
          f"{'support':>10}")
    print(f"  {'-'*14}{'-'*16}{'-'*18}{'-'*10}")
    a_pc = per_class_all[0].set_index("class")
    b_pc = per_class_all[1].set_index("class") if len(per_class_all) > 1 else None
    for cls in a_pc.index:
        bval = b_pc.loc[cls, "best_possible_recall"] if b_pc is not None else np.nan
        print(f"  {str(cls):<14}{a_pc.loc[cls,'best_possible_recall']:>16.3f}"
              f"{bval:>18.3f}{int(a_pc.loc[cls,'support']):>10,}")
    print("=" * 78)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    pd.concat(per_class_all).to_csv("ceiling_per_class.csv", index=False,
                                    float_format="%.6f")
    print(f"  saved: {OUT_CSV}, ceiling_per_class.csv")

    print("\n  HOW TO READ THIS")
    print("   - Ceiling accuracy is the score a PERFECT model would get.")
    print("   - If our models scored close to it, they are already near")
    print("     optimal and the dataset — not the method — is the limit.")
    print("   - A class whose best possible recall is low can NEVER be")
    print("     detected well from these features, by any model.")


if __name__ == "__main__":
    main()
