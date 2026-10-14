"""
═══════════════════════════════════════════════════════════════════════════
 RUN ALL MODELS — Random Forest, LSTM, CNN   (traffic vs traffic+centralities)
═══════════════════════════════════════════════════════════════════════════

 CENTRALISED training on a partitioned feature file. This answers the
 question "do the centralities improve the model?" without yet introducing
 federated training — one model is trained on all flows; only the CENTRALITY
 VALUES come from the per-client subgraphs.

   MODEL A  traffic only        SELECTED_TRAFFIC (9 features)
   MODEL B  + centralities      src/dst centralities + MODEL_B_RAW

 Point FEATURES_CSV at:
   edge_features_global.csv          -> Step 1 (global centralities)
   edge_features_local_3client.csv   -> Step 2 (local centralities)
   (later) a corrected-centrality file -> Step 3
 Everything else stays identical, so the steps stay comparable.

 Every model uses the SAME rows and the SAME stratified 70/30 split
 (test_size=0.30, seed=42), and the same metrics:
   Accuracy, Macro Accuracy (balanced), Macro F1, Weighted F1, FNR, FPR

 ── OUTPUT LAYOUT ─────────────────────────────────────────────────────────
   <OUTPUT_ROOT>/
       RF/     RF_summary.csv, RF_perclass.csv,
               RF_confusion_A.csv, RF_confusion_B.csv, RF_comparison.png
       LSTM/   (same files)
       CNN/    (same files)
       all_models_summary.csv
       centrality_effect.csv
       all_models_comparison.png

 ── TIME ─────────────────────────────────────────────────────────────────
 SAMPLE_N = None runs the full ~1.2M flows: roughly 20 min (RF) + 45 min
 (LSTM) + 20 min (CNN). Set SAMPLE_N = 300_000 for a fast pass, and
 RUN_MODELS = ["RF"] to test the pipeline first.

 Run: python run_all_models.py
═══════════════════════════════════════════════════════════════════════════
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────
FEATURES_CSV = "edge_features_corrected_3client.csv"
TAG          = "step3_corrected"
OUTPUT_ROOT  = "results_step3_corrected"
TARGET       = "Attack"
SEED         = 42
TEST_SIZE, VAL_SIZE = 0.30, 0.15
SAMPLE_N     = None                   # None = full data
RUN_MODELS   = ["RF", "LSTM", "CNN"]

EPOCHS, BATCH, PATIENCE = 30, 512, 5
N_TREES, MAX_DEPTH, MIN_LEAF, N_JOBS = 200, 25, 5, 2

SELECTED_TRAFFIC = ['L4_DST_PORT', 'OUT_BYTES', 'IN_BYTES', 'TCP_FLAGS',
                    'L4_SRC_PORT', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS',
                    'IN_PKTS', 'L7_PROTO']
CENTRALITIES = ["degree_undir", "betweenness", "closeness", "pagerank"]
MODEL_B_RAW = ["L4_DST_PORT", "OUT_BYTES", "IN_BYTES", "TCP_FLAGS", "L4_SRC_PORT"]

METRICS = ["Accuracy", "Macro Accuracy", "Macro F1", "Weighted F1", "FNR", "FPR"]
LOWER_BETTER = {"FNR", "FPR"}
TEXT, MUTED, GRID, A_C, B_C = "#1A2330", "#5B6B7A", "#E0E6EC", "#90A4AE", "#2E7D32"

np.random.seed(SEED)


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


def build_dl(kind, n_features, n_classes):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (LSTM, Conv1D, BatchNormalization, Dense,
                                         Dropout, GlobalMaxPooling1D, Input)
    if kind == "LSTM":
        m = Sequential([Input(shape=(n_features, 1)),
                        LSTM(64), Dropout(0.3),
                        Dense(32, activation="relu"), Dropout(0.2),
                        Dense(n_classes, activation="softmax")])
    else:
        m = Sequential([Input(shape=(n_features, 1)),
                        Conv1D(64, 3, padding="same", activation="relu"),
                        BatchNormalization(),
                        Conv1D(128, 3, padding="same", activation="relu"),
                        GlobalMaxPooling1D(),
                        Dense(64, activation="relu"), Dropout(0.3),
                        Dense(n_classes, activation="softmax")])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def save_model_outputs(folder, family, res_by_variant, per_class, confusions, labels):
    """Write this model family's own CSVs and A-vs-B chart into its folder."""
    os.makedirs(folder, exist_ok=True)

    pd.DataFrame(res_by_variant).T[METRICS].to_csv(
        os.path.join(folder, f"{family}_summary.csv"), float_format="%.6f")
    pd.concat(per_class).to_csv(
        os.path.join(folder, f"{family}_perclass.csv"),
        index=False, float_format="%.6f")
    for variant, cm in confusions.items():
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            os.path.join(folder, f"{family}_confusion_{variant}.csv"))

    a_key, b_key = f"{family}-A", f"{family}-B"
    if a_key in res_by_variant and b_key in res_by_variant:
        av = [res_by_variant[a_key][m] for m in METRICS]
        bv = [res_by_variant[b_key][m] for m in METRICS]
        x = np.arange(len(METRICS)); w = 0.36
        fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
        fig.suptitle(f"{family} — traffic only vs + centralities   "
                     f"[FNR/FPR: lower is better]",
                     fontsize=14, fontweight="bold", color=TEXT, y=0.97)
        ax.bar(x - w/2, av, w, color=A_C, edgecolor="white", linewidth=1.2,
               zorder=3, label="A: traffic only")
        ax.bar(x + w/2, bv, w, color=B_C, edgecolor="white", linewidth=1.2,
               zorder=3, label="B: + centralities")
        for xi, (a, b) in enumerate(zip(av, bv)):
            ax.text(xi - w/2, a + 0.012, f"{a:.3f}", ha="center", va="bottom",
                    fontsize=9.5, color=TEXT, fontweight="bold")
            ax.text(xi + w/2, b + 0.012, f"{b:.3f}", ha="center", va="bottom",
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
        print(f"  stratified sample -> {len(d):,} rows (real proportions kept)")

    le = LabelEncoder()
    y_int = le.fit_transform(d[TARGET].astype(str))
    labels = list(le.classes_)

    cent = [f"src_{c}" for c in CENTRALITIES if f"src_{c}" in d.columns] + \
           [f"dst_{c}" for c in CENTRALITIES if f"dst_{c}" in d.columns]
    feat_sets = {"A": [c for c in SELECTED_TRAFFIC if c in d.columns],
                 "B": cent + [c for c in MODEL_B_RAW if c in d.columns]}

    idx = np.arange(len(d))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_SIZE,
                                      stratify=y_int, random_state=SEED)
    tr2_idx, val_idx = train_test_split(tr_idx, test_size=VAL_SIZE,
                                        stratify=y_int[tr_idx], random_state=SEED)
    y_te_lab = le.inverse_transform(y_int[te_idx])

    print("=" * 90)
    print(f"  {TAG.upper()}  |  {FEATURES_CSV}")
    print(f"  n = {len(d):,} flows, {len(labels)} classes   "
          f"train {len(tr_idx):,} / test {len(te_idx):,}")
    print(f"  A = {len(feat_sets['A'])} traffic feats   "
          f"B = {len(feat_sets['B'])} feats ({len(cent)} centrality + few traffic)")
    print(f"  outputs -> {OUTPUT_ROOT}{os.sep}<MODEL>{os.sep}")
    print("=" * 90)

    all_summaries = {}

    for family in RUN_MODELS:
        fam_summaries, fam_perclass, fam_conf = {}, [], {}

        for variant, feats in feat_sets.items():
            name = f"{family}-{variant}"
            X = d[feats].to_numpy(float)
            t0 = time.time()
            print(f"\n  === {name}: {len(feats)} features ===")

            if family == "RF":
                model = RandomForestClassifier(
                    n_estimators=N_TREES, max_depth=MAX_DEPTH,
                    min_samples_leaf=MIN_LEAF, class_weight=None,
                    random_state=SEED, n_jobs=N_JOBS).fit(X[tr_idx], y_int[tr_idx])
                pred = le.inverse_transform(model.predict(X[te_idx]))
            else:
                import tensorflow as tf
                from tensorflow.keras.callbacks import EarlyStopping
                tf.random.set_seed(SEED)
                sc = StandardScaler().fit(X[tr2_idx])
                Xtr = sc.transform(X[tr2_idx])[..., None]
                Xval = sc.transform(X[val_idx])[..., None]
                Xte = sc.transform(X[te_idx])[..., None]
                model = build_dl(family, len(feats), len(labels))
                es = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                   restore_best_weights=True)
                model.fit(Xtr, y_int[tr2_idx],
                          validation_data=(Xval, y_int[val_idx]),
                          epochs=EPOCHS, batch_size=BATCH,
                          callbacks=[es], verbose=2)
                pred = le.inverse_transform(
                    model.predict(Xte, batch_size=BATCH, verbose=0).argmax(axis=1))

            pc, cm = per_class_metrics(y_te_lab, pred, labels)
            pc.insert(0, "model", name)
            fam_perclass.append(pc)
            fam_conf[variant] = cm
            s = summarize(pc, y_te_lab, pred)
            fam_summaries[name] = s
            all_summaries[name] = s
            report(name, pc, time.time() - t0)

        save_model_outputs(os.path.join(OUTPUT_ROOT, family), family,
                           fam_summaries, fam_perclass, fam_conf, labels)

    # ── combined outputs at the root ──────────────────────────────────
    order = [f"{m}-{v}" for m in RUN_MODELS for v in ("A", "B")
             if f"{m}-{v}" in all_summaries]
    print("\n" + "=" * 90)
    print(f"  {TAG.upper()} — ALL MODELS   (FNR/FPR: LOWER is better)")
    print("=" * 90)
    print(f"  {'model':<10}" + "".join(f"{m:>16}" for m in METRICS))
    print(f"  {'-'*10}" + "".join("-" * 16 for _ in METRICS))
    for nm in order:
        print(f"  {nm:<10}" + "".join(f"{all_summaries[nm][m]:>16.4f}" for m in METRICS))
    print("=" * 90)

    print("\n  EFFECT OF ADDING CENTRALITIES  (B − A)")
    print(f"  {'family':<10}" + "".join(f"{m:>16}" for m in METRICS))
    print(f"  {'-'*10}" + "".join("-" * 16 for _ in METRICS))
    eff_rows = []
    for fam in RUN_MODELS:
        if f"{fam}-A" in all_summaries and f"{fam}-B" in all_summaries:
            deltas = {m: all_summaries[f"{fam}-B"][m] - all_summaries[f"{fam}-A"][m]
                      for m in METRICS}
            print(f"  {fam:<10}" + "".join(f"{deltas[m]:>+16.4f}" for m in METRICS))
            eff_rows.append({"family": fam, **deltas})
    print("=" * 90)

    pd.DataFrame(all_summaries).T[METRICS].to_csv(
        os.path.join(OUTPUT_ROOT, "all_models_summary.csv"), float_format="%.6f")
    pd.DataFrame(eff_rows).to_csv(
        os.path.join(OUTPUT_ROOT, "centrality_effect.csv"),
        index=False, float_format="%.6f")

    fams = [f for f in RUN_MODELS if f"{f}-A" in all_summaries]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="white")
    fig.suptitle(f"{TAG} — traffic features vs traffic + centralities",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.99)
    x = np.arange(len(fams)); w = 0.36
    for ax, metric in zip(axes.flat, METRICS):
        av = [all_summaries[f"{f}-A"][metric] for f in fams]
        bv = [all_summaries[f"{f}-B"][metric] for f in fams]
        ax.bar(x - w/2, av, w, color=A_C, edgecolor="white", linewidth=1.2,
               zorder=3, label="traffic only")
        ax.bar(x + w/2, bv, w, color=B_C, edgecolor="white", linewidth=1.2,
               zorder=3, label="+ centralities")
        for xi, (a, b) in enumerate(zip(av, bv)):
            ax.text(xi - w/2, a + 0.012, f"{a:.3f}", ha="center", va="bottom",
                    fontsize=9, color=TEXT, fontweight="bold")
            ax.text(xi + w/2, b + 0.012, f"{b:.3f}", ha="center", va="bottom",
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
    plt.savefig(os.path.join(OUTPUT_ROOT, "all_models_comparison.png"),
                dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  saved: {OUTPUT_ROOT}{os.sep}all_models_summary.csv")
    print(f"  saved: {OUTPUT_ROOT}{os.sep}centrality_effect.csv")
    print(f"  saved: {OUTPUT_ROOT}{os.sep}all_models_comparison.png")
    print(f"\n  total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()