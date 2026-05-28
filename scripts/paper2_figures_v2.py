#!/usr/bin/env python3
"""
Paper 2 — Regenerate Fig 3 and Fig 5 with GroupKFold metrics.
Fixes: Figure/text metric mismatch.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
import warnings
warnings.filterwarnings("ignore")

from config.settings import DATA_DIR

COMBO_DIR = DATA_DIR / "combination"
FIG_DIR = Path(__file__).parent.parent / "paper2" / "figures"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def get_gene_groups(df):
    all_genes = sorted(set(df["target_A"]) | set(df["target_B"]))
    gene_to_id = {g: i for i, g in enumerate(all_genes)}
    groups = []
    for _, row in df.iterrows():
        g = min(row["target_A"], row["target_B"])
        groups.append(gene_to_id[g])
    return np.array(groups)


def fig3_ml_performance_groupkfold():
    """Fig 3 with GroupKFold metrics."""
    X = pd.read_csv(COMBO_DIR / "feature_matrix.csv")
    df = pd.read_csv(COMBO_DIR / "curated_combinations.csv")
    y_binary = (df["interaction"] == "synergistic").astype(int).values
    groups = get_gene_groups(df)
    gkf = GroupKFold(n_splits=5)

    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1])

    # Panel A: ROC curves (GroupKFold)
    ax = fig.add_subplot(gs[0])
    for name, clf_factory, color, ls in [
        ("Random Forest", lambda: RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=2,
            class_weight="balanced", random_state=42), "#2196F3", "-"),
        ("Gradient Boosting", lambda: GradientBoostingClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=2,
            random_state=42), "#F44336", "--"),
    ]:
        y_prob = np.zeros(len(y_binary), dtype=float)
        y_pred = np.zeros_like(y_binary)
        for train_idx, test_idx in gkf.split(X, y_binary, groups):
            clf = clf_factory()
            clf.fit(X.iloc[train_idx], y_binary[train_idx])
            y_pred[test_idx] = clf.predict(X.iloc[test_idx])
            y_prob[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]

        fpr, tpr, _ = roc_curve(y_binary, y_prob)
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(y_binary, y_pred)
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2,
                label=f"{name}\n(AUC={roc_auc:.3f}, F1={f1:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("A. ROC Curves (GroupKFold)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Panel B: Confusion matrix (GB, GroupKFold, binary)
    ax = fig.add_subplot(gs[1])
    y_pred_gb = np.zeros_like(y_binary)
    for train_idx, test_idx in gkf.split(X, y_binary, groups):
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=2, random_state=42)
        clf.fit(X.iloc[train_idx], y_binary[train_idx])
        y_pred_gb[test_idx] = clf.predict(X.iloc[test_idx])

    cm = confusion_matrix(y_binary, y_pred_gb)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Syn", "Synergistic"], fontsize=10)
    ax.set_yticklabels(["Non-Syn", "Synergistic"], fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    f1_gb = f1_score(y_binary, y_pred_gb)
    ax.set_title(f"B. Confusion Matrix (GB, F1={f1_gb:.3f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > 10 else "black")

    # Panel C: Permutation test (GroupKFold)
    ax = fig.add_subplot(gs[2])
    our_f1 = f1_gb

    n_mc = 1000
    random_f1s = []
    for seed in range(n_mc):
        rng = np.random.RandomState(seed)
        y_perm = rng.permutation(y_binary)
        y_pred_perm = np.zeros_like(y_binary)
        for train_idx, test_idx in gkf.split(X, y_perm, groups):
            if len(set(y_perm[train_idx])) < 2:
                continue
            clf = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, min_samples_leaf=2, random_state=seed)
            clf.fit(X.iloc[train_idx], y_perm[train_idx])
            y_pred_perm[test_idx] = clf.predict(X.iloc[test_idx])
        random_f1s.append(f1_score(y_perm, y_pred_perm, zero_division=0))
    random_f1s = np.array(random_f1s)

    z = (our_f1 - random_f1s.mean()) / (random_f1s.std() + 1e-10)
    p = (random_f1s >= our_f1).mean()

    ax.hist(random_f1s, bins=30, color="#BDBDBD", edgecolor="white", alpha=0.8,
            label=f"Random (mean={random_f1s.mean():.3f})")
    ax.axvline(our_f1, color="#F44336", linewidth=2.5, linestyle="-",
               label=f"Our Model (F1={our_f1:.3f})")
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Count")
    ax.set_title(f"C. Permutation Test (z={z:.2f}, p={p:.4f})")
    ax.legend(fontsize=9)

    fig.suptitle("Figure 3. ML Synergy Prediction (Gene-Level GroupKFold CV)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_ml_performance.png")
    fig.savefig(FIG_DIR / "fig3_ml_performance.pdf")
    print(f"  Fig 3 saved (GroupKFold: F1={our_f1:.3f}, z={z:.2f}, p={p:.4f})")
    plt.close()


def fig5_regression_groupkfold():
    """Fig 5 with GroupKFold regression."""
    X = pd.read_csv(COMBO_DIR / "feature_matrix.csv")
    df = pd.read_csv(COMBO_DIR / "curated_combinations.csv")
    y_bliss = df["bliss_score"].values
    groups = get_gene_groups(df)
    gkf = GroupKFold(n_splits=5)

    y_pred = np.zeros_like(y_bliss)
    for train_idx, test_idx in gkf.split(X, y_bliss, groups):
        reg = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=42)
        reg.fit(X.iloc[train_idx], y_bliss[train_idx])
        y_pred[test_idx] = reg.predict(X.iloc[test_idx])

    from sklearn.metrics import r2_score
    r2 = r2_score(y_bliss, y_pred)
    corr = np.corrcoef(y_bliss, y_pred)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 7))

    for itype, color, marker in [
        ("synergistic", "#2196F3", "o"),
        ("additive", "#9E9E9E", "s"),
        ("antagonistic", "#F44336", "^"),
    ]:
        mask = df["interaction"] == itype
        ax.scatter(y_bliss[mask], y_pred[mask], c=color, marker=marker,
                   s=80, alpha=0.8, edgecolors="white", linewidth=0.5,
                   label=f"{itype} (n={mask.sum()})")

    lims = [-0.7, 1.0]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.3)

    ax.set_xlabel("Observed Bliss Score (Literature)")
    ax.set_ylabel("Predicted Bliss Score (GroupKFold CV)")
    ax.set_title(f"Figure 5. Bliss Score Prediction (GroupKFold)\n"
                 f"R² = {r2:.3f}, Pearson r = {corr:.3f}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_regression_scatter.png")
    fig.savefig(FIG_DIR / "fig5_regression_scatter.pdf")
    print(f"  Fig 5 saved (GroupKFold: R²={r2:.3f}, r={corr:.3f})")
    plt.close()


def main():
    print("=" * 60)
    print("  PAPER 2 — FIGURE REGENERATION (GroupKFold)")
    print("=" * 60)
    fig3_ml_performance_groupkfold()
    fig5_regression_groupkfold()
    print("=" * 60)


if __name__ == "__main__":
    main()
