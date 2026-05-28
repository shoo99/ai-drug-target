#!/usr/bin/env python3
"""
Paper 2 — Generate all figures.

Fig 1: FBA LP threshold behavior (negative result visualization)
Fig 2: Curated combination dataset overview (pathway × pathway heatmap)
Fig 3: ML model performance (ROC + confusion matrix)
Fig 4: Feature importance (top 20 horizontal bar)
Fig 5: Bliss score prediction scatter (observed vs predicted)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import roc_curve, auc, confusion_matrix, r2_score

from config.settings import DATA_DIR

COMBO_DIR = DATA_DIR / "combination"
FIG_DIR = Path(__file__).parent.parent / "paper2" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "synergistic": "#2196F3",
    "antagonistic": "#F44336",
    "additive": "#9E9E9E",
    "accent": "#FF9800",
    "dark": "#212121",
}


def fig1_lp_threshold():
    """Fig 1: FBA LP threshold behavior — why synergy detection fails."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: Single gene dose-response (step function)
    ax = axes[0]
    x = np.linspace(0, 1, 100)
    # FBA: step function at threshold=1.0
    y_fba = np.where(x < 0.99, 1.0, 0.0)
    # Real biology: sigmoid
    y_real = 1.0 / (1.0 + np.exp(15 * (x - 0.5)))

    ax.plot(x, y_fba, "b-", linewidth=2.5, label="FBA (LP optimal)")
    ax.plot(x, y_real, "r--", linewidth=2, label="Biological (expected)")
    ax.fill_between(x, y_fba, y_real, alpha=0.15, color="red")
    ax.set_xlabel("Gene Inhibition Level")
    ax.set_ylabel("Growth Ratio")
    ax.set_title("A. Single-Gene Dose Response")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.1)
    ax.annotate("LP reroutes\nmetabolism\nperfectly",
                xy=(0.5, 0.95), fontsize=9, ha="center", color="blue",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow"))

    # Panel B: Double gene — FBA gives no synergy info
    ax = axes[1]
    # Heatmap: FBA double inhibition (all 1.0 except edges)
    grid_size = 15
    inhib = np.linspace(0, 1, grid_size)
    fba_grid = np.ones((grid_size, grid_size))
    fba_grid[-1, :] = 0  # gene A = 100% → dead
    fba_grid[:, -1] = 0  # gene B = 100% → dead
    fba_grid[-1, -1] = 0

    im = ax.imshow(fba_grid, cmap="RdYlGn", origin="lower",
                   extent=[0, 1, 0, 1], vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("Gene A Inhibition")
    ax.set_ylabel("Gene B Inhibition")
    ax.set_title("B. FBA Double Inhibition (Observed)")
    ax.annotate("Growth = 1.0\n(LP optimal)",
                xy=(0.4, 0.4), fontsize=10, ha="center", color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
    ax.annotate("Dead", xy=(0.5, 0.97), fontsize=9, ha="center", color="white",
                fontweight="bold")
    ax.annotate("Dead", xy=(0.97, 0.5), fontsize=9, ha="center", color="white",
                fontweight="bold", rotation=90)
    plt.colorbar(im, ax=ax, label="Growth Ratio", shrink=0.8)

    # Panel C: What biology should look like (synergy visible)
    ax = axes[2]
    xx, yy = np.meshgrid(inhib, inhib)
    # Simulated biological response with synergy
    bio_grid = (1 / (1 + np.exp(12 * (xx - 0.5)))) * (1 / (1 + np.exp(12 * (yy - 0.5))))
    # Add synergy: combo is worse than expected
    synergy_effect = -0.3 * np.exp(-((xx - 0.4)**2 + (yy - 0.4)**2) / 0.05)
    bio_grid = np.clip(bio_grid + synergy_effect, 0, 1)

    im = ax.imshow(bio_grid, cmap="RdYlGn", origin="lower",
                   extent=[0, 1, 0, 1], vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("Gene A Inhibition")
    ax.set_ylabel("Gene B Inhibition")
    ax.set_title("C. Biological (Expected with Synergy)")
    ax.annotate("Synergy\nregion",
                xy=(0.4, 0.4), fontsize=10, ha="center", color="white",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="red", alpha=0.7))
    plt.colorbar(im, ax=ax, label="Growth Ratio", shrink=0.8)

    fig.suptitle("Figure 1. FBA Cannot Detect Drug Combination Synergy Due to LP Optimality",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_lp_threshold.png")
    fig.savefig(FIG_DIR / "fig1_lp_threshold.pdf")
    print("  Fig 1 saved")
    plt.close()


def fig2_dataset_heatmap():
    """Fig 2: Pathway × pathway interaction heatmap from curated data."""
    df = pd.read_csv(COMBO_DIR / "curated_combinations.csv")

    pathways = sorted(set(df["pathway_A"]) | set(df["pathway_B"]))
    n = len(pathways)
    pw_idx = {p: i for i, p in enumerate(pathways)}

    # Count matrix and mean Bliss matrix
    count_mat = np.zeros((n, n))
    bliss_mat = np.full((n, n), np.nan)
    bliss_sums = np.zeros((n, n))
    bliss_counts = np.zeros((n, n))

    for _, row in df.iterrows():
        i, j = pw_idx[row["pathway_A"]], pw_idx[row["pathway_B"]]
        count_mat[i, j] += 1
        count_mat[j, i] += 1
        bliss_sums[i, j] += row["bliss_score"]
        bliss_sums[j, i] += row["bliss_score"]
        bliss_counts[i, j] += 1
        bliss_counts[j, i] += 1

    mask = bliss_counts > 0
    bliss_mat[mask] = bliss_sums[mask] / bliss_counts[mask]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Panel A: Count heatmap
    ax = axes[0]
    im = ax.imshow(count_mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(pathways, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(pathways, fontsize=8)
    ax.set_title("A. Number of Curated Combinations")
    for i in range(n):
        for j in range(n):
            if count_mat[i, j] > 0:
                ax.text(j, i, f"{int(count_mat[i,j])}", ha="center", va="center",
                        fontsize=7, color="white" if count_mat[i,j] > 3 else "black")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Count")

    # Panel B: Mean Bliss score heatmap
    ax = axes[1]
    im = ax.imshow(bliss_mat, cmap="RdBu_r", aspect="auto", vmin=-0.6, vmax=0.8)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(pathways, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(pathways, fontsize=8)
    ax.set_title("B. Mean Bliss Synergy Score")
    for i in range(n):
        for j in range(n):
            if not np.isnan(bliss_mat[i, j]):
                ax.text(j, i, f"{bliss_mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(bliss_mat[i,j]) > 0.3 else "black")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Bliss Score")

    fig.suptitle("Figure 2. Pathway-Level Antibiotic Combination Landscape (45 Curated Pairs)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_pathway_heatmap.png")
    fig.savefig(FIG_DIR / "fig2_pathway_heatmap.pdf")
    print("  Fig 2 saved")
    plt.close()


def fig3_ml_performance():
    """Fig 3: ROC curve + confusion matrix."""
    X = pd.read_csv(COMBO_DIR / "feature_matrix.csv")
    df = pd.read_csv(COMBO_DIR / "curated_combinations.csv")
    y_binary = (df["interaction"] == "synergistic").astype(int).values
    y_3class = df["interaction"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1])

    # Panel A: ROC curves
    ax = fig.add_subplot(gs[0])
    for name, clf, color, ls in [
        ("Random Forest", RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=2,
            class_weight="balanced", random_state=42), "#2196F3", "-"),
        ("Gradient Boosting", GradientBoostingClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=2,
            random_state=42), "#F44336", "--"),
    ]:
        y_prob = cross_val_predict(clf, X, y_binary, cv=skf, method="predict_proba")[:, 1]
        fpr, tpr, _ = roc_curve(y_binary, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2,
                label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("A. ROC Curves (Binary: Synergy vs Rest)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Panel B: Confusion matrix (GB, 3-class)
    ax = fig.add_subplot(gs[1])
    clf_gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=42)
    y_pred_3 = cross_val_predict(clf_gb, X, y_3class, cv=skf)
    labels = ["synergistic", "additive", "antagonistic"]
    cm = confusion_matrix(y_3class, y_pred_3, labels=labels)

    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["Syn", "Add", "Ant"], fontsize=10)
    ax.set_yticklabels(["Syn", "Add", "Ant"], fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("B. Confusion Matrix (3-Class, GB)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > 5 else "black")

    # Panel C: Permutation test
    ax = fig.add_subplot(gs[2])
    clf_rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=2,
        class_weight="balanced", random_state=42)
    our_pred = cross_val_predict(clf_rf, X, y_binary, cv=skf)
    from sklearn.metrics import f1_score
    our_f1 = f1_score(y_binary, our_pred)

    # Permutation test
    n_mc = 1000
    random_f1s = []
    for seed in range(n_mc):
        y_rand = np.random.RandomState(seed).permutation(y_binary)
        y_pred_r = cross_val_predict(
            RandomForestClassifier(n_estimators=50, max_depth=3,
                                   class_weight="balanced", random_state=seed),
            X, y_rand, cv=3)
        random_f1s.append(f1_score(y_rand, y_pred_r, zero_division=0))
    random_f1s = np.array(random_f1s)

    ax.hist(random_f1s, bins=30, color="#BDBDBD", edgecolor="white", alpha=0.8,
            label=f"Random (mean={random_f1s.mean():.3f})")
    ax.axvline(our_f1, color="#F44336", linewidth=2.5, linestyle="-",
               label=f"Our Model (F1={our_f1:.3f})")
    z = (our_f1 - random_f1s.mean()) / (random_f1s.std() + 1e-10)
    p = (random_f1s >= our_f1).mean()
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Count")
    ax.set_title(f"C. Permutation Test (z={z:.2f}, p={p:.4f})")
    ax.legend(fontsize=9)

    fig.suptitle("Figure 3. ML Synergy Prediction Performance",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_ml_performance.png")
    fig.savefig(FIG_DIR / "fig3_ml_performance.pdf")
    print("  Fig 3 saved")
    plt.close()


def fig4_feature_importance():
    """Fig 4: Feature importance horizontal bar chart."""
    X = pd.read_csv(COMBO_DIR / "feature_matrix.csv")
    df = pd.read_csv(COMBO_DIR / "curated_combinations.csv")
    y_binary = (df["interaction"] == "synergistic").astype(int).values

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=2,
        class_weight="balanced", random_state=42)
    clf.fit(X, y_binary)

    imp = pd.Series(clf.feature_importances_, index=X.columns)
    top = imp.nlargest(20).sort_values()

    # Categorize features
    colors = []
    for feat in top.index:
        if feat.startswith("pw_"):
            colors.append("#2196F3")  # pathway
        elif "flux" in feat or "react" in feat or "subsystem" in feat:
            colors.append("#4CAF50")  # FBA metabolic
        elif "mapped" in feat or "lethal" in feat or "ko_" in feat:
            colors.append("#FF9800")  # essentiality
        else:
            colors.append("#9E9E9E")  # other

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(top)), top.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f.replace("pw_A_", "Pathway A: ").replace("pw_B_", "Pathway B: ")
                        .replace("A_", "Gene A: ").replace("B_", "Gene B: ")
                        for f in top.index], fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Figure 4. Top 20 Feature Importances for Synergy Prediction",
                 fontsize=13, fontweight="bold")

    # Legend
    patches = [
        mpatches.Patch(color="#2196F3", label="Pathway identity"),
        mpatches.Patch(color="#4CAF50", label="FBA metabolic features"),
        mpatches.Patch(color="#FF9800", label="Gene essentiality"),
        mpatches.Patch(color="#9E9E9E", label="Other"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_feature_importance.png")
    fig.savefig(FIG_DIR / "fig4_feature_importance.pdf")
    print("  Fig 4 saved")
    plt.close()


def fig5_regression_scatter():
    """Fig 5: Observed vs predicted Bliss score scatter."""
    X = pd.read_csv(COMBO_DIR / "feature_matrix.csv")
    df = pd.read_csv(COMBO_DIR / "curated_combinations.csv")
    y_bliss = df["bliss_score"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    reg = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=42)
    y_pred = cross_val_predict(reg, X, y_bliss, cv=kf)

    r2 = r2_score(y_bliss, y_pred)
    corr = np.corrcoef(y_bliss, y_pred)[0, 1]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Color by interaction type
    for itype, color, marker in [
        ("synergistic", "#2196F3", "o"),
        ("additive", "#9E9E9E", "s"),
        ("antagonistic", "#F44336", "^"),
    ]:
        mask = df["interaction"] == itype
        ax.scatter(y_bliss[mask], y_pred[mask], c=color, marker=marker,
                   s=80, alpha=0.8, edgecolors="white", linewidth=0.5,
                   label=f"{itype} (n={mask.sum()})")

    # Perfect prediction line
    lims = [-0.7, 1.0]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

    # Zero lines
    ax.axhline(0, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.3)

    ax.set_xlabel("Observed Bliss Score (Literature)")
    ax.set_ylabel("Predicted Bliss Score (CV)")
    ax.set_title(f"Figure 5. Bliss Score Prediction\n(R² = {r2:.3f}, Pearson r = {corr:.3f})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    # Annotate quadrants
    ax.text(0.6, 0.7, "True\nSynergy", fontsize=10, color="#2196F3", alpha=0.5,
            ha="center", fontweight="bold")
    ax.text(-0.4, -0.5, "True\nAntagonism", fontsize=10, color="#F44336", alpha=0.5,
            ha="center", fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_regression_scatter.png")
    fig.savefig(FIG_DIR / "fig5_regression_scatter.pdf")
    print("  Fig 5 saved")
    plt.close()


def main():
    print("=" * 60)
    print("  PAPER 2 — FIGURE GENERATION")
    print("=" * 60)

    fig1_lp_threshold()
    fig2_dataset_heatmap()
    fig3_ml_performance()
    fig4_feature_importance()
    fig5_regression_scatter()

    print(f"\n  All figures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
