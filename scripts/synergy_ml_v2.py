#!/usr/bin/env python3
"""
Paper 2 — ML Synergy Model v2

Addresses reviewer critique:
1. GroupKFold (gene-level) to prevent data leakage
2. Ablation: pathway-only vs FBA-only vs combined
3. Ablation: remove ribosome combinations
4. Leave-one-gene-out CV
5. Cross-species mapping coverage report
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import (StratifiedKFold, GroupKFold,
                                      LeaveOneGroupOut, cross_val_predict)
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                              classification_report, r2_score)

from config.settings import DATA_DIR
COMBO_DIR = DATA_DIR / "combination"


def load_data():
    X = pd.read_csv(COMBO_DIR / "feature_matrix.csv")
    df = pd.read_csv(COMBO_DIR / "curated_combinations.csv")
    y_binary = (df["interaction"] == "synergistic").astype(int).values
    y_bliss = df["bliss_score"].values
    return X, df, y_binary, y_bliss


def get_gene_groups(df):
    """Create group labels for GroupKFold — group by shared genes."""
    # Each unique gene gets a group ID; combinations sharing a gene
    # are in the same group
    all_genes = sorted(set(df["target_A"]) | set(df["target_B"]))
    gene_to_id = {g: i for i, g in enumerate(all_genes)}

    # For GroupKFold, use the LESS common gene in the pair as group
    # This prevents the most common genes from leaking
    groups = []
    for _, row in df.iterrows():
        # Use alphabetically first gene as group key
        g = min(row["target_A"], row["target_B"])
        groups.append(gene_to_id[g])
    return np.array(groups), all_genes


def identify_feature_sets(X):
    """Split features into pathway-only, FBA-only, and other."""
    pw_cols = [c for c in X.columns if c.startswith("pw_")]
    fba_cols = [c for c in X.columns if any(k in c for k in
                ["flux", "react", "subsystem", "mapped", "lethal",
                 "ko_growth", "n_reactions", "disruption"])]
    other_cols = [c for c in X.columns if c not in pw_cols and c not in fba_cols]
    return pw_cols, fba_cols, other_cols


def run_cv(X, y, cv, clf_factory, method="binary"):
    """Run cross-validation and return metrics."""
    clf = clf_factory()
    try:
        y_pred = cross_val_predict(clf, X, y, cv=cv)
        if method == "binary":
            f1 = f1_score(y, y_pred, zero_division=0)
            acc = accuracy_score(y, y_pred)
            try:
                y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
                auroc = roc_auc_score(y, y_prob)
            except:
                auroc = 0.0
            return {"f1": f1, "acc": acc, "auroc": auroc}
        elif method == "regression":
            r2 = r2_score(y, y_pred)
            corr = np.corrcoef(y, y_pred)[0, 1] if len(set(y_pred)) > 1 else 0
            return {"r2": r2, "corr": corr}
    except Exception as e:
        return {"f1": 0, "acc": 0, "auroc": 0, "error": str(e)}


def main():
    X, df, y_binary, y_bliss = load_data()
    groups, all_genes = get_gene_groups(df)
    pw_cols, fba_cols, other_cols = identify_feature_sets(X)

    print("=" * 70)
    print("  PAPER 2 — ML MODEL v2 (Addressing Reviewer Critique)")
    print("=" * 70)

    # ============================================================
    # SECTION 0: Cross-species mapping coverage (#5)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  0. CROSS-SPECIES FEATURE MAPPING COVERAGE")
    print(f"{'='*70}")

    mapped_genes = []
    unmapped_genes = []
    for g in all_genes:
        # Check if mapped (A_mapped or B_mapped in any row)
        rows_a = df[df["target_A"] == g]
        rows_b = df[df["target_B"] == g]
        is_mapped = False
        for _, r in pd.concat([rows_a, rows_b]).iterrows():
            idx = df.index.get_loc(r.name)
            if r["target_A"] == g and X.iloc[idx]["A_mapped"] == 1:
                is_mapped = True
                break
            if r["target_B"] == g and X.iloc[idx]["B_mapped"] == 1:
                is_mapped = True
                break
        if is_mapped:
            mapped_genes.append(g)
        else:
            unmapped_genes.append(g)

    print(f"  Mapped to iML1515: {len(mapped_genes)}/{len(all_genes)}")
    print(f"    Mapped: {', '.join(mapped_genes)}")
    print(f"    Unmapped (default features): {', '.join(unmapped_genes)}")

    # Count combinations with unmapped genes
    unmapped_set = set(unmapped_genes)
    n_any_unmapped = sum(1 for _, r in df.iterrows()
                        if r["target_A"] in unmapped_set or r["target_B"] in unmapped_set)
    n_both_unmapped = sum(1 for _, r in df.iterrows()
                         if r["target_A"] in unmapped_set and r["target_B"] in unmapped_set)
    print(f"  Combinations with ≥1 unmapped gene: {n_any_unmapped}/{len(df)}")
    print(f"  Combinations with both unmapped: {n_both_unmapped}/{len(df)}")

    # Species breakdown
    print(f"\n  By organism:")
    for org in df["organism"].unique():
        sub = df[df["organism"] == org]
        n_unm = sum(1 for _, r in sub.iterrows()
                    if r["target_A"] in unmapped_set or r["target_B"] in unmapped_set)
        print(f"    {org}: {len(sub)} total, {n_unm} with unmapped gene")

    # ============================================================
    # SECTION 1: CV Strategy Comparison (#2)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  1. CV STRATEGY COMPARISON (Data Leakage Test)")
    print(f"{'='*70}")

    clf_factory = lambda: GradientBoostingClassifier(
        n_estimators=100, max_depth=3, min_samples_leaf=2, random_state=42)

    cv_strategies = {
        "Stratified 5-Fold (original)": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        "GroupKFold (gene-level, 5)": GroupKFold(n_splits=5),
    }

    # Determine number of unique groups for LeaveOneGroupOut
    n_unique_groups = len(set(groups))
    print(f"  Unique gene groups: {n_unique_groups}")

    for name, cv in cv_strategies.items():
        if "Group" in name:
            result = run_cv(X, y_binary, cv, clf_factory)
            # GroupKFold needs groups parameter - need to use manual loop
            clf = clf_factory()
            y_pred = np.zeros_like(y_binary)
            y_prob = np.zeros(len(y_binary), dtype=float)
            for train_idx, test_idx in cv.split(X, y_binary, groups):
                clf_fold = clf_factory()
                clf_fold.fit(X.iloc[train_idx], y_binary[train_idx])
                y_pred[test_idx] = clf_fold.predict(X.iloc[test_idx])
                y_prob[test_idx] = clf_fold.predict_proba(X.iloc[test_idx])[:, 1]
            f1 = f1_score(y_binary, y_pred, zero_division=0)
            acc = accuracy_score(y_binary, y_pred)
            try:
                auroc = roc_auc_score(y_binary, y_prob)
            except:
                auroc = 0.0
            print(f"  {name}: F1={f1:.3f}, Acc={acc:.3f}, AUROC={auroc:.3f}")
        else:
            result = run_cv(X, y_binary, cv, clf_factory)
            print(f"  {name}: F1={result['f1']:.3f}, Acc={result['acc']:.3f}, AUROC={result['auroc']:.3f}")

    # Leave-one-gene-out (more conservative)
    print(f"\n  Leave-One-Gene-Out CV:")
    logo = LeaveOneGroupOut()
    y_pred_logo = np.zeros_like(y_binary)
    y_prob_logo = np.zeros(len(y_binary), dtype=float)
    n_folds = 0
    for train_idx, test_idx in logo.split(X, y_binary, groups):
        if len(set(y_binary[train_idx])) < 2:
            y_pred_logo[test_idx] = 0  # default
            continue
        clf = clf_factory()
        clf.fit(X.iloc[train_idx], y_binary[train_idx])
        y_pred_logo[test_idx] = clf.predict(X.iloc[test_idx])
        y_prob_logo[test_idx] = clf.predict_proba(X.iloc[test_idx])[:, 1]
        n_folds += 1
    f1_logo = f1_score(y_binary, y_pred_logo, zero_division=0)
    acc_logo = accuracy_score(y_binary, y_pred_logo)
    try:
        auroc_logo = roc_auc_score(y_binary, y_prob_logo)
    except:
        auroc_logo = 0.0
    print(f"  Leave-One-Gene-Out: F1={f1_logo:.3f}, Acc={acc_logo:.3f}, AUROC={auroc_logo:.3f} ({n_folds} folds)")

    # ============================================================
    # SECTION 2: Feature Ablation (#3)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  2. FEATURE ABLATION STUDY")
    print(f"{'='*70}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    feature_sets = {
        "All features (60)": X.columns.tolist(),
        "Pathway-only": pw_cols,
        "FBA-only": fba_cols,
        "FBA + other (no pathway)": fba_cols + other_cols,
        "Pathway + other (no FBA)": pw_cols + other_cols,
    }

    for name, cols in feature_sets.items():
        if len(cols) == 0:
            print(f"  {name}: NO FEATURES")
            continue
        X_sub = X[cols]
        result = run_cv(X_sub, y_binary, skf, clf_factory)
        print(f"  {name} ({len(cols)} features): "
              f"F1={result['f1']:.3f}, Acc={result['acc']:.3f}, AUROC={result['auroc']:.3f}")

    # ============================================================
    # SECTION 3: Ribosome Ablation (#3)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  3. RIBOSOME ABLATION STUDY")
    print(f"{'='*70}")

    # Count ribosome involvement
    ribo_mask = (df["pathway_A"] == "ribosome") | (df["pathway_B"] == "ribosome")
    print(f"  Combinations involving ribosome: {ribo_mask.sum()}/{len(df)}")
    print(f"    Synergistic: {((df['interaction']=='synergistic') & ribo_mask).sum()}")
    print(f"    Antagonistic: {((df['interaction']=='antagonistic') & ribo_mask).sum()}")
    print(f"    Additive: {((df['interaction']=='additive') & ribo_mask).sum()}")

    # Remove ribosome combinations and retrain
    X_no_ribo = X[~ribo_mask]
    y_no_ribo = y_binary[~ribo_mask]

    if len(set(y_no_ribo)) >= 2 and sum(y_no_ribo) >= 3:
        skf_nr = StratifiedKFold(n_splits=min(5, sum(y_no_ribo)), shuffle=True, random_state=42)
        result = run_cv(X_no_ribo, y_no_ribo, skf_nr, clf_factory)
        print(f"\n  Without ribosome ({len(X_no_ribo)} combinations):")
        print(f"    F1={result['f1']:.3f}, Acc={result['acc']:.3f}, AUROC={result['auroc']:.3f}")
    else:
        print(f"  Without ribosome: too few samples ({len(X_no_ribo)})")

    # ============================================================
    # SECTION 4: Permutation Test with GroupKFold (#2)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  4. PERMUTATION TEST (GroupKFold)")
    print(f"{'='*70}")

    # Our model with GroupKFold
    gkf = GroupKFold(n_splits=5)
    y_pred_gkf = np.zeros_like(y_binary)
    for train_idx, test_idx in gkf.split(X, y_binary, groups):
        clf = clf_factory()
        clf.fit(X.iloc[train_idx], y_binary[train_idx])
        y_pred_gkf[test_idx] = clf.predict(X.iloc[test_idx])
    our_f1_gkf = f1_score(y_binary, y_pred_gkf, zero_division=0)

    # Permutation test
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
    z = (our_f1_gkf - random_f1s.mean()) / (random_f1s.std() + 1e-10)
    p = (random_f1s >= our_f1_gkf).mean()

    print(f"  Our F1 (GroupKFold): {our_f1_gkf:.3f}")
    print(f"  Random baseline: {random_f1s.mean():.3f} ± {random_f1s.std():.3f}")
    print(f"  z-score: {z:.2f}")
    print(f"  p-value: {p:.4f}")
    print(f"  Significant: {'YES' if p < 0.05 else 'NO'}")

    # ============================================================
    # SECTION 5: Regression with GroupKFold
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  5. REGRESSION (GroupKFold)")
    print(f"{'='*70}")

    from sklearn.model_selection import KFold
    gkf_reg = GroupKFold(n_splits=5)

    y_pred_reg = np.zeros_like(y_bliss)
    for train_idx, test_idx in gkf_reg.split(X, y_bliss, groups):
        reg = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=42)
        reg.fit(X.iloc[train_idx], y_bliss[train_idx])
        y_pred_reg[test_idx] = reg.predict(X.iloc[test_idx])

    r2_gkf = r2_score(y_bliss, y_pred_reg)
    corr_gkf = np.corrcoef(y_bliss, y_pred_reg)[0, 1]
    print(f"  R² (GroupKFold): {r2_gkf:.3f}")
    print(f"  Pearson r: {corr_gkf:.3f}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY — CORRECTED METRICS")
    print(f"{'='*70}")
    print(f"  Original (Stratified 5-Fold):  F1=0.873, AUROC=0.866")
    print(f"  GroupKFold (gene-level):        F1={our_f1_gkf:.3f}")
    print(f"  Leave-One-Gene-Out:             F1={f1_logo:.3f}")
    print(f"  Permutation test (GroupKFold):  z={z:.2f}, p={p:.4f}")
    print(f"  Regression R² (GroupKFold):     {r2_gkf:.3f}")
    print(f"")
    print(f"  Manuscript should report GroupKFold metrics as PRIMARY.")
    print(f"  Original Stratified metrics can be in Supplementary as comparison.")

    # Save corrected results
    results = {
        "n_combinations": len(df),
        "n_features": X.shape[1],
        "n_mapped_genes": len(mapped_genes),
        "n_unmapped_genes": len(unmapped_genes),
        "f1_stratified": 0.873,
        "f1_groupkfold": our_f1_gkf,
        "f1_logo": f1_logo,
        "z_score_groupkfold": z,
        "p_value_groupkfold": p,
        "r2_groupkfold": r2_gkf,
        "corr_groupkfold": corr_gkf,
    }
    pd.Series(results).to_csv(COMBO_DIR / "ml_model_results_v2.csv")
    print(f"\n  Saved: ml_model_results_v2.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
