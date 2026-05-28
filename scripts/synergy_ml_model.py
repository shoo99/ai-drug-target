#!/usr/bin/env python3
"""
Paper 2 — ML Synergy Prediction Model

Uses FBA-derived metabolic features + pathway features to predict
antibiotic combination synergy from curated experimental labels.

Key insight: FBA cannot directly compute synergy (LP limitation),
but metabolic FEATURES from FBA can be predictive when combined
with experimental labels.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import cobra
from cobra.io import load_json_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict, StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score, accuracy_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

from config.settings import DATA_DIR

GEM_DIR = DATA_DIR / "gem_models"
COMBO_DIR = DATA_DIR / "combination"


def extract_gene_fba_features(model, gene_name, wt_growth, wt_fluxes):
    """Extract FBA-derived features for a single gene."""
    gene_lower = gene_name.lower()
    target_gene = None
    for g in model.genes:
        if g.id.lower() == gene_lower or g.name.lower() == gene_lower:
            target_gene = g
            break
        if gene_lower in g.id.lower() or gene_lower in g.name.lower():
            target_gene = g
            break

    features = {}
    if not target_gene:
        # Gene not in model — return NaN features
        features["mapped"] = 0
        features["n_reactions"] = 0
        features["ko_growth_ratio"] = np.nan
        features["ko_lethal"] = np.nan
        features["n_subsystems_affected"] = 0
        features["mean_flux_change"] = 0
        features["max_flux_change"] = 0
        features["n_upregulated"] = 0
        features["n_downregulated"] = 0
        features["flux_disruption_score"] = 0
        return features

    features["mapped"] = 1
    features["n_reactions"] = len(target_gene.reactions)

    # Knockout FBA
    with model as m:
        for rxn in target_gene.reactions:
            rxn.upper_bound = 0
            rxn.lower_bound = 0
        sol = m.optimize()
        ko_growth = sol.objective_value if sol.status == "optimal" else 0.0
        ko_fluxes = sol.fluxes if sol.status == "optimal" else pd.Series(dtype=float)

    features["ko_growth_ratio"] = ko_growth / wt_growth if wt_growth > 0 else 0
    features["ko_lethal"] = 1 if features["ko_growth_ratio"] < 0.01 else 0

    # Flux changes
    if len(ko_fluxes) > 0:
        flux_diff = ko_fluxes - wt_fluxes
        flux_diff = flux_diff.dropna()
        abs_diff = flux_diff.abs()

        features["mean_flux_change"] = abs_diff.mean()
        features["max_flux_change"] = abs_diff.max()
        features["n_upregulated"] = (flux_diff > 0.01).sum()
        features["n_downregulated"] = (flux_diff < -0.01).sum()
        features["flux_disruption_score"] = abs_diff.sum()

        # Subsystem-level features
        subsystems_affected = set()
        for rxn_id in flux_diff.index:
            if rxn_id in model.reactions:
                rxn = model.reactions.get_by_id(rxn_id)
                if abs(flux_diff[rxn_id]) > 0.01 and rxn.subsystem:
                    subsystems_affected.add(rxn.subsystem)
        features["n_subsystems_affected"] = len(subsystems_affected)
    else:
        features["mean_flux_change"] = 0
        features["max_flux_change"] = 0
        features["n_upregulated"] = 0
        features["n_downregulated"] = 0
        features["flux_disruption_score"] = 0
        features["n_subsystems_affected"] = 0

    return features


def extract_pair_features(model, gene_a, gene_b, wt_growth, wt_fluxes,
                          feat_a, feat_b, pathway_a, pathway_b):
    """Extract pairwise features for a gene combination."""
    features = {}

    # Individual gene features (prefixed)
    for k, v in feat_a.items():
        features[f"A_{k}"] = v
    for k, v in feat_b.items():
        features[f"B_{k}"] = v

    # Pairwise features
    features["both_mapped"] = feat_a["mapped"] * feat_b["mapped"]
    features["both_lethal"] = feat_a.get("ko_lethal", 0) * feat_b.get("ko_lethal", 0)
    features["any_lethal"] = max(feat_a.get("ko_lethal", 0), feat_b.get("ko_lethal", 0))
    features["cross_pathway"] = 1 if pathway_a != pathway_b else 0

    # Pathway encoding
    all_pathways = ["peptidoglycan", "LPS", "ribosome", "DNA_topology",
                    "transcription", "fatty_acid", "folate", "isoprenoid",
                    "cell_division", "signaling", "outer_membrane",
                    "LPS_transport", "DNA_replication", "secretion",
                    "translation_init"]
    for p in all_pathways:
        features[f"pw_A_{p}"] = 1 if pathway_a == p else 0
        features[f"pw_B_{p}"] = 1 if pathway_b == p else 0

    # Interaction features (if both mapped)
    if feat_a["mapped"] and feat_b["mapped"]:
        features["reaction_overlap"] = len(
            set(r.id for g in [gene_a] for r in g.reactions if g) &
            set(r.id for g in [gene_b] for r in g.reactions if g)
        ) if gene_a and gene_b else 0

        features["total_reactions"] = feat_a["n_reactions"] + feat_b["n_reactions"]
        features["disruption_ratio"] = (
            feat_a["flux_disruption_score"] /
            (feat_b["flux_disruption_score"] + 1e-10)
        )
        features["combined_disruption"] = (
            feat_a["flux_disruption_score"] + feat_b["flux_disruption_score"]
        )
        features["subsystem_overlap"] = abs(
            feat_a["n_subsystems_affected"] - feat_b["n_subsystems_affected"]
        )
    else:
        features["reaction_overlap"] = 0
        features["total_reactions"] = feat_a["n_reactions"] + feat_b["n_reactions"]
        features["disruption_ratio"] = 0
        features["combined_disruption"] = 0
        features["subsystem_overlap"] = 0

    return features


def find_gene_obj(model, gene_name):
    """Find gene object in model."""
    gene_lower = gene_name.lower()
    for g in model.genes:
        if g.id.lower() == gene_lower or g.name.lower() == gene_lower:
            return g
        if gene_lower in g.id.lower() or gene_lower in g.name.lower():
            return g
    return None


def main():
    print("=" * 60)
    print("  PAPER 2 — ML SYNERGY PREDICTION MODEL")
    print("=" * 60)

    # Load curated combinations
    combo_path = COMBO_DIR / "curated_combinations.csv"
    df_combo = pd.read_csv(combo_path)
    print(f"  Loaded {len(df_combo)} curated combinations")

    # Load E. coli model (primary, most combinations)
    model = load_json_model(str(GEM_DIR / "iML1515.json"))
    wt_sol = model.optimize()
    wt_growth = wt_sol.objective_value
    wt_fluxes = wt_sol.fluxes
    print(f"  Model: iML1515, WT growth: {wt_growth:.4f}")

    # Extract features for all unique genes
    all_genes = set(df_combo["target_A"]) | set(df_combo["target_B"])
    print(f"\n  Extracting FBA features for {len(all_genes)} unique genes...")

    gene_features = {}
    gene_objects = {}
    for gname in all_genes:
        gene_objects[gname] = find_gene_obj(model, gname)
        gene_features[gname] = extract_gene_fba_features(
            model, gname, wt_growth, wt_fluxes
        )
        status = "mapped" if gene_features[gname]["mapped"] else "NOT mapped"
        print(f"    {gname:12s}: {status}")

    # Build feature matrix
    print(f"\n  Building feature matrix...")
    feature_rows = []
    labels = []

    for _, row in df_combo.iterrows():
        ga_name = row["target_A"]
        gb_name = row["target_B"]
        ga_obj = gene_objects[ga_name]
        gb_obj = gene_objects[gb_name]

        pair_feat = extract_pair_features(
            model, ga_obj, gb_obj, wt_growth, wt_fluxes,
            gene_features[ga_name], gene_features[gb_name],
            row["pathway_A"], row["pathway_B"]
        )
        # Add bliss_score as feature? No — that's the label (or proxy).
        # Add evidence level as feature
        pair_feat["evidence_clinical"] = 1 if "clinical" in row["evidence"] else 0

        feature_rows.append(pair_feat)
        labels.append(row["interaction"])

    X = pd.DataFrame(feature_rows).fillna(0)
    y = np.array(labels)

    print(f"  Feature matrix: {X.shape}")
    print(f"  Labels: {pd.Series(y).value_counts().to_dict()}")

    # ============================================================
    # MODEL 1: 3-class classification (synergistic/additive/antagonistic)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  MODEL 1: 3-CLASS CLASSIFICATION")
    print(f"{'='*60}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Stratified K-fold (LOOCV is too noisy with 45 samples)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in [
        ("Random Forest", RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=3,
            class_weight="balanced", random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=3,
            random_state=42)),
    ]:
        y_pred = cross_val_predict(clf, X, y_encoded, cv=skf)
        y_pred_labels = le.inverse_transform(y_pred)

        acc = accuracy_score(y_encoded, y_pred)
        f1_macro = f1_score(y_encoded, y_pred, average="macro")
        f1_weighted = f1_score(y_encoded, y_pred, average="weighted")

        print(f"\n  {name}:")
        print(f"    Accuracy: {acc:.3f}")
        print(f"    F1 (macro): {f1_macro:.3f}")
        print(f"    F1 (weighted): {f1_weighted:.3f}")
        print(f"    Classification Report:")
        print(classification_report(y, y_pred_labels, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y, y_pred_labels, labels=["synergistic", "additive", "antagonistic"])
        print(f"    Confusion Matrix:")
        print(f"                    Pred_Syn  Pred_Add  Pred_Ant")
        print(f"    True_Syn:       {cm[0,0]:8d}  {cm[0,1]:8d}  {cm[0,2]:8d}")
        print(f"    True_Add:       {cm[1,0]:8d}  {cm[1,1]:8d}  {cm[1,2]:8d}")
        print(f"    True_Ant:       {cm[2,0]:8d}  {cm[2,1]:8d}  {cm[2,2]:8d}")

    # ============================================================
    # MODEL 2: Binary classification (synergistic vs non-synergistic)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  MODEL 2: BINARY CLASSIFICATION (synergistic vs rest)")
    print(f"{'='*60}")

    y_binary = (y == "synergistic").astype(int)

    for name, clf in [
        ("Random Forest", RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=2,
            class_weight="balanced", random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=2,
            random_state=42)),
    ]:
        y_pred = cross_val_predict(clf, X, y_binary, cv=skf)
        y_prob = cross_val_predict(clf, X, y_binary, cv=skf, method="predict_proba")[:, 1]

        acc = accuracy_score(y_binary, y_pred)
        f1 = f1_score(y_binary, y_pred)
        try:
            auroc = roc_auc_score(y_binary, y_prob)
        except:
            auroc = 0.0

        print(f"\n  {name}:")
        print(f"    Accuracy: {acc:.3f}")
        print(f"    F1: {f1:.3f}")
        print(f"    AUROC: {auroc:.3f}")
        cm = confusion_matrix(y_binary, y_pred)
        print(f"    Confusion: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    # ============================================================
    # MODEL 3: Regression (Bliss score prediction)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  MODEL 3: REGRESSION (Bliss score)")
    print(f"{'='*60}")

    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_absolute_error

    y_bliss = df_combo["bliss_score"].values
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Use KFold for regression (not stratified)
    from sklearn.model_selection import KFold
    kf_reg = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, reg in [
        ("Random Forest", RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=42)),
    ]:
        y_pred = cross_val_predict(reg, X, y_bliss, cv=kf_reg)
        r2 = r2_score(y_bliss, y_pred)
        mae = mean_absolute_error(y_bliss, y_pred)
        corr = np.corrcoef(y_bliss, y_pred)[0, 1]

        print(f"\n  {name}:")
        print(f"    R²: {r2:.3f}")
        print(f"    MAE: {mae:.3f}")
        print(f"    Pearson r: {corr:.3f}")

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  FEATURE IMPORTANCE (Random Forest, binary)")
    print(f"{'='*60}")

    clf_final = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=2,
        class_weight="balanced", random_state=42
    )
    clf_final.fit(X, y_binary)

    importances = pd.Series(clf_final.feature_importances_, index=X.columns)
    top_features = importances.nlargest(20)
    print(f"\n  Top 20 features:")
    for feat, imp in top_features.items():
        print(f"    {feat:35s}: {imp:.4f}")

    # ============================================================
    # RANDOM BASELINE COMPARISON
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  RANDOM BASELINE COMPARISON")
    print(f"{'='*60}")

    n_mc = 1000
    random_f1s = []
    random_aurocs = []
    for _ in range(n_mc):
        y_rand = np.random.permutation(y_binary)
        y_pred_rand = cross_val_predict(
            RandomForestClassifier(n_estimators=50, max_depth=3,
                                   class_weight="balanced", random_state=None),
            X, y_rand, cv=3
        )
        random_f1s.append(f1_score(y_rand, y_pred_rand, zero_division=0))

    random_f1s = np.array(random_f1s)

    # Our model's F1
    our_pred = cross_val_predict(clf_final, X, y_binary, cv=skf)
    our_f1 = f1_score(y_binary, our_pred)

    z_score = (our_f1 - random_f1s.mean()) / (random_f1s.std() + 1e-10)
    p_value = (random_f1s >= our_f1).mean()

    print(f"  Our F1: {our_f1:.3f}")
    print(f"  Random baseline F1: {random_f1s.mean():.3f} ± {random_f1s.std():.3f}")
    print(f"  z-score: {z_score:.2f}")
    print(f"  p-value (permutation): {p_value:.4f}")
    print(f"  Statistically significant: {'YES' if p_value < 0.05 else 'NO'}")

    # Save results
    results = {
        "n_combinations": len(df_combo),
        "n_features": X.shape[1],
        "binary_f1": our_f1,
        "random_f1_mean": random_f1s.mean(),
        "z_score": z_score,
        "p_value": p_value,
    }
    pd.Series(results).to_csv(COMBO_DIR / "ml_model_results.csv")
    X.to_csv(COMBO_DIR / "feature_matrix.csv", index=False)

    print(f"\n  Saved: feature_matrix.csv, ml_model_results.csv")
    print(f"{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
