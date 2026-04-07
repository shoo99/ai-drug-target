"""
CALMA Engine — Full reimplementation of the CALMA methodology
"Combining Antibiotics by Leveraging Metabolism-informed Artificial neural networks"

Key components:
1. 4-Feature Sigma/Delta profiles (V+sigma, V-sigma, V+delta, V-delta)
2. 3-Layer Subsystem-Structured ANN (input sub-layers → hidden sub-layers → subsystem neurons → output)
3. 2D Potency-Toxicity Landscape with Pareto optimization
4. Weight Analysis + Feature Knock-off for interpretability
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from tqdm import tqdm
import cobra
from cobra.io import load_json_model
from config.settings import DATA_DIR, MODELS_DIR

GEM_DIR = DATA_DIR / "gem_models"
RESULTS_DIR = DATA_DIR / "calma_results"


class CALMAFeatureGenerator:
    """Generate 4-feature sigma/delta profiles from GEM flux simulations."""

    def __init__(self, model_name: str = "iML1515"):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = GEM_DIR / f"{model_name}.json"
        self.model = load_json_model(str(path))
        self.model_name = model_name

        # Wild-type
        wt_sol = self.model.optimize()
        self.wt_growth = wt_sol.objective_value
        self.wt_fluxes = wt_sol.fluxes

        # Subsystem mapping
        self.subsystems = sorted(set(r.subsystem for r in self.model.reactions if r.subsystem))
        self.rxn_to_subsystem = {r.id: r.subsystem for r in self.model.reactions if r.subsystem}
        self.subsystem_rxns = defaultdict(list)
        for rxn in self.model.reactions:
            if rxn.subsystem:
                self.subsystem_rxns[rxn.subsystem].append(rxn.id)

        print(f"  CALMA Engine: {model_name} | {len(self.model.reactions)} rxns | "
              f"{len(self.subsystems)} subsystems | WT growth: {self.wt_growth:.4f}")

    def _find_gene(self, gene_id: str):
        for g in self.model.genes:
            if g.id.lower() == gene_id.lower() or g.name.lower() == gene_id.lower():
                return g
            if gene_id.lower() in g.id.lower() or gene_id.lower() in g.name.lower():
                return g
        return None

    def compute_differential_flux(self, gene_id: str) -> dict:
        """Compute per-reaction differential flux activity for a gene knockout.
        Returns V_positive (upregulated reactions) and V_negative (downregulated)."""
        with self.model as m:
            gene = self._find_gene(gene_id)
            if not gene:
                return None

            gene.knock_out()
            ko_sol = m.optimize()
            ko_growth = ko_sol.objective_value

            # Per-reaction flux changes
            v_positive = {}  # Upregulated reactions
            v_negative = {}  # Downregulated reactions

            for rxn in m.reactions:
                if not rxn.subsystem:
                    continue
                wt_flux = abs(self.wt_fluxes.get(rxn.id, 0))
                ko_flux = abs(ko_sol.fluxes.get(rxn.id, 0)) if ko_sol.status == "optimal" else 0

                diff = ko_flux - wt_flux
                threshold = max(wt_flux * 0.1, 1e-6)  # 10% change threshold

                if diff > threshold:
                    v_positive[rxn.id] = diff
                elif diff < -threshold:
                    v_negative[rxn.id] = abs(diff)

            return {
                "gene": gene_id,
                "ko_growth": ko_growth,
                "growth_ratio": ko_growth / self.wt_growth if self.wt_growth > 0 else 0,
                "v_positive": v_positive,
                "v_negative": v_negative,
                "n_upregulated": len(v_positive),
                "n_downregulated": len(v_negative),
            }

    def compute_sigma_delta_4feature(self, flux_a: dict, flux_b: dict) -> dict:
        """
        Compute CALMA 4-feature sigma/delta scores per subsystem.

        For each subsystem:
        - V+_sigma = average of V_positive effects (shared upregulation)
        - V-_sigma = average of V_negative effects (shared downregulation)
        - V+_delta = difference in V_positive effects (unique upregulation)
        - V-_delta = difference in V_negative effects (unique downregulation)
        """
        features = {}

        for subsystem in self.subsystems:
            rxns = self.subsystem_rxns[subsystem]

            # Aggregate per-subsystem
            vp_a = sum(flux_a["v_positive"].get(r, 0) for r in rxns)
            vn_a = sum(flux_a["v_negative"].get(r, 0) for r in rxns)
            vp_b = sum(flux_b["v_positive"].get(r, 0) for r in rxns)
            vn_b = sum(flux_b["v_negative"].get(r, 0) for r in rxns)

            features[subsystem] = {
                "vp_sigma": (vp_a + vp_b) / 2,      # Shared upregulation
                "vn_sigma": (vn_a + vn_b) / 2,      # Shared downregulation
                "vp_delta": abs(vp_a - vp_b),        # Unique upregulation
                "vn_delta": abs(vn_a - vn_b),        # Unique downregulation
            }

        return features

    def generate_combination_features(self, gene_list: list[str]) -> tuple:
        """Generate 4-feature sigma/delta for all gene pairs."""
        print(f"\n  Computing differential fluxes for {len(gene_list)} genes...")

        # Single gene fluxes
        single_fluxes = {}
        for gene in tqdm(gene_list, desc="Single gene fluxes"):
            result = self.compute_differential_flux(gene)
            if result:
                single_fluxes[gene] = result

        found = list(single_fluxes.keys())
        print(f"  Found in model: {len(found)}/{len(gene_list)}")

        if len(found) < 2:
            return pd.DataFrame(), []

        # Generate 4-feature profiles for all pairs
        pairs = list(combinations(found, 2))
        print(f"  Computing {len(pairs)} pair sigma/delta profiles...")

        rows = []
        feature_names = []

        for gene_a, gene_b in tqdm(pairs, desc="Sigma/Delta 4-feature"):
            sd = self.compute_sigma_delta_4feature(
                single_fluxes[gene_a], single_fluxes[gene_b]
            )

            row = {"gene_a": gene_a, "gene_b": gene_b}

            # Growth info
            row["growth_a"] = single_fluxes[gene_a]["growth_ratio"]
            row["growth_b"] = single_fluxes[gene_b]["growth_ratio"]

            # Double knockout
            with self.model as m:
                ga = self._find_gene(gene_a)
                gb = self._find_gene(gene_b)
                if ga and gb:
                    ga.knock_out()
                    gb.knock_out()
                    dko = m.optimize()
                    row["growth_ab"] = dko.objective_value / self.wt_growth if self.wt_growth > 0 else 0
                else:
                    row["growth_ab"] = 1.0

            # Bliss synergy
            bliss_expected = row["growth_a"] * row["growth_b"]
            row["bliss_score"] = bliss_expected - row["growth_ab"]
            row["bliss_expected"] = bliss_expected

            if row["bliss_score"] > 0.05:
                row["interaction"] = "synergistic"
            elif row["bliss_score"] < -0.05:
                row["interaction"] = "antagonistic"
            else:
                row["interaction"] = "additive"

            # 4-feature per subsystem
            for subsystem, feat in sd.items():
                safe_name = subsystem.replace(" ", "_").replace(",", "").replace("/", "_")[:60]
                row[f"{safe_name}_vp_sigma"] = feat["vp_sigma"]
                row[f"{safe_name}_vn_sigma"] = feat["vn_sigma"]
                row[f"{safe_name}_vp_delta"] = feat["vp_delta"]
                row[f"{safe_name}_vn_delta"] = feat["vn_delta"]

            rows.append(row)

        df = pd.DataFrame(rows)

        # Feature column names (for NN input)
        feature_cols = [c for c in df.columns
                        if any(c.endswith(s) for s in ["_vp_sigma", "_vn_sigma", "_vp_delta", "_vn_delta"])]

        df.to_csv(RESULTS_DIR / "sigma_delta_4feature.csv", index=False)
        print(f"  Generated {len(df)} pair profiles with {len(feature_cols)} features")

        return df, feature_cols


class CALMANeuralNetwork(nn.Module):
    """
    3-Layer Subsystem-Structured ANN — exact CALMA architecture.

    Layer 1 (Input sub-layers): Each subsystem has its own input sub-layer
    Layer 2 (First hidden sub-layers): Process within each subsystem
    Layer 3 (Second hidden): 1 neuron per subsystem (bottleneck)
    Output: Potency + Toxicity heads
    """

    def __init__(self, subsystem_feature_sizes: dict, dropout: float = 0.3):
        super().__init__()

        self.subsystem_names = list(subsystem_feature_sizes.keys())
        self.n_subsystems = len(self.subsystem_names)

        # Layer 1+2: Input sub-layer → First hidden sub-layer per subsystem
        self.subsystem_encoders = nn.ModuleDict()
        for name, n_features in subsystem_feature_sizes.items():
            safe_name = name.replace(" ", "_").replace(",", "").replace("/", "_")[:60]
            hidden = max(n_features // 2, 2)
            self.subsystem_encoders[safe_name] = nn.Sequential(
                nn.Linear(n_features, hidden),  # Input sub-layer
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden, 1),           # First hidden → 1 neuron (bottleneck)
                nn.Tanh(),
            )

        # Layer 3: Second hidden layer (subsystem neurons concatenated)
        # Output: Potency + Toxicity
        self.potency_head = nn.Sequential(
            nn.Linear(self.n_subsystems, max(self.n_subsystems // 2, 4)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(self.n_subsystems // 2, 4), 1),
            nn.Sigmoid(),
        )

        self.toxicity_head = nn.Sequential(
            nn.Linear(self.n_subsystems, max(self.n_subsystems // 2, 4)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(self.n_subsystems // 2, 4), 1),
            nn.Sigmoid(),
        )

    def forward(self, subsystem_inputs: dict) -> tuple:
        """
        subsystem_inputs: dict of {subsystem_name: tensor}
        Returns: (potency, toxicity)
        """
        subsystem_outputs = []
        for name in self.subsystem_names:
            safe_name = name.replace(" ", "_").replace(",", "").replace("/", "_")[:60]
            if safe_name in subsystem_inputs:
                out = self.subsystem_encoders[safe_name](subsystem_inputs[safe_name])
                subsystem_outputs.append(out)

        if not subsystem_outputs:
            batch_size = 1
            dummy = torch.zeros(batch_size, self.n_subsystems)
            return self.potency_head(dummy), self.toxicity_head(dummy)

        # Second hidden layer: concatenate all subsystem neurons
        combined = torch.cat(subsystem_outputs, dim=1)

        potency = self.potency_head(combined)
        toxicity = self.toxicity_head(combined)
        return potency, toxicity

    def get_subsystem_weights(self) -> dict:
        """Extract second hidden layer weights for interpretability."""
        weights = {}

        # Potency head weights
        pot_w = self.potency_head[0].weight.data.numpy()
        tox_w = self.toxicity_head[0].weight.data.numpy()

        for i, name in enumerate(self.subsystem_names):
            weights[name] = {
                "potency_weight": float(np.mean(np.abs(pot_w[:, i]))),
                "toxicity_weight": float(np.mean(np.abs(tox_w[:, i]))),
                "potency_direction": "positive" if np.mean(pot_w[:, i]) > 0 else "negative",
                "toxicity_direction": "positive" if np.mean(tox_w[:, i]) > 0 else "negative",
            }

        return dict(sorted(weights.items(),
                          key=lambda x: x[1]["potency_weight"] + x[1]["toxicity_weight"],
                          reverse=True))

    def feature_knockoff(self, subsystem_inputs: dict) -> dict:
        """Feature knock-off analysis: zero out each subsystem and measure impact."""
        self.eval()
        with torch.no_grad():
            base_pot, base_tox = self.forward(subsystem_inputs)
            base_p = base_pot.mean().item()
            base_t = base_tox.mean().item()

        impacts = {}
        for name in self.subsystem_names:
            safe_name = name.replace(" ", "_").replace(",", "").replace("/", "_")[:60]
            # Zero out this subsystem
            masked_inputs = {}
            for k, v in subsystem_inputs.items():
                if k == safe_name:
                    masked_inputs[k] = torch.zeros_like(v)
                else:
                    masked_inputs[k] = v

            with torch.no_grad():
                ko_pot, ko_tox = self.forward(masked_inputs)

            pot_change = (ko_pot.mean().item() - base_p) / (abs(base_p) + 1e-8) * 100
            tox_change = (ko_tox.mean().item() - base_t) / (abs(base_t) + 1e-8) * 100

            impacts[name] = {
                "potency_change_pct": round(pot_change, 2),
                "toxicity_change_pct": round(tox_change, 2),
                "potency_direction": "increases" if pot_change > 0 else "decreases",
                "toxicity_direction": "increases" if tox_change > 0 else "decreases",
            }

        return dict(sorted(impacts.items(),
                          key=lambda x: abs(x[1]["potency_change_pct"]) + abs(x[1]["toxicity_change_pct"]),
                          reverse=True))


class CALMATrainer:
    """Train and evaluate the full CALMA pipeline."""

    def __init__(self):
        self.model = None
        self.scalers = {}
        self.subsystem_feature_map = {}

    def prepare_subsystem_inputs(self, df: pd.DataFrame, feature_cols: list[str],
                                  subsystems: list[str]) -> dict:
        """Split features into per-subsystem tensors."""
        self.subsystem_feature_map = {}

        for subsystem in subsystems:
            safe_name = subsystem.replace(" ", "_").replace(",", "").replace("/", "_")[:60]
            sub_cols = [c for c in feature_cols if c.startswith(safe_name)]
            if sub_cols:
                self.subsystem_feature_map[subsystem] = sub_cols

        subsystem_inputs = {}
        for subsystem, cols in self.subsystem_feature_map.items():
            safe_name = subsystem.replace(" ", "_").replace(",", "").replace("/", "_")[:60]
            values = df[cols].fillna(0).values.astype(np.float32)
            if safe_name not in self.scalers:
                self.scalers[safe_name] = StandardScaler()
                values = self.scalers[safe_name].fit_transform(values)
            else:
                values = self.scalers[safe_name].transform(values)
            subsystem_inputs[safe_name] = torch.FloatTensor(values)

        return subsystem_inputs

    def train(self, df: pd.DataFrame, feature_cols: list[str],
              subsystems: list[str], epochs: int = 300) -> dict:
        """Train the CALMA neural network."""
        print(f"\n{'='*60}")
        print("CALMA NEURAL NETWORK TRAINING")
        print(f"{'='*60}")

        subsystem_inputs = self.prepare_subsystem_inputs(df, feature_cols, subsystems)

        # Feature sizes per subsystem
        feature_sizes = {}
        for subsystem, cols in self.subsystem_feature_map.items():
            feature_sizes[subsystem] = len(cols)

        # Target: potency = 1 - growth_ab, toxicity = proxy from pathway overlap
        potency_target = (1.0 - df["growth_ab"].fillna(1).values).astype(np.float32)
        # Toxicity proxy: bliss_score < 0 means antagonistic (potentially toxic interaction)
        toxicity_proxy = np.clip(-df["bliss_score"].fillna(0).values + 0.5, 0, 1).astype(np.float32)

        y_pot = torch.FloatTensor(potency_target).unsqueeze(1)
        y_tox = torch.FloatTensor(toxicity_proxy).unsqueeze(1)

        # Build model
        self.model = CALMANeuralNetwork(feature_sizes)

        n_params = sum(p.numel() for p in self.model.parameters())
        n_features_total = sum(feature_sizes.values())
        n_params_fc = n_features_total * 32 + 32 * 16 + 16 * 2
        reduction = (1 - n_params / max(n_params_fc, 1)) * 100

        print(f"  Subsystems: {len(feature_sizes)}")
        print(f"  Total features: {n_features_total}")
        print(f"  Model parameters: {n_params}")
        print(f"  Equivalent FC params: {n_params_fc}")
        print(f"  Parameter reduction: {reduction:.1f}%")

        # Training
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            pred_pot, pred_tox = self.model(subsystem_inputs)
            loss = criterion(pred_pot, y_pot) + criterion(pred_tox, y_tox)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

        # Evaluate
        self.model.eval()
        with torch.no_grad():
            final_pot, final_tox = self.model(subsystem_inputs)

        pot_r2 = r2_score(potency_target, final_pot.numpy().flatten()) if len(potency_target) > 2 else 0
        tox_r2 = r2_score(toxicity_proxy, final_tox.numpy().flatten()) if len(toxicity_proxy) > 2 else 0

        print(f"\n  Final Potency R²: {pot_r2:.3f}")
        print(f"  Final Toxicity R²: {tox_r2:.3f}")

        # Save predictions
        df["calma_potency"] = final_pot.numpy().flatten()
        df["calma_toxicity"] = final_tox.numpy().flatten()
        df["calma_quality"] = df["calma_potency"] * (1 - df["calma_toxicity"])

        # Save model
        torch.save(self.model.state_dict(), MODELS_DIR / "calma_nn.pt")

        return {
            "n_params": n_params,
            "param_reduction": round(reduction, 1),
            "potency_r2": round(pot_r2, 3),
            "toxicity_r2": round(tox_r2, 3),
            "best_loss": round(best_loss, 4),
        }

    def generate_landscape(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 2D potency-toxicity landscape with Pareto optimal selection."""
        print(f"\n  Generating Potency-Toxicity Landscape...")

        if "calma_potency" not in df.columns:
            return df

        # Pareto optimal: high potency AND low toxicity
        pareto = []
        for i, row in df.iterrows():
            dominated = False
            for j, other in df.iterrows():
                if i == j:
                    continue
                if (other["calma_potency"] >= row["calma_potency"] and
                    other["calma_toxicity"] <= row["calma_toxicity"] and
                    (other["calma_potency"] > row["calma_potency"] or
                     other["calma_toxicity"] < row["calma_toxicity"])):
                    dominated = True
                    break
            pareto.append(not dominated)

        df["pareto_optimal"] = pareto

        # Quadrant classification
        pot_median = df["calma_potency"].median()
        tox_median = df["calma_toxicity"].median()

        def classify(row):
            if row["calma_potency"] >= pot_median and row["calma_toxicity"] < tox_median:
                return "Q1: High Potency + Low Toxicity (IDEAL)"
            elif row["calma_potency"] >= pot_median and row["calma_toxicity"] >= tox_median:
                return "Q2: High Potency + High Toxicity (RISKY)"
            elif row["calma_potency"] < pot_median and row["calma_toxicity"] < tox_median:
                return "Q3: Low Potency + Low Toxicity (SAFE)"
            else:
                return "Q4: Low Potency + High Toxicity (AVOID)"

        df["quadrant"] = df.apply(classify, axis=1)

        # Stats
        ideal = df[df["quadrant"].str.contains("IDEAL")]
        pareto_count = df["pareto_optimal"].sum()
        total = len(df)
        reduction = (1 - len(ideal) / total) * 100 if total > 0 else 0

        print(f"  Total combinations: {total}")
        print(f"  Pareto optimal: {pareto_count}")
        print(f"  Ideal quadrant (High Pot + Low Tox): {len(ideal)}")
        print(f"  Search space reduction: {reduction:.1f}%")

        df.to_csv(RESULTS_DIR / "calma_landscape.csv", index=False)
        return df

    def interpret_model(self, subsystem_inputs: dict) -> dict:
        """Full model interpretation: weights + knock-off."""
        if self.model is None:
            return {}

        print(f"\n{'='*60}")
        print("MODEL INTERPRETATION")
        print(f"{'='*60}")

        # Weight analysis
        weights = self.model.get_subsystem_weights()
        print(f"\n  Weight Analysis (top 10):")
        for i, (name, w) in enumerate(list(weights.items())[:10], 1):
            short = name[:35]
            print(f"    {i:2d}. {short:37s} | Pot: {w['potency_weight']:.4f} ({w['potency_direction']}) | "
                  f"Tox: {w['toxicity_weight']:.4f} ({w['toxicity_direction']})")

        # Feature knock-off
        knockoff = self.model.feature_knockoff(subsystem_inputs)
        print(f"\n  Feature Knock-off (top 10):")
        for i, (name, ko) in enumerate(list(knockoff.items())[:10], 1):
            short = name[:35]
            print(f"    {i:2d}. {short:37s} | Pot Δ: {ko['potency_change_pct']:+.2f}% | "
                  f"Tox Δ: {ko['toxicity_change_pct']:+.2f}%")

        # Save
        w_df = pd.DataFrame([{"pathway": k, **v} for k, v in weights.items()])
        w_df.to_csv(RESULTS_DIR / "calma_weight_analysis.csv", index=False)

        ko_df = pd.DataFrame([{"pathway": k, **v} for k, v in knockoff.items()])
        ko_df.to_csv(RESULTS_DIR / "calma_knockoff_analysis.csv", index=False)

        return {"weights": weights, "knockoff": knockoff}

    def generate_experiment_design(self, df: pd.DataFrame, top_n: int = 5) -> str:
        """Generate experimental validation design for top combinations."""
        print(f"\n{'='*60}")
        print("EXPERIMENTAL VALIDATION DESIGN")
        print(f"{'='*60}")

        if "calma_quality" not in df.columns:
            return ""

        top = df.nlargest(top_n, "calma_quality")

        design = []
        design.append("=" * 60)
        design.append("EXPERIMENTAL VALIDATION PROTOCOL")
        design.append("Based on CALMA Analysis Results")
        design.append("=" * 60)

        design.append("\n## 1. In Vitro Validation — Potency")
        design.append("Organism: E. coli MG1655 (or clinical ESKAPE isolates)")
        design.append("Method: Checkerboard assay + SynergyFinder+ (Loewe model)")
        design.append("\nPriority combinations to test:")

        for i, (_, row) in enumerate(top.iterrows(), 1):
            design.append(f"\n  {i}. {row['gene_a']} + {row['gene_b']}")
            design.append(f"     Predicted: Potency={row['calma_potency']:.3f} | "
                         f"Toxicity={row['calma_toxicity']:.3f}")
            design.append(f"     Interaction: {row.get('interaction', 'N/A')}")
            design.append(f"     Bliss score: {row.get('bliss_score', 0):.4f}")

        design.append("\n\n## 2. In Vitro Validation — Toxicity")
        design.append("Cell lines: HEK293 (kidney), HEPG2 (liver), HK2 (kidney), HEP3B (liver)")
        design.append("Method: CellTiter-Glo luminescent viability assay")
        design.append("Synergy analysis: SynergyFinder+ with Loewe synergy scores")

        design.append("\n\n## 3. Nucleotide Salvage Pathway Validation")
        design.append("(Key finding from CALMA: nucleotide salvage modulates toxicity)")
        design.append("\nSupplements to test:")
        design.append("  - Guanine (purine) — 100μM")
        design.append("  - Thymine (pyrimidine) — 100μM")
        design.append("  - Xanthine (purine analog) — 100μM")
        design.append("  - Gemcitabine (nucleoside inhibitor) — 1μM")
        design.append("\nProtocol:")
        design.append("  1. Treat cells with drug combinations ± nucleotide supplements")
        design.append("  2. Measure viability at 24h, 48h, 72h")
        design.append("  3. Calculate Bliss synergy scores for each condition")
        design.append("  4. If supplement rescues toxicity → pathway-specific mechanism confirmed")

        design.append("\n\n## 4. EHR Validation (if data access available)")
        design.append("Database: Hospital EHR or insurance claims data")
        design.append("Method: Propensity score matching")
        design.append("Endpoint: Nephrotoxicity/hepatotoxicity codes within 6 months")

        text = "\n".join(design)

        output = RESULTS_DIR / "experiment_design.txt"
        with open(output, "w") as f:
            f.write(text)
        print(f"\n  Saved: {output}")
        print(text[:500] + "\n...")

        return text
