"""
Metabolism-Informed Neural Network — CALMA-inspired model
Uses metabolic subsystem structure to constrain neural network architecture,
enabling interpretable predictions of drug target potency and toxicity.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
from config.settings import DATA_DIR, MODELS_DIR


class MetabolismInformedNN(nn.Module):
    """
    Neural network with architecture reflecting metabolic subsystems.
    Instead of fully connected layers, the first layer groups inputs
    by metabolic subsystem — dramatically reducing parameters.
    """

    def __init__(self, n_subsystems: int, subsystem_sizes: list[int],
                 hidden_dim: int = 32, n_outputs: int = 2):
        super().__init__()

        self.n_subsystems = n_subsystems
        self.subsystem_sizes = subsystem_sizes

        # Subsystem-specific layers (each subsystem has its own small network)
        self.subsystem_layers = nn.ModuleList()
        for size in subsystem_sizes:
            if size > 0:
                self.subsystem_layers.append(nn.Sequential(
                    nn.Linear(size, max(size // 2, 1)),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ))
            else:
                self.subsystem_layers.append(nn.Identity())

        # Compute subsystem output size
        subsystem_output_size = sum(
            max(s // 2, 1) if s > 0 else 0 for s in subsystem_sizes
        )

        # Integration layers (merge subsystem representations)
        self.integration = nn.Sequential(
            nn.Linear(subsystem_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Output heads: potency and toxicity
        self.potency_head = nn.Linear(hidden_dim // 2, 1)
        self.toxicity_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x, subsystem_indices: list[tuple]):
        """
        x: input features (metabolic flux changes per subsystem)
        subsystem_indices: list of (start, end) indices for each subsystem
        """
        subsystem_outputs = []
        for i, (start, end) in enumerate(subsystem_indices):
            if end > start:
                sub_input = x[:, start:end]
                sub_output = self.subsystem_layers[i](sub_input)
                subsystem_outputs.append(sub_output)

        if not subsystem_outputs:
            # Fallback: use full input
            combined = x
        else:
            combined = torch.cat(subsystem_outputs, dim=1)

        integrated = self.integration(combined)

        potency = torch.sigmoid(self.potency_head(integrated))
        toxicity = torch.sigmoid(self.toxicity_head(integrated))

        return potency, toxicity

    def get_subsystem_importance(self, x, subsystem_indices, subsystem_names):
        """Get importance of each metabolic subsystem for predictions."""
        self.eval()
        importances = {}

        with torch.no_grad():
            base_potency, base_toxicity = self.forward(x, subsystem_indices)
            base_pot = base_potency.mean().item()
            base_tox = base_toxicity.mean().item()

        for i, (start, end) in enumerate(subsystem_indices):
            if end <= start:
                continue

            # Zero out this subsystem
            x_masked = x.clone()
            x_masked[:, start:end] = 0

            with torch.no_grad():
                masked_pot, masked_tox = self.forward(x_masked, subsystem_indices)

            pot_impact = abs(base_pot - masked_pot.mean().item())
            tox_impact = abs(base_tox - masked_tox.mean().item())

            name = subsystem_names[i] if i < len(subsystem_names) else f"subsystem_{i}"
            importances[name] = {
                "potency_impact": round(pot_impact, 4),
                "toxicity_impact": round(tox_impact, 4),
                "total_impact": round(pot_impact + tox_impact, 4),
            }

        return dict(sorted(importances.items(),
                          key=lambda x: x[1]["total_impact"], reverse=True))


class MetabolismNNTrainer:
    """Train and evaluate the metabolism-informed neural network."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.subsystem_names = []
        self.subsystem_indices = []

    def prepare_data(self, metabolic_df: pd.DataFrame,
                     subsystem_columns: list[str]) -> tuple:
        """Prepare training data from metabolic analysis results."""
        self.subsystem_names = subsystem_columns

        # Validate columns exist
        valid_cols = [c for c in subsystem_columns if c in metabolic_df.columns]
        if not valid_cols:
            raise ValueError(f"No subsystem columns found in dataframe. "
                           f"Expected: {subsystem_columns[:3]}... "
                           f"Got: {list(metabolic_df.columns[:10])}")
        if len(valid_cols) < len(subsystem_columns):
            missing = set(subsystem_columns) - set(valid_cols)
            print(f"  Warning: {len(missing)} subsystem columns missing, using {len(valid_cols)}")

        # Features: subsystem flux changes
        X = metabolic_df[valid_cols].fillna(0).values
        self.subsystem_names = valid_cols

        # Targets: potency (1 - growth_ratio) and toxicity
        def find_col(df, *candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            pattern_matches = [col for col in df.columns if any(p in col for p in candidates)]
            if pattern_matches:
                return pattern_matches[0]
            raise ValueError(f"Column not found. Tried: {candidates}")

        gr_col = find_col(metabolic_df, "growth_ratio", "growth_ratio_fba")
        ts_col = find_col(metabolic_df, "toxicity_score", "toxicity_score_fba")

        potency = 1.0 - metabolic_df[gr_col].fillna(1).values
        toxicity = metabolic_df[ts_col].fillna(0.5).values

        y = np.column_stack([potency, toxicity])

        # Build subsystem indices (each subsystem = 1 feature in this case)
        self.subsystem_indices = [(i, i+1) for i in range(len(subsystem_columns))]

        return X, y

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 100, lr: float = 0.001) -> dict:
        """Train the model with cross-validation."""
        X_scaled = self.scaler.fit_transform(X)

        n_subsystems = len(self.subsystem_names)
        subsystem_sizes = [1] * n_subsystems  # 1 feature per subsystem

        self.model = MetabolismInformedNN(
            n_subsystems=n_subsystems,
            subsystem_sizes=subsystem_sizes,
            hidden_dim=min(32, n_subsystems),
            n_outputs=2,
        )

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        n_params_fc = n_subsystems * 32 + 32 * 16 + 16 * 2  # equivalent FC
        reduction = (1 - n_params / n_params_fc) * 100 if n_params_fc > 0 else 0

        print(f"  Model parameters: {n_params}")
        print(f"  Equivalent FC params: {n_params_fc}")
        print(f"  Parameter reduction: {reduction:.1f}%")

        # Training
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        # Cross-validation
        kf = KFold(n_splits=min(3, len(X)), shuffle=True, random_state=42)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train = X_tensor[train_idx]
            y_train = y_tensor[train_idx]
            X_val = X_tensor[val_idx]
            y_val = y_tensor[val_idx]

            # Reset model
            self.model.apply(self._reset_weights)

            for epoch in range(epochs):
                self.model.train()
                optimizer.zero_grad()
                pred_pot, pred_tox = self.model(X_train, self.subsystem_indices)
                pred = torch.cat([pred_pot, pred_tox], dim=1)
                loss = criterion(pred, y_train)
                loss.backward()
                optimizer.step()

            # Evaluate
            self.model.eval()
            with torch.no_grad():
                val_pot, val_tox = self.model(X_val, self.subsystem_indices)
                val_pred = torch.cat([val_pot, val_tox], dim=1).numpy()
                val_true = y_val.numpy()

                pot_r2 = r2_score(val_true[:, 0], val_pred[:, 0]) if len(val_true) > 1 else 0
                tox_r2 = r2_score(val_true[:, 1], val_pred[:, 1]) if len(val_true) > 1 else 0
                cv_scores.append({"potency_r2": pot_r2, "toxicity_r2": tox_r2})

        # Final training on all data
        self.model.apply(self._reset_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            pred_pot, pred_tox = self.model(X_tensor, self.subsystem_indices)
            pred = torch.cat([pred_pot, pred_tox], dim=1)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()

        avg_pot_r2 = np.mean([s["potency_r2"] for s in cv_scores])
        avg_tox_r2 = np.mean([s["toxicity_r2"] for s in cv_scores])

        print(f"  CV Potency R²: {avg_pot_r2:.3f}")
        print(f"  CV Toxicity R²: {avg_tox_r2:.3f}")

        # Save model
        model_path = MODELS_DIR / "metabolism_nn.pt"
        torch.save(self.model.state_dict(), model_path)

        return {
            "n_params": n_params,
            "param_reduction": round(reduction, 1),
            "cv_potency_r2": round(avg_pot_r2, 3),
            "cv_toxicity_r2": round(avg_tox_r2, 3),
        }

    def get_pathway_importance(self, X: np.ndarray) -> dict:
        """Get importance of each metabolic pathway."""
        if self.model is None:
            return {}

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        return self.model.get_subsystem_importance(
            X_tensor, self.subsystem_indices, self.subsystem_names
        )

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        """Predict potency and toxicity for new targets."""
        if self.model is None:
            return pd.DataFrame()

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        self.model.eval()
        with torch.no_grad():
            potency, toxicity = self.model(X_tensor, self.subsystem_indices)

        return pd.DataFrame({
            "predicted_potency": potency.numpy().flatten(),
            "predicted_toxicity": toxicity.numpy().flatten(),
            "predicted_selectivity": (potency - toxicity).numpy().flatten(),
        })

    @staticmethod
    def _reset_weights(m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()
