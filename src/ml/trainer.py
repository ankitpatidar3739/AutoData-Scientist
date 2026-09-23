"""
Comprehensive model training framework for:
1. PyTorch TabularDeepNet with EarlyStopping and Cosine Annealing
2. LightGBM baseline gradient boosted trees
3. Unified metric calculation (ROC-AUC, F1, LogLoss, RMSE, MAE)
"""

import time
import copy
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from src.ml.architectures import TabularDeepNet
from src.ml.preprocessor import PyTorchTabularDataset

def calculate_metrics(y_true: np.ndarray, y_pred_prob_or_val: np.ndarray, task_type: str) -> Dict[str, float]:
    """Calculates comprehensive evaluation metrics based on task type."""
    metrics = {}
    if task_type == "binary_classification":
        probs = np.clip(y_pred_prob_or_val, 1e-7, 1 - 1e-7)
        preds = (probs >= 0.5).astype(int)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
        except Exception:
            metrics["roc_auc"] = 0.5
        metrics["f1"] = float(f1_score(y_true, preds, zero_division=0))
        metrics["precision"] = float(precision_score(y_true, preds, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, preds, zero_division=0))
        metrics["accuracy"] = float(accuracy_score(y_true, preds))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, preds))
        metrics["log_loss"] = float(log_loss(y_true, probs))
    elif task_type == "multiclass_classification":
        if y_pred_prob_or_val.ndim == 2:
            preds = np.argmax(y_pred_prob_or_val, axis=1)
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_prob_or_val, multi_class="ovr"))
            except Exception:
                pass
        else:
            preds = np.round(y_pred_prob_or_val).astype(int)
        metrics["f1"] = float(f1_score(y_true, preds, average="weighted", zero_division=0))
        metrics["precision"] = float(precision_score(y_true, preds, average="weighted", zero_division=0))
        metrics["recall"] = float(recall_score(y_true, preds, average="weighted", zero_division=0))
        metrics["accuracy"] = float(accuracy_score(y_true, preds))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, preds))
    elif task_type == "regression":
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred_prob_or_val)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred_prob_or_val))
        metrics["r2"] = float(r2_score(y_true, y_pred_prob_or_val))
    return metrics

class PyTorchTrainer:
    def __init__(
        self,
        model: TabularDeepNet,
        task_type: str = "binary_classification",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        pos_weight: Optional[float] = None
    ):
        self.model = model.to(device)
        self.task_type = task_type
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Loss function
        if task_type == "binary_classification":
            pw = torch.tensor([pos_weight], device=device) if pos_weight else None
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        elif task_type == "multiclass_classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 35,
        patience: int = 7
    ) -> Dict[str, Any]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_metric": []}

        for epoch in range(epochs):
            # Training Phase
            self.model.train()
            total_train_loss = 0.0
            for x_cont, x_cat, y in train_loader:
                x_cont = x_cont.to(self.device) if x_cont.shape[1] > 0 else None
                x_cat = x_cat.to(self.device) if x_cat.shape[1] > 0 else None
                y = y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x_cont, x_cat)
                loss = self.criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_train_loss += loss.item() * len(y)

            scheduler.step()
            avg_train_loss = total_train_loss / len(train_loader.dataset)

            # Validation Phase
            val_loss, val_metrics = self.evaluate(val_loader)
            primary_val_metric = val_metrics.get("roc_auc", -val_loss) if self.task_type == "binary_classification" else -val_metrics.get("rmse", val_loss)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_metric"].append(primary_val_metric)

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final Evaluation
        final_val_loss, final_val_metrics = self.evaluate(val_loader)
        return {
            "best_val_loss": best_val_loss,
            "final_val_metrics": final_val_metrics,
            "history": history,
            "epochs_trained": len(history["train_loss"])
        }

    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_cont, x_cat, y in loader:
                x_cont = x_cont.to(self.device) if x_cont.shape[1] > 0 else None
                x_cat = x_cat.to(self.device) if x_cat.shape[1] > 0 else None
                y = y.to(self.device)

                logits = self.model(x_cont, x_cat)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * len(y)

                if self.task_type == "binary_classification":
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(probs.flatten())
                elif self.task_type == "multiclass_classification":
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    all_preds.extend(probs)
                else:
                    all_preds.extend(logits.cpu().numpy().flatten())
                all_targets.extend(y.cpu().numpy().flatten())

        avg_loss = total_loss / len(loader.dataset)
        metrics = calculate_metrics(np.array(all_targets), np.array(all_preds), self.task_type)
        return avg_loss, metrics

    def predict_proba_or_val(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for x_cont, x_cat, _ in loader:
                x_cont = x_cont.to(self.device) if x_cont.shape[1] > 0 else None
                x_cat = x_cat.to(self.device) if x_cat.shape[1] > 0 else None
                logits = self.model(x_cont, x_cat)
                if self.task_type == "binary_classification":
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(probs.flatten())
                elif self.task_type == "multiclass_classification":
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    all_preds.extend(probs)
                else:
                    all_preds.extend(logits.cpu().numpy().flatten())
        return np.array(all_preds)

class LightGBMTrainer:
    """Trains a LightGBM GBDT baseline for side-by-side benchmark comparison."""
    def __init__(self, task_type: str = "binary_classification", params: Optional[Dict[str, Any]] = None):
        self.task_type = task_type
        default_params = {
            "verbosity": -1,
            "random_state": 42,
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1
        }
        if task_type == "binary_classification":
            default_params["objective"] = "binary"
            default_params["metric"] = "auc"
        elif task_type == "multiclass_classification":
            default_params["objective"] = "multiclass"
            default_params["metric"] = "multi_logloss"
        else:
            default_params["objective"] = "regression"
            default_params["metric"] = "rmse"

        if params:
            default_params.update(params)
        self.params = default_params
        self.model = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        categorical_feature: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        cat_feat = categorical_feature if (categorical_feature and len(categorical_feature) > 0) else 'auto'
        if self.task_type == "binary_classification":
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
                categorical_feature=cat_feat
            )
            val_preds = self.model.predict_proba(X_val)[:, 1]
            train_preds = self.model.predict_proba(X_train)[:, 1]
        elif self.task_type == "multiclass_classification":
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
                categorical_feature=cat_feat
            )
            val_preds = self.model.predict_proba(X_val)
            train_preds = self.model.predict_proba(X_train)
        else:
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
                categorical_feature=cat_feat
            )
            val_preds = self.model.predict(X_val)
            train_preds = self.model.predict(X_train)

        val_metrics = calculate_metrics(y_val, val_preds, self.task_type)
        train_metrics = calculate_metrics(y_train, train_preds, self.task_type)

        return {
            "model": self.model,
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
            "best_iteration": self.model.best_iteration_
        }

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class RandomForestTrainer:
    """Trains a Random Forest (Bagging) baseline for variance reduction and outlier robustness."""
    def __init__(self, task_type: str = "binary_classification", n_estimators: int = 150, max_depth: Optional[int] = 12):
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        if self.task_type == "binary_classification":
            self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42, n_jobs=-1)
            self.model.fit(X_train, y_train)
            val_preds = self.model.predict_proba(X_val)[:, 1]
            train_preds = self.model.predict_proba(X_train)[:, 1]
        elif self.task_type == "multiclass_classification":
            self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42, n_jobs=-1)
            self.model.fit(X_train, y_train)
            val_preds = self.model.predict_proba(X_val)
            train_preds = self.model.predict_proba(X_train)
        else:
            self.model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42, n_jobs=-1)
            self.model.fit(X_train, y_train)
            val_preds = self.model.predict(X_val)
            train_preds = self.model.predict(X_train)

        val_metrics = calculate_metrics(y_val, val_preds, self.task_type)
        train_metrics = calculate_metrics(y_train, train_preds, self.task_type)
        return {
            "model": self.model,
            "val_metrics": val_metrics,
            "train_metrics": train_metrics
        }

class CatBoostTrainer:
    """Trains a CatBoost gradient boosting model with symmetric trees and ordered boosting."""
    def __init__(self, task_type: str = "binary_classification", iterations: int = 250, learning_rate: float = 0.05, depth: int = 6):
        self.task_type = task_type
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.model = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        if not HAS_CATBOOST:
            raise RuntimeError("CatBoost is not installed. Please install it using `pip install catboost`.")

        common_params = {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "random_seed": 42,
            "verbose": False,
            "early_stopping_rounds": 20
        }

        if self.task_type == "binary_classification":
            self.model = CatBoostClassifier(**common_params, eval_metric="Logloss")
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            val_preds = self.model.predict_proba(X_val)[:, 1]
            train_preds = self.model.predict_proba(X_train)[:, 1]
        elif self.task_type == "multiclass_classification":
            self.model = CatBoostClassifier(**common_params, eval_metric="MultiClass")
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            val_preds = self.model.predict_proba(X_val)
            train_preds = self.model.predict_proba(X_train)
        else:
            self.model = CatBoostRegressor(**common_params, eval_metric="RMSE")
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            val_preds = self.model.predict(X_val)
            train_preds = self.model.predict(X_train)

        val_metrics = calculate_metrics(y_val, val_preds, self.task_type)
        train_metrics = calculate_metrics(y_train, train_preds, self.task_type)
        best_iter = getattr(self.model, "best_iteration_", self.iterations)

        return {
            "model": self.model,
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
            "best_iteration": best_iter
        }

class EnsembleBlender:
    """
    Kaggle Grandmaster-style Stacking & Soft-Voting Ensemble Blender.
    Measures prediction error diversity and blends predictions using performance-weighted averaging.
    """
    def __init__(self, task_type: str = "binary_classification", metric_name: str = "roc_auc"):
        self.task_type = task_type
        self.metric_name = metric_name
        self.weights: Dict[str, float] = {}

    def compute_diversity_matrix(self, predictions_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculates Pearson correlation between model predictions to verify error diversity."""
        model_names = list(predictions_dict.keys())
        diversity = {}
        for m1 in model_names:
            diversity[m1] = {}
            for m2 in model_names:
                p1 = np.array(predictions_dict[m1]).flatten()
                p2 = np.array(predictions_dict[m2]).flatten()
                if len(p1) > 1 and len(p2) > 1 and len(p1) == len(p2):
                    try:
                        corr = float(np.corrcoef(p1, p2)[0, 1])
                        diversity[m1][m2] = round(corr, 4) if not np.isnan(corr) else 1.0
                    except Exception:
                        diversity[m1][m2] = 1.0
                else:
                    diversity[m1][m2] = 1.0
        return diversity

    def fit_blend(self, val_predictions_dict: Dict[str, np.ndarray], y_val: np.ndarray) -> Dict[str, float]:
        """Calculates optimal weights inversely proportional to validation loss or proportional to validation score."""
        scores = {}
        for name, preds in val_predictions_dict.items():
            m = calculate_metrics(y_val, preds, self.task_type)
            score = m.get(self.metric_name, 0.5)
            scores[name] = max(0.01, score) if self.metric_name not in ("rmse", "mae") else 1.0 / (score + 1e-5)

        total_score = sum(scores.values())
        self.weights = {name: round(s / total_score, 4) for name, s in scores.items()}
        return self.weights

    def predict_blend(self, test_predictions_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Blends test predictions according to learned weights."""
        blend = None
        for name, preds in test_predictions_dict.items():
            w = self.weights.get(name, 1.0 / len(test_predictions_dict))
            arr = np.array(preds)
            if blend is None:
                blend = w * arr
            else:
                blend += w * arr
        if self.task_type == "multiclass_classification" and blend is not None:
            if blend.ndim == 2:
                return np.argmax(blend, axis=1)
            return np.round(blend).astype(int)
        return blend
