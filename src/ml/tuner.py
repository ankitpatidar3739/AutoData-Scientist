"""
Optuna hyperparameter optimization manager for PyTorch Deep Learning and LightGBM models.
"""

from typing import Dict, Any, Callable
import optuna
from optuna.samplers import TPESampler
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.ml.architectures import TabularDeepNet
from src.ml.trainer import PyTorchTrainer, LightGBMTrainer
from src.ml.preprocessor import PyTorchTabularDataset

optuna.logging.set_verbosity(optuna.logging.WARNING)

class HyperparameterTuner:
    def __init__(self, task_type: str = "binary_classification", n_trials: int = 10, seed: int = 42):
        self.task_type = task_type
        self.n_trials = n_trials
        self.seed = seed
        self.direction = "maximize" if task_type == "binary_classification" else "minimize"

    def tune_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Runs Optuna TPE search over LightGBM hyperparameter space."""
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
            trainer = LightGBMTrainer(task_type=self.task_type, params=params)
            res = trainer.train(X_train, y_train, X_val, y_val)
            val_metrics = res["val_metrics"]
            return val_metrics["roc_auc"] if self.task_type == "binary_classification" else val_metrics["rmse"]

        study = optuna.create_study(direction=self.direction, sampler=TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.n_trials)

        return {
            "best_params": study.best_params,
            "best_score": study.best_value
        }

    def tune_pytorch(
        self,
        train_dataset: PyTorchTabularDataset,
        val_dataset: PyTorchTabularDataset,
        num_continuous: int,
        cat_cardinalities: list[int],
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Runs Optuna TPE search over PyTorch neural network architectures and learning rates."""
        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            dropout = trial.suggest_float("dropout", 0.1, 0.4)
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            n_layers = trial.suggest_int("n_layers", 1, 3)

            hidden_dims = []
            cur_dim = trial.suggest_categorical("first_dim", [128, 64, 32])
            hidden_dims.append(cur_dim)
            for _ in range(n_layers - 1):
                cur_dim = max(16, cur_dim // 2)
                hidden_dims.append(cur_dim)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = TabularDeepNet(
                num_continuous=num_continuous,
                cat_cardinalities=cat_cardinalities,
                output_dim=1,
                task_type=self.task_type,
                hidden_dims=hidden_dims,
                dropout_rate=dropout,
                use_residual=True
            )

            trainer = PyTorchTrainer(model=model, task_type=self.task_type, learning_rate=lr, weight_decay=weight_decay, device=device)
            res = trainer.train(train_loader, val_loader, epochs=15, patience=4)
            val_metrics = res["final_val_metrics"]
            return val_metrics.get("roc_auc", 0.0) if self.task_type == "binary_classification" else val_metrics.get("rmse", 9999.0)

        study = optuna.create_study(direction=self.direction, sampler=TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=min(self.n_trials, 5))

        return {
            "best_params": study.best_params,
            "best_score": study.best_value
        }
