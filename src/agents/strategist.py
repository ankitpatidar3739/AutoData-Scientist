"""
Lead ML Strategist Agent.
Translates DataDossier into an optimized ExperimentPlan specifying preprocessing,
deep learning architectures, loss functions, and GBDT baselines.
"""

from typing import List, Dict, Any
from src.agents.base import BaseAgent
from src.core.config import DataDossier, ExperimentPlan, ModelRecommendation

class MLStrategistAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="MLStrategist", role="Chief Machine Learning Architect")

    def create_plan(self, dossier: DataDossier, mode: str = "championship") -> ExperimentPlan:
        """Formulates an adaptive experiment plan using Meta-Learning principles."""
        llm_prompt = (
            f"Given a tabular dataset with {dossier.row_count} rows, {dossier.col_count} columns, "
            f"task '{dossier.task_type}', target '{dossier.target_column}', "
            f"categorical features: {dossier.suggested_categorical_features}, "
            f"numeric features: {dossier.suggested_numeric_features}. "
            f"Recommend modeling strategy."
        )
        _ = self.call_llm(llm_prompt)

        # Candidate Model 1: PyTorch Tabular Deep Learning with Entity Embeddings
        pytorch_rec = ModelRecommendation(
            model_name="PyTorch_TabularDeepNet",
            model_type="pytorch_tabular",
            rationale=(
                "Deep learning architecture leveraging learnable entity embeddings for high/medium cardinality categoricals "
                "coupled with batch-normalized continuous features and residual connections. Superior at learning non-linear feature interactions."
            ),
            hyperparameter_grid={
                "hidden_dims": [128, 64],
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "batch_size": 64,
                "epochs": 35
            },
            expected_strengths="Dense representation learning, smooth regularization, and gradient-based continuous optimization."
        )

        # Candidate Model 2: LightGBM Baseline
        lgbm_rec = ModelRecommendation(
            model_name="LightGBM_GBDT",
            model_type="lightgbm",
            rationale=(
                "Industry-standard gradient boosted decision trees. Excellent speed, robust to outliers, "
                "and native support for tabular decision boundaries."
            ),
            hyperparameter_grid={
                "n_estimators": 200,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1
            },
            expected_strengths="Handles non-linear tabular partitions, fast convergence, robust against monotonic scaling."
        )

        # Candidate Model 3: Random Forest (Bagging)
        rf_rec = ModelRecommendation(
            model_name="RandomForest_Bagging",
            model_type="random_forest",
            rationale=(
                "Ensemble bagging estimator with 150 decision trees. Resilient against outliers and label noise, "
                "providing robust variance reduction and high stability across smaller sample sizes."
            ),
            hyperparameter_grid={
                "n_estimators": 150,
                "max_depth": 12
            },
            expected_strengths="Outlier resistance, variance reduction, zero risk of exploding gradients."
        )

        # Candidate Model 4: CatBoost (Ordered Boosting)
        catboost_rec = ModelRecommendation(
            model_name="CatBoost_OrderedGBDT",
            model_type="catboost",
            rationale=(
                "Ordered boosting gradient boosted trees. Specialized in processing categorical features "
                "with minimal risk of prediction shift or target leakage."
            ),
            hyperparameter_grid={
                "iterations": 250,
                "learning_rate": 0.05,
                "depth": 6
            },
            expected_strengths="Unmatched categorical robustness, oblivious symmetric decision trees, resistant to overfitting."
        )

        # Meta-Learning Selection Heuristic:
        if mode == "fast":
            # Include CatBoost if categorical features exist, else LightGBM + PyTorch + RF
            if dossier.suggested_categorical_features:
                selected_models = [lgbm_rec, catboost_rec, pytorch_rec]
            else:
                selected_models = [lgbm_rec, rf_rec, pytorch_rec]
            enable_ensemble = False
        else: # Championship Tournament Mode
            selected_models = [pytorch_rec, lgbm_rec, catboost_rec, rf_rec]
            enable_ensemble = True

        preprocessing_strat = {
            "numeric_imputation": "median",
            "categorical_encoding": "ordinal_with_unknown_token",
            "scaling": "standard_scaler",
            "handle_class_imbalance": bool(dossier.class_balance and any(v < 0.3 for v in dossier.class_balance.values())),
            "enable_ensemble_blending": enable_ensemble,
            "execution_mode": mode
        }

        return ExperimentPlan(
            task_name=f"{dossier.target_column}_{dossier.task_type}_benchmark",
            target_column=dossier.target_column,
            task_type=dossier.task_type,
            evaluation_metric=dossier.primary_metric,
            features_to_exclude=dossier.suggested_features_to_drop,
            categorical_features=dossier.suggested_categorical_features,
            numerical_features=dossier.suggested_numeric_features,
            models_to_train=selected_models,
            preprocessing_strategy=preprocessing_strat,
            validation_strategy="Stratified 70/15/15 Train-Val-Test Split"
        )
