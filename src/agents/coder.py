"""
Autonomous Coder Agent with Self-Healing Execution Sandbox.
Generates training pipelines, runs them in isolated subprocesses, analyzes tracebacks,
and iteratively fixes code until successful convergence.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from src.agents.base import BaseAgent
from src.core.config import ExperimentPlan, ExecutionResult, settings
from src.core.sandbox import CodeSandbox

logger = logging.getLogger(__name__)

class CodeGeneratorAgent(BaseAgent):
    def __init__(self, sandbox: Optional[CodeSandbox] = None):
        super().__init__(name="MLCoder", role="Autonomous ML Software & Deep Learning Engineer")
        self.sandbox = sandbox or CodeSandbox()
        self.max_retries = settings.max_self_healing_retries

    def generate_pipeline_code(self, plan: ExperimentPlan, dataset_path: str, output_dir: str) -> str:
        """Generates a complete, runnable Python script to execute the ExperimentPlan."""
        # Sanitize paths for Windows backslash escaping
        dataset_path_clean = Path(dataset_path).resolve().as_posix()
        output_dir_clean = Path(output_dir).resolve().as_posix()

        code = f'''"""
Auto-generated ML & Deep Learning Pipeline
Produced autonomously by AutoDataScientist Coder Agent
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.ml.preprocessor import TabularPreprocessor, PyTorchTabularDataset
from src.ml.architectures import TabularDeepNet
from src.ml.trainer import PyTorchTrainer, LightGBMTrainer, CatBoostTrainer, RandomForestTrainer, EnsembleBlender, calculate_metrics

def run():
    dataset_path = "{dataset_path_clean}"
    output_dir = "{output_dir_clean}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset from: {{dataset_path}}")
    df = pd.read_csv(dataset_path)

    target_col = "{plan.target_column}"
    task_type = "{plan.task_type}"
    numeric_cols = {plan.numerical_features}
    categorical_cols = {plan.categorical_features}

    print(f"Dataset Shape: {{df.shape}}")
    print(f"Target: {{target_col}} | Task Type: {{task_type}}")

    # 1. Preprocessing
    print("Fitting TabularPreprocessor on training folds...")
    preprocessor = TabularPreprocessor(
        target_col=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        task_type=task_type,
        random_state=42
    )
    train_data, val_data, test_data = preprocessor.fit_transform(df)
    preprocessor.save(os.path.join(output_dir, "preprocessor.pkl"))

    # Prepare PyTorch Datasets & Loaders
    train_ds = PyTorchTabularDataset(train_data["X_cont"], train_data["X_cat"], train_data["y"], task_type=task_type)
    val_ds = PyTorchTabularDataset(val_data["X_cont"], val_data["X_cat"], val_data["y"], task_type=task_type)
    test_ds = PyTorchTabularDataset(test_data["X_cont"], test_data["X_cat"], test_data["y"], task_type=task_type)

    batch_size = 64
    drop_last = len(train_ds) > batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = "{settings.device}"
    print(f"PyTorch Device: {{device}}")

    # 2. Train PyTorch TabularDeepNet
    print("\\n--- Training PyTorch TabularDeepNet ---")
    out_dim = len(preprocessor.target_mapping) if (preprocessor.target_mapping and task_type == "multiclass_classification") else 1
    model = TabularDeepNet(
        num_continuous=len(numeric_cols),
        cat_cardinalities=preprocessor.cat_cardinalities,
        output_dim=out_dim,
        task_type=task_type,
        hidden_dims=[128, 64],
        dropout_rate=0.2,
        use_residual=True
    )
    pytorch_trainer = PyTorchTrainer(
        model=model,
        task_type=task_type,
        learning_rate=1e-3,
        weight_decay=1e-4,
        device=device
    )
    torch_res = pytorch_trainer.train(train_loader, val_loader, epochs=30, patience=6)
    torch.save(model.state_dict(), os.path.join(output_dir, "best_pytorch_model.pt"))

    # Test Evaluation for PyTorch
    test_loss, torch_test_metrics = pytorch_trainer.evaluate(test_loader)
    _, torch_train_metrics = pytorch_trainer.evaluate(train_loader)
    torch_test_preds = pytorch_trainer.predict_proba_or_val(test_loader)
    torch_val_preds = pytorch_trainer.predict_proba_or_val(val_loader)
    print(f"PyTorch Val Metrics: {{torch_res['final_val_metrics']}}")
    print(f"PyTorch Test Metrics: {{torch_test_metrics}}")

    # 3. Train LightGBM Baseline
    print("\\n--- Training LightGBM Baseline ---")
    X_train_gbdt = np.hstack([train_data["X_cont"], train_data["X_cat"]]) if train_data["X_cont"].size > 0 and train_data["X_cat"].size > 0 else (train_data["X_cont"] if train_data["X_cont"].size > 0 else train_data["X_cat"])
    X_val_gbdt = np.hstack([val_data["X_cont"], val_data["X_cat"]]) if val_data["X_cont"].size > 0 and val_data["X_cat"].size > 0 else (val_data["X_cont"] if val_data["X_cont"].size > 0 else val_data["X_cat"])
    X_test_gbdt = np.hstack([test_data["X_cont"], test_data["X_cat"]]) if test_data["X_cont"].size > 0 and test_data["X_cat"].size > 0 else (test_data["X_cont"] if test_data["X_cont"].size > 0 else test_data["X_cat"])

    n_cont = train_data["X_cont"].shape[1] if train_data["X_cont"].size > 0 else 0
    n_cat = train_data["X_cat"].shape[1] if train_data["X_cat"].size > 0 else 0
    cat_indices = list(range(n_cont, n_cont + n_cat)) if n_cat > 0 else None

    lgbm_trainer = LightGBMTrainer(task_type=task_type)
    lgbm_res = lgbm_trainer.train(X_train_gbdt, train_data["y"], X_val_gbdt, val_data["y"], categorical_feature=cat_indices)
    
    if task_type == "binary_classification":
        lgbm_val_preds = lgbm_res["model"].predict_proba(X_val_gbdt)[:, 1]
        lgbm_test_preds = lgbm_res["model"].predict_proba(X_test_gbdt)[:, 1]
    elif task_type == "multiclass_classification":
        lgbm_val_preds = lgbm_res["model"].predict_proba(X_val_gbdt)
        lgbm_test_preds = lgbm_res["model"].predict_proba(X_test_gbdt)
    else:
        lgbm_val_preds = lgbm_res["model"].predict(X_val_gbdt)
        lgbm_test_preds = lgbm_res["model"].predict(X_test_gbdt)
    lgbm_test_metrics = calculate_metrics(test_data["y"], lgbm_test_preds, task_type)
    print(f"LightGBM Val Metrics: {{lgbm_res['val_metrics']}}")
    print(f"LightGBM Test Metrics: {{lgbm_test_metrics}}")

    # 4. Train CatBoost (Ordered Boosting)
    print("\\n--- Training CatBoost (Ordered Boosting) ---")
    catboost_trainer = CatBoostTrainer(task_type=task_type)
    catboost_res = catboost_trainer.train(X_train_gbdt, train_data["y"], X_val_gbdt, val_data["y"])
    if task_type == "binary_classification":
        catboost_val_preds = catboost_res["model"].predict_proba(X_val_gbdt)[:, 1]
        catboost_test_preds = catboost_res["model"].predict_proba(X_test_gbdt)[:, 1]
    elif task_type == "multiclass_classification":
        catboost_val_preds = catboost_res["model"].predict_proba(X_val_gbdt)
        catboost_test_preds = catboost_res["model"].predict_proba(X_test_gbdt)
    else:
        catboost_val_preds = catboost_res["model"].predict(X_val_gbdt)
        catboost_test_preds = catboost_res["model"].predict(X_test_gbdt)
    catboost_test_metrics = calculate_metrics(test_data["y"], catboost_test_preds, task_type)
    print(f"CatBoost Val Metrics: {{catboost_res['val_metrics']}}")
    print(f"CatBoost Test Metrics: {{catboost_test_metrics}}")

    # 5. Train Random Forest (Bagging)
    print("\\n--- Training Random Forest (Bagging) ---")
    rf_trainer = RandomForestTrainer(task_type=task_type)
    rf_res = rf_trainer.train(X_train_gbdt, train_data["y"], X_val_gbdt, val_data["y"])
    if task_type == "binary_classification":
        rf_val_preds = rf_res["model"].predict_proba(X_val_gbdt)[:, 1]
        rf_test_preds = rf_res["model"].predict_proba(X_test_gbdt)[:, 1]
    elif task_type == "multiclass_classification":
        rf_val_preds = rf_res["model"].predict_proba(X_val_gbdt)
        rf_test_preds = rf_res["model"].predict_proba(X_test_gbdt)
    else:
        rf_val_preds = rf_res["model"].predict(X_val_gbdt)
        rf_test_preds = rf_res["model"].predict(X_test_gbdt)
    rf_test_metrics = calculate_metrics(test_data["y"], rf_test_preds, task_type)
    print(f"Random Forest Test Metrics: {{rf_test_metrics}}")

    # 6. Grandmaster Ensemble Stacking & Blending
    print("\\n--- Building Grandmaster Ensemble Blend ---")
    val_preds_map = {{"pytorch": torch_val_preds, "lightgbm": lgbm_val_preds, "catboost": catboost_val_preds, "random_forest": rf_val_preds}}
    test_preds_map = {{"pytorch": torch_test_preds, "lightgbm": lgbm_test_preds, "catboost": catboost_test_preds, "random_forest": rf_test_preds}}

    blender = EnsembleBlender(task_type=task_type, metric_name="{plan.evaluation_metric}")
    diversity_matrix = blender.compute_diversity_matrix(test_preds_map)
    ensemble_weights = blender.fit_blend(val_preds_map, val_data["y"])
    ensemble_test_preds = blender.predict_blend(test_preds_map)
    ensemble_val_preds = blender.predict_blend(val_preds_map)

    ensemble_test_metrics = calculate_metrics(test_data["y"], ensemble_test_preds, task_type)
    ensemble_val_metrics = calculate_metrics(val_data["y"], ensemble_val_preds, task_type)
    print(f"Ensemble Learned Weights: {{ensemble_weights}}")
    print(f"Ensemble Test Metrics: {{ensemble_test_metrics}}")

    # 7. Save Unified Leaderboard & Metrics
    unified_metrics = {{
        "pytorch": {{
            "train_metrics": torch_train_metrics,
            "val_metrics": torch_res["final_val_metrics"],
            "test_metrics": torch_test_metrics,
            "epochs_trained": torch_res["epochs_trained"],
            "history": torch_res["history"]
        }},
        "lightgbm": {{
            "train_metrics": lgbm_res["train_metrics"],
            "val_metrics": lgbm_res["val_metrics"],
            "test_metrics": lgbm_test_metrics,
            "best_iteration": lgbm_res["best_iteration"]
        }},
        "catboost": {{
            "train_metrics": catboost_res["train_metrics"],
            "val_metrics": catboost_res["val_metrics"],
            "test_metrics": catboost_test_metrics,
            "best_iteration": catboost_res["best_iteration"]
        }},
        "random_forest": {{
            "train_metrics": rf_res["train_metrics"],
            "val_metrics": rf_res["val_metrics"],
            "test_metrics": rf_test_metrics
        }},
        "ensemble": {{
            "val_metrics": ensemble_val_metrics,
            "test_metrics": ensemble_test_metrics,
            "weights": ensemble_weights,
            "diversity_matrix": diversity_matrix
        }}
    }}

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(unified_metrics, f, indent=2)
    print(f"METRICS_JSON: {{json.dumps(unified_metrics)}}")
    print("Multi-model tournament and Grandmaster ensemble completed successfully!")

if __name__ == "__main__":
    run()
'''
        return code

    def run_with_self_healing(
        self,
        plan: ExperimentPlan,
        dataset_path: str,
        output_dir: str
    ) -> Tuple[ExecutionResult, int, str]:
        """
        Executes code inside the sandbox.
        If a failure occurs, analyzes the traceback, heals the code, and re-executes.
        """
        current_code = self.generate_pipeline_code(plan, dataset_path, output_dir)
        script_path = os.path.join(output_dir, "run_pipeline.py")

        healing_attempts = 0
        last_result: Optional[ExecutionResult] = None

        while healing_attempts <= self.max_retries:
            # Write script
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(current_code)

            result = self.sandbox.execute_script(script_path)
            last_result = result

            if result.success:
                logger.info(f"Pipeline succeeded on attempt {healing_attempts + 1}!")
                return result, healing_attempts, current_code

            # Execution failed -> Self-Healing Step
            healing_attempts += 1
            if healing_attempts > self.max_retries:
                logger.error(f"Exceeded max healing retries ({self.max_retries}). Last error: {result.error_summary}")
                break

            logger.warning(
                f"Execution failed on attempt {healing_attempts}: {result.error_summary}. "
                f"Invoking Self-Healing Loop..."
            )
            current_code = self._heal_code(current_code, result)

        return last_result, healing_attempts, current_code

    def _heal_code(self, buggy_code: str, failed_result: ExecutionResult) -> str:
        """Analyzes traceback and repairs code either via LLM or rule-based heuristics."""
        prompt = (
            f"You are an expert ML Software Engineer. A Python ML training script failed with this error:\n\n"
            f"Error Summary: {failed_result.error_summary}\n"
            f"Traceback:\n{failed_result.error_traceback}\n\n"
            f"Buggy Code:\n```python\n{buggy_code}\n```\n\n"
            f"Fix the bug and output ONLY the corrected, complete python code in a python markdown block."
        )
        llm_response = self.call_llm(prompt)
        if llm_response and "```python" in llm_response:
            fixed_code = llm_response.split("```python")[1].split("```")[0].strip()
            return fixed_code

        # Deterministic Heuristic Healers for common ML exceptions:
        fixed_code = buggy_code

        # Heuristic 1: CUDA Out of Memory or Device issue
        if "CUDA" in str(failed_result.stderr) or "device" in str(failed_result.stderr):
            fixed_code = fixed_code.replace('device = "cuda"', 'device = "cpu"')

        # Heuristic 2: Batch size too large
        if "out of memory" in str(failed_result.stderr).lower():
            fixed_code = fixed_code.replace("batch_size = 64", "batch_size = 16")

        # Heuristic 3: Dimension mismatch in hstack
        if "all the input array dimensions except for the concatenation axis must match exactly" in str(failed_result.stderr):
            fixed_code = fixed_code.replace(
                "X_train_gbdt = np.hstack([train_data[\"X_cont\"], train_data[\"X_cat\"]])",
                "X_train_gbdt = np.hstack([train_data[\"X_cont\"], train_data[\"X_cat\"]]) if train_data[\"X_cont\"].shape[0] == train_data[\"X_cat\"].shape[0] else train_data[\"X_cont\"]"
            )

        # Heuristic 4: Missing directory
        if "FileNotFoundError" in str(failed_result.stderr):
            fixed_code = f"import os\nos.makedirs(output_dir, exist_ok=True)\n" + fixed_code

        # Heuristic 5: Batch size 1 BatchNorm error
        if "Expected more than 1 value per channel" in str(failed_result.stderr):
            fixed_code = fixed_code.replace("shuffle=True)", "shuffle=True, drop_last=True)")
            fixed_code = fixed_code.replace("nn.BatchNorm1d", "nn.LayerNorm")

        return fixed_code
