"""
ML Critic and Diagnostics Agent.
Analyzes train vs validation curves, detects overfitting and data leakage,
and provides objective model critique before deployment.
"""

from typing import Dict, Any, Optional
from src.agents.base import BaseAgent
from src.core.config import MLReview, ExperimentPlan

class MLCriticAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="MLCritic", role="Model Quality & Guardrails Auditor")

    def evaluate_results(self, metrics: Dict[str, Any], plan: ExperimentPlan) -> MLReview:
        """Inspects metrics for PyTorch and LightGBM models, checking generalization gap and leakage."""
        primary_metric = plan.evaluation_metric
        task_type = plan.task_type

        # Select best model dynamically across all candidate architectures (including ensemble)
        candidate_scores = {}
        for name, data in metrics.items():
            if isinstance(data, dict) and "test_metrics" in data:
                score = data["test_metrics"].get(primary_metric)
                if score is not None:
                    candidate_scores[name] = score

        if not candidate_scores:
            best_model_name = "pytorch"
        elif task_type == "regression" and primary_metric in ("rmse", "mae"):
            best_model_name = min(candidate_scores, key=candidate_scores.get)
        else:
            best_model_name = max(candidate_scores, key=candidate_scores.get)

        best_data = metrics.get(best_model_name, {})
        train_score = best_data.get("train_metrics", {}).get(primary_metric, 0.0)
        val_score = best_data.get("val_metrics", {}).get(primary_metric, 0.0)
        test_score = best_data.get("test_metrics", {}).get(primary_metric, 0.0)

        # Diagnostics
        overfitting = False
        leakage = False
        critiques = []
        improvements = []

        if task_type == "binary_classification":
            gap = train_score - test_score
            if gap > 0.15:
                overfitting = True
                critiques.append(f"High generalization gap ({gap:.3f}). Train {primary_metric}={train_score:.3f} vs Test={test_score:.3f}.")
                improvements.append("Increase dropout rate, add L2 weight decay, or introduce data augmentation.")
            if test_score > 0.99 and train_score > 0.99:
                leakage = True
                critiques.append("Suspiciously perfect performance (Metric > 0.99). High probability of target leakage in features.")
                improvements.append("Audit dataset columns for post-outcome indicators or IDs.")
            if test_score < 0.55:
                critiques.append(f"Model performance is close to random chance ({primary_metric}={test_score:.3f}).")
                improvements.append("Perform feature engineering or verify target correlation.")
        else: # Regression
            gap = test_score - train_score # for RMSE/MAE, lower is better
            rel_gap = gap / (train_score + 1e-6)
            if rel_gap > 0.3:
                overfitting = True
                critiques.append(f"Test RMSE/MAE is {rel_gap:.1%} worse than Train. Model may be memorizing noise.")
                improvements.append("Apply regularization or prune tree depth.")

        # Final verdict
        if leakage:
            verdict = "rejected"
            approved = False
        elif overfitting and (test_score < 0.60 if task_type == "binary_classification" else False):
            verdict = "needs_iteration"
            approved = False
        else:
            verdict = "approved"
            approved = True

        if not critiques:
            critiques.append(f"Model generalized robustly! {best_model_name.upper()} achieved {primary_metric}={test_score:.4f} on unseen test split.")

        return MLReview(
            approved=approved,
            verdict=verdict,
            overfitting_detected=overfitting,
            leakage_detected=leakage,
            train_score=round(float(train_score), 4) if train_score is not None else None,
            val_score=round(float(val_score), 4) if val_score is not None else None,
            test_score=round(float(test_score), 4) if test_score is not None else None,
            generalization_gap=round(float(gap), 4) if 'gap' in locals() else None,
            critique=" | ".join(critiques),
            suggested_improvements=improvements
        )
