"""
Multi-Agent Orchestrator coordinating:
1. Data Profiler Agent (Statistical profiling & anti-leakage guards)
2. ML Strategist Agent (Experiment & architecture planning)
3. Code Generator Agent (Sandboxed self-healing code execution)
4. ML Critic Agent (Overfitting & generalization review)
5. Model Explainer Agent (SHAP attribution & executive reporting)
"""

import os
import json
import logging
import traceback
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import pandas as pd

from src.core.config import DataDossier, ExperimentPlan, ExecutionResult, MLReview
from src.agents.profiler import DataProfilerAgent
from src.agents.strategist import MLStrategistAgent
from src.agents.coder import CodeGeneratorAgent
from src.agents.critic import MLCriticAgent
from src.agents.explainer import ModelExplainerAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AutoDataScientist")

class AutoDataScientistOrchestrator:
    def __init__(self, output_root: str = "artifacts"):
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.profiler = DataProfilerAgent()
        self.strategist = MLStrategistAgent()
        self.coder = CodeGeneratorAgent()
        self.critic = MLCriticAgent()
        self.explainer = ModelExplainerAgent()

    def run_pipeline(
        self,
        dataset_path: str,
        target_column: str,
        primary_metric: Optional[str] = None,
        mode: str = "championship",
        progress_callback: Optional[Callable[[str, str, Any], None]] = None
    ) -> Dict[str, Any]:
        """
        Runs the full autonomous ML/Deep Learning engineering loop.
        progress_callback: fn(agent_name, status_message, data_payload)
        """
        def notify(agent: str, msg: str, data: Any = None):
            logger.info(f"[{agent}] {msg}")
            if progress_callback:
                progress_callback(agent, msg, data)

        dataset_file = Path(dataset_path).resolve()
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        run_id = f"run_{dataset_file.stem}_{target_column}"
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # STEP 1: Data Profiling & Leakage Detection
        # -------------------------------------------------------------
        notify("DataProfiler", f"Reading and profiling dataset: {dataset_file.name}...")
        try:
            df = pd.read_csv(dataset_file)
            dossier = self.profiler.profile(df, target_col=target_column, primary_metric=primary_metric)
        except Exception as e:
            logger.error(f"Data profiling error: {e}")
            notify("DataProfiler", f"Data profiling failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "dossier": None,
                "plan": None,
                "metrics": {}
            }
        notify("DataProfiler", f"Profiling complete! Found {dossier.row_count} rows, {dossier.col_count} columns.", dossier)

        # -------------------------------------------------------------
        # STEP 2: ML Strategy & Architecture Planning
        # -------------------------------------------------------------
        notify("MLStrategist", f"Designing candidate architectures (Meta-learning mode: {mode.upper()})...")
        try:
            plan = self.strategist.create_plan(dossier, mode=mode)
        except Exception as e:
            logger.error(f"ML Strategist error: {e}")
            notify("MLStrategist", f"Strategy planning failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "dossier": dossier,
                "plan": None,
                "metrics": {}
            }
        notify("MLStrategist", f"Experiment plan finalized for task '{plan.task_type}' targeting '{plan.evaluation_metric}'.", plan)

        # -------------------------------------------------------------
        # STEP 3: Sandboxed Code Generation & Self-Healing Loop
        # -------------------------------------------------------------
        notify("MLCoder", "Generating Python training script and initiating sandboxed execution...")
        exec_res, retries, final_code = self.coder.run_with_self_healing(plan, str(dataset_file), str(run_dir))
        
        if not exec_res.success:
            notify("MLCoder", f"Execution failed after {retries} retries. Error: {exec_res.error_summary}", exec_res)
            return {
                "success": False,
                "error": exec_res.error_summary,
                "traceback": exec_res.error_traceback,
                "dossier": dossier,
                "plan": plan
            }

        notify("MLCoder", f"Pipeline executed successfully in {exec_res.execution_time_seconds:.1f}s (Healing retries: {retries})!", exec_res.metrics)

        # -------------------------------------------------------------
        # STEP 4: ML Critic & Overfitting Diagnostics
        # -------------------------------------------------------------
        notify("MLCritic", "Auditing model performance, generalization gap, and overfitting risk...")
        review = self.critic.evaluate_results(exec_res.metrics, plan)
        notify("MLCritic", f"Review complete. Verdict: {review.verdict.upper()}. {review.critique}", review)

        # -------------------------------------------------------------
        # STEP 5: SHAP Explainability & Executive Reporting
        # -------------------------------------------------------------
        notify("ModelExplainer", "Computing feature attributions and compiling executive report...")
        report_path = self.explainer.explain_and_report(dossier, plan, exec_res.metrics, review, str(run_dir))
        notify("ModelExplainer", f"Report and charts generated at: {report_path}", report_path)

        return {
            "success": True,
            "run_dir": str(run_dir),
            "dossier": dossier,
            "plan": plan,
            "execution": exec_res,
            "review": review,
            "report_path": report_path,
            "metrics": exec_res.metrics,
            "healing_retries": retries
        }

if __name__ == "__main__":
    import sys
    sample_csv = "data/samples/customer_churn.csv"
    orch = AutoDataScientistOrchestrator()
    print("Running AutoDataScientist on sample churn dataset...")
    res = orch.run_pipeline(sample_csv, target_column="churn")
    print(f"Run completed! Success={res['success']}")
    if res["success"]:
        print(f"Executive Report: {res['report_path']}")
        print(f"Review Verdict: {res['review'].verdict}")
