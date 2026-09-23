"""
End-to-end integration test running the complete AutoDataScientist orchestrator pipeline.
"""

import pytest
import os
from pathlib import Path
from src.orchestrator import AutoDataScientistOrchestrator

def test_full_pipeline_run(tmp_path):
    sample_csv = Path(__file__).resolve().parent.parent / "data" / "samples" / "customer_churn.csv"
    assert sample_csv.exists(), f"Sample dataset {sample_csv} must exist."

    orchestrator = AutoDataScientistOrchestrator(output_root=str(tmp_path))
    result = orchestrator.run_pipeline(
        dataset_path=str(sample_csv),
        target_column="churn",
        primary_metric="roc_auc"
    )

    assert result["success"] is True, f"Pipeline failed: {result.get('error')}"
    assert "metrics" in result
    assert "pytorch" in result["metrics"]
    assert "lightgbm" in result["metrics"]
    assert Path(result["report_path"]).exists()
    assert result["review"].approved is True or result["review"].verdict in ("approved", "needs_iteration")
