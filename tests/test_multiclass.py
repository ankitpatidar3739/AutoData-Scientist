"""
Integration test for multiclass classification (e.g. Titanic Embarked target with S, C, Q).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.orchestrator import AutoDataScientistOrchestrator

def test_multiclass_pipeline(tmp_path):
    # Synthesize small Titanic-like dataset
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "PassengerId": [f"P_{i}" for i in range(n)],
        "Pclass": np.random.choice([1, 2, 3], size=n),
        "Sex": np.random.choice(["male", "female"], size=n),
        "Age": np.random.normal(30, 12, size=n),
        "Fare": np.random.exponential(32, size=n),
        "Embarked": np.random.choice(["S", "C", "Q"], size=n, p=[0.7, 0.2, 0.1])
    })
    # Add a couple of NaNs in Embarked like real Titanic
    df.loc[0, "Embarked"] = np.nan
    df.loc[1, "Embarked"] = np.nan

    csv_path = tmp_path / "titanic_mock.csv"
    df.to_csv(csv_path, index=False)

    orchestrator = AutoDataScientistOrchestrator(output_root=str(tmp_path))
    result = orchestrator.run_pipeline(
        dataset_path=str(csv_path),
        target_column="Embarked",
        primary_metric="f1"
    )

    assert result["success"] is True, f"Multiclass pipeline failed: {result.get('error')}"
    assert result["dossier"].task_type == "multiclass_classification"
    assert "pytorch" in result["metrics"]
    assert "lightgbm" in result["metrics"]
