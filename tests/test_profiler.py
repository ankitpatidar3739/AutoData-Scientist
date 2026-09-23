"""
Unit tests for DataProfilerAgent and leakage detection.
"""

import pytest
import pandas as pd
import numpy as np
from src.agents.profiler import DataProfilerAgent

def test_profiler_detects_task_and_columns():
    df = pd.DataFrame({
        "id": [f"ID_{i}" for i in range(100)],
        "age": np.random.randint(18, 70, size=100),
        "income": np.random.normal(50000, 15000, size=100),
        "city": np.random.choice(["NYC", "LA", "Chicago"], size=100),
        "target": np.random.choice([0, 1], size=100)
    })

    profiler = DataProfilerAgent()
    dossier = profiler.profile(df, target_col="target")

    assert dossier.row_count == 100
    assert dossier.col_count == 5
    assert dossier.task_type == "binary_classification"
    assert dossier.primary_metric == "roc_auc"
    # ID column should be flagged to drop
    assert "id" in dossier.suggested_features_to_drop
    assert "age" in dossier.suggested_numeric_features
    assert "city" in dossier.suggested_categorical_features

def test_profiler_detects_target_leakage():
    np.random.seed(42)
    target = np.random.choice([0, 1], size=200)
    # Perfectly correlated feature (leakage)
    leaky_feature = target * 1.0 + np.random.normal(0, 0.001, size=200)

    df = pd.DataFrame({
        "feature_a": np.random.normal(0, 1, size=200),
        "leaky_score": leaky_feature,
        "target": target
    })

    profiler = DataProfilerAgent()
    dossier = profiler.profile(df, target_col="target")

    assert "leaky_score" in dossier.potential_leakage_features
    assert "leaky_score" in dossier.suggested_features_to_drop

def test_profiler_multiclass_string_target():
    # Simulates Titanic Embarked target
    df = pd.DataFrame({
        "age": [22, 38, 26, 35, np.nan, 54, 2, 27, 14, 4],
        "fare": [7.25, 71.28, 7.92, 53.10, 8.05, 51.86, 21.07, 11.13, 30.07, 16.70],
        "embarked": ["S", "C", "S", "S", "Q", "S", "S", "S", "C", np.nan]
    })
    # Convert embarked to pandas StringDtype
    df["embarked"] = df["embarked"].astype("string")

    profiler = DataProfilerAgent()
    dossier = profiler.profile(df, target_col="embarked")

    assert dossier.task_type == "multiclass_classification"
    assert dossier.primary_metric in ("f1", "accuracy")
