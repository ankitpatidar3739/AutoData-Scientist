import numpy as np
import pytest
from src.ml.trainer import CatBoostTrainer, HAS_CATBOOST

@pytest.mark.skipif(not HAS_CATBOOST, reason="CatBoost is not installed")
def test_catboost_trainer_binary():
    np.random.seed(42)
    X_train = np.random.randn(80, 5)
    y_train = np.random.randint(0, 2, 80)
    X_val = np.random.randn(20, 5)
    y_val = np.random.randint(0, 2, 20)

    trainer = CatBoostTrainer(task_type="binary_classification", iterations=30)
    res = trainer.train(X_train, y_train, X_val, y_val)

    assert "model" in res
    assert "val_metrics" in res
    assert "accuracy" in res["val_metrics"]

@pytest.mark.skipif(not HAS_CATBOOST, reason="CatBoost is not installed")
def test_catboost_trainer_regression():
    np.random.seed(42)
    X_train = np.random.randn(80, 5)
    y_train = np.random.randn(80) * 10
    X_val = np.random.randn(20, 5)
    y_val = np.random.randn(20) * 10

    trainer = CatBoostTrainer(task_type="regression", iterations=30)
    res = trainer.train(X_train, y_train, X_val, y_val)

    assert "model" in res
    assert "val_metrics" in res
    assert "rmse" in res["val_metrics"]
