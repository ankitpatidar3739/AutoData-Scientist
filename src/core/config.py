"""
Core configuration schemas and data structures for AutoDataScientist.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
import os

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    inferred_type: Literal["numeric", "categorical", "datetime", "text", "id_or_constant"]
    missing_count: int
    missing_percentage: float
    unique_count: int
    sample_values: List[Any]
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    top_categories: Optional[Dict[str, int]] = None
    target_correlation_or_mi: Optional[float] = None
    leakage_risk: bool = False
    leakage_reason: Optional[str] = None

class DataDossier(BaseModel):
    row_count: int
    col_count: int
    target_column: str
    task_type: Literal["binary_classification", "multiclass_classification", "regression"]
    primary_metric: str
    class_balance: Optional[Dict[str, float]] = None
    columns: List[ColumnProfile]
    suggested_features_to_drop: List[str] = Field(default_factory=list)
    suggested_categorical_features: List[str] = Field(default_factory=list)
    suggested_numeric_features: List[str] = Field(default_factory=list)
    high_missing_features: List[str] = Field(default_factory=list)
    potential_leakage_features: List[str] = Field(default_factory=list)
    summary_notes: List[str] = Field(default_factory=list)

class ModelRecommendation(BaseModel):
    model_name: str
    model_type: Literal["pytorch_tabular", "lightgbm", "xgboost", "random_forest", "catboost"]
    rationale: str
    hyperparameter_grid: Dict[str, Any]
    expected_strengths: str

class ExperimentPlan(BaseModel):
    task_name: str
    target_column: str
    task_type: Literal["binary_classification", "multiclass_classification", "regression"]
    evaluation_metric: str
    features_to_exclude: List[str]
    categorical_features: List[str]
    numerical_features: List[str]
    models_to_train: List[ModelRecommendation]
    preprocessing_strategy: Dict[str, Any]
    validation_strategy: str

class ExecutionResult(BaseModel):
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time_seconds: float
    error_summary: Optional[str] = None
    error_traceback: Optional[str] = None
    artifacts_generated: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

class MLReview(BaseModel):
    approved: bool
    verdict: Literal["approved", "needs_iteration", "rejected"]
    overfitting_detected: bool
    leakage_detected: bool
    train_score: Optional[float] = None
    val_score: Optional[float] = None
    test_score: Optional[float] = None
    generalization_gap: Optional[float] = None
    critique: str
    suggested_improvements: List[str] = Field(default_factory=list)

class Settings(BaseModel):
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    sandbox_timeout_seconds: int = Field(default_factory=lambda: int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "180")))
    max_self_healing_retries: int = Field(default_factory=lambda: int(os.getenv("MAX_SELF_HEALING_RETRIES", "3")))
    device: str = Field(default_factory=lambda: os.getenv("DEVICE", "cpu"))
    random_seed: int = Field(default_factory=lambda: int(os.getenv("DEFAULT_RANDOM_SEED", "42")))

settings = Settings()
