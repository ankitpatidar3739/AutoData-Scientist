"""
Data Profiling and Leakage Detection Agent.
Performs statistical profiling, column type inference, and flags potential target leakage and ID columns.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from src.agents.base import BaseAgent
from src.core.config import DataDossier, ColumnProfile

class DataProfilerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="DataProfiler", role="Data Profiling & Leakage Detection Specialist")

    def profile(self, df: pd.DataFrame, target_col: str, primary_metric: Optional[str] = None) -> DataDossier:
        """Profiles the dataset, detects leakage and data types, and produces a DataDossier."""
        n_rows, n_cols = df.shape
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

        # Infer task type
        y = df[target_col].dropna()
        n_unique_y = int(y.nunique())

        def clean_dirty_numeric(s: pd.Series) -> pd.Series:
            return pd.to_numeric(
                s.astype(str)
                .str.replace(r'[\$,₹€£]', '', regex=True)
                .str.replace(',', '', regex=False)
                .str.replace(r'\s*(kms|km|kmpl|cc|bhp|kg|years|yrs|%)\b', '', regex=True, case=False)
                .str.strip(),
                errors='coerce'
            )

        # Test if target is genuinely numeric or dirty numeric (e.g. "80,000", "4,25,000", "$100", "Ask For Price")
        y_numeric_test = pd.to_numeric(y, errors="coerce")
        is_numeric = (y_numeric_test.notna().sum() / max(len(y), 1)) > 0.85

        if not is_numeric:
            y_cleaned = clean_dirty_numeric(y)
            if (y_cleaned.notna().sum() / max(len(y), 1)) > 0.60:
                is_numeric = True
                y = y_cleaned.dropna()
                n_unique_y = int(y.nunique())

        if is_numeric:
            if n_unique_y == 2:
                task_type = "binary_classification"
                primary_metric = primary_metric or "roc_auc"
            elif 2 < n_unique_y <= 10 and pd.api.types.is_integer_dtype(y):
                task_type = "multiclass_classification"
                primary_metric = primary_metric if primary_metric in ("f1", "accuracy") else "f1"
            else:
                task_type = "regression"
                primary_metric = primary_metric or "rmse"
        else:
            # Non-numeric target (string/categorical/text)
            if n_unique_y == 2:
                task_type = "binary_classification"
                primary_metric = primary_metric or "roc_auc"
            elif 2 < n_unique_y <= 50:
                task_type = "multiclass_classification"
                primary_metric = primary_metric if primary_metric in ("f1", "accuracy") else "f1"
            else:
                other_cols = [c for c in df.columns if c != target_col]
                suggest = f" Did you mean to select '{other_cols[0]}' as target?" if other_cols else ""
                raise ValueError(
                    f"Selected target column '{target_col}' contains non-numeric text with {n_unique_y} unique values (e.g. sentences/messages). "
                    f"A target column must be numeric for regression, or categorical (<= 50 classes) for classification.{suggest}"
                )

        class_balance = None
        if task_type in ("binary_classification", "multiclass_classification"):
            counts = y.value_counts(normalize=True).to_dict()
            class_balance = {str(k): round(float(v), 4) for k, v in counts.items()}

        columns_profile: List[ColumnProfile] = []
        features_to_drop = []
        categorical_features = []
        numeric_features = []
        high_missing_features = []
        potential_leakage = []

        # Target series for correlation/leakage check
        y_numeric = None
        if task_type == "binary_classification":
            y_numeric = (y == y.unique()[0]).astype(int)
        elif task_type == "multiclass_classification":
            y_numeric = pd.Series(pd.factorize(y)[0], index=y.index)
        elif task_type == "regression":
            y_numeric = pd.to_numeric(y, errors="coerce")

        for col in df.columns:
            series = df[col]
            missing_count = int(series.isna().sum())
            missing_pct = round(float(missing_count / n_rows), 4)
            n_unique = int(series.nunique())
            sample_vals = series.dropna().head(3).tolist()

            # Column type inference
            inferred_type = "numeric"
            mean_val = std_val = min_val = max_val = None
            top_cats = None

            if missing_pct > 0.60:
                high_missing_features.append(col)

            # Check if constant or ID column
            is_id_column = False
            col_lower = col.lower()
            avg_str_len = series.dropna().astype(str).str.len().mean() if not series.dropna().empty else 0
            is_free_text = (series.dtype == "object" or pd.api.types.is_string_dtype(series)) and avg_str_len > 25 and n_unique > 0.4 * n_rows

            if n_unique <= 1:
                inferred_type = "id_or_constant"
                features_to_drop.append(col)
            elif ("id" in col_lower or "guid" in col_lower or "uuid" in col_lower) and n_unique > 0.8 * n_rows:
                inferred_type = "id_or_constant"
                is_id_column = True
                features_to_drop.append(col)
            # Check if dirty numeric column (e.g. "45,000 kms", "1500 cc", "$100", commas)
            cleaned_num_series = None
            if not pd.api.types.is_numeric_dtype(series):
                cleaned_num_series = clean_dirty_numeric(series)

            is_dirty_numeric = (
                cleaned_num_series is not None 
                and (cleaned_num_series.notna().sum() / max(len(series.dropna()), 1)) > 0.60
                and cleaned_num_series.nunique() > 10
            )

            if is_dirty_numeric:
                inferred_type = "numeric"
                if col != target_col:
                    numeric_features.append(col)
                valid_num = cleaned_num_series.dropna()
                mean_val = round(float(valid_num.mean()), 3) if not valid_num.empty else None
                std_val = round(float(valid_num.std()), 3) if not valid_num.empty else None
                min_val = round(float(valid_num.min()), 3) if not valid_num.empty else None
                max_val = round(float(valid_num.max()), 3) if not valid_num.empty else None
            elif is_free_text:
                inferred_type = "text"
                features_to_drop.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                inferred_type = "numeric"
                if col != target_col:
                    numeric_features.append(col)
                mean_val = round(float(series.mean()), 3) if not series.dropna().empty else None
                std_val = round(float(series.std()), 3) if not series.dropna().empty else None
                min_val = round(float(series.min()), 3) if not series.dropna().empty else None
                max_val = round(float(series.max()), 3) if not series.dropna().empty else None
            elif pd.api.types.is_datetime64_any_dtype(series):
                inferred_type = "datetime"
                features_to_drop.append(col)
            else:
                inferred_type = "categorical"
                if col != target_col:
                    categorical_features.append(col)
                top_cats = series.value_counts().head(5).to_dict()
                top_cats = {str(k): int(v) for k, v in top_cats.items()}

            # Target Leakage Analysis
            leakage_risk = False
            leakage_reason = None
            target_corr = None

            if col != target_col and y_numeric is not None:
                if inferred_type == "numeric":
                    num_series = cleaned_num_series if is_dirty_numeric else pd.to_numeric(series, errors="coerce")
                    if num_series is not None:
                        valid_idx = num_series.dropna().index.intersection(y_numeric.dropna().index)
                        if len(valid_idx) > 20:
                            try:
                                corr = np.corrcoef(num_series.loc[valid_idx].astype(float), y_numeric.loc[valid_idx].astype(float))[0, 1]
                                if not np.isnan(corr):
                                    target_corr = round(float(abs(corr)), 4)
                                    if target_corr > 0.95:
                                        leakage_risk = True
                                        leakage_reason = f"Extreme correlation ({target_corr:.2f}) with target."
                                        potential_leakage.append(col)
                                        features_to_drop.append(col)
                            except Exception:
                                pass

            # Suspicious feature name keywords (e.g. churn_reason, canceled_date)
            leakage_keywords = ["churn_reason", "cancellation", "exit_date", "post_event", "fraud_label"]
            if any(k in col_lower for k in leakage_keywords) and col != target_col:
                leakage_risk = True
                leakage_reason = f"Column name implies post-event data leakage."
                potential_leakage.append(col)
                features_to_drop.append(col)

            columns_profile.append(ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                inferred_type=inferred_type,
                missing_count=missing_count,
                missing_percentage=missing_pct,
                unique_count=n_unique,
                sample_values=sample_vals,
                mean=mean_val,
                std=std_val,
                min=min_val,
                max=max_val,
                top_categories=top_cats,
                target_correlation_or_mi=target_corr,
                leakage_risk=leakage_risk,
                leakage_reason=leakage_reason
            ))

        # Filter dropped features from active lists
        features_to_drop = list(set(features_to_drop))
        categorical_features = [c for c in categorical_features if c not in features_to_drop]
        numeric_features = [c for c in numeric_features if c not in features_to_drop]

        if not numeric_features and not categorical_features:
            raise ValueError(
                f"No usable predictive features found in dataset. All non-target columns ({features_to_drop}) "
                f"were identified as identifiers, timestamps, or high-cardinality unstructured text. "
                f"Machine learning models require at least one numerical or categorical feature to learn from."
            )

        summary_notes = [
            f"Dataset contains {n_rows} rows and {n_cols} columns.",
            f"Identified task as '{task_type}' with target '{target_col}'.",
            f"Detected {len(numeric_features)} continuous features and {len(categorical_features)} categorical features.",
        ]
        if features_to_drop:
            summary_notes.append(f"Recommended dropping {len(features_to_drop)} columns: {features_to_drop} (IDs, high missingness, or target leakage).")
        if class_balance and task_type == "binary_classification":
            summary_notes.append(f"Class distribution: {class_balance}")

        return DataDossier(
            row_count=n_rows,
            col_count=n_cols,
            target_column=target_col,
            task_type=task_type,
            primary_metric=primary_metric,
            class_balance=class_balance,
            columns=columns_profile,
            suggested_features_to_drop=features_to_drop,
            suggested_categorical_features=categorical_features,
            suggested_numeric_features=numeric_features,
            high_missing_features=high_missing_features,
            potential_leakage_features=potential_leakage,
            summary_notes=summary_notes
        )
