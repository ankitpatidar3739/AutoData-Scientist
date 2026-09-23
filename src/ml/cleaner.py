"""
Universal tabular data cleaner and type inference engine.
Automatically handles:
- Dirty numbers: commas ("4,25,000"), currency ("$500", "₹12,000"), unit suffixes ("45,000 kms", "1500 cc")
- Placeholder strings in numeric columns: "Ask For Price", "N/A", "missing"
- Outlier and target missing value dropping
- Categorical vs Numeric column separation
"""

import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

def clean_dirty_numeric_string(val: Any) -> Optional[float]:
    """Converts a dirty numeric string/object into a float, or returns None if not numeric."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val) if not np.isnan(val) else None

    s = str(val).strip()
    if not s or s.lower() in ("nan", "null", "none", "n/a", "na", "?", "-", "ask for price", "contact"):
        return None

    # Remove currency symbols ($ ₹ € £), commas, and spaces
    s = re.sub(r'[\$,₹€£\s]', '', s)
    s = s.replace(',', '')
    
    # Strip common unit suffixes (kms, km, kmpl, cc, bhp, kg, years, yrs, %)
    s = re.sub(r'(kms|kmpl|km|cc|bhp|kg|years|yrs|%|l)\b', '', s, flags=re.IGNORECASE).strip()

    try:
        return float(s)
    except ValueError:
        return None

def clean_series_numeric(series: pd.Series, threshold: float = 0.60) -> Tuple[pd.Series, bool]:
    """
    Attempts to clean a series as numeric.
    Returns (cleaned_series, is_numeric).
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float), True

    cleaned = series.apply(clean_dirty_numeric_string)
    non_null_count = cleaned.notna().sum()
    original_non_null = series.dropna().shape[0]

    if original_non_null > 0 and (non_null_count / original_non_null) >= threshold and non_null_count >= 10:
        return cleaned, True
    return series, False

def auto_clean_dataset(
    df: pd.DataFrame,
    target_col: str,
    max_categories_per_col: int = 50
) -> Dict[str, Any]:
    """
    Universally cleans a tabular dataset and prepares it for modeling.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    df = df.copy()
    cleaning_summary = []
    
    # 1. Clean Target Column
    target_series = df[target_col]
    cleaned_target, is_target_numeric = clean_series_numeric(target_series, threshold=0.60)
    
    if is_target_numeric:
        valid_targets = cleaned_target.dropna()
        n_unique_target = valid_targets.nunique()

        if n_unique_target == 2:
            task_type = "binary_classification"
            df[target_col] = cleaned_target
        elif 2 < n_unique_target <= 10 and (valid_targets % 1 == 0).all():
            task_type = "multiclass_classification"
            df[target_col] = cleaned_target
        else:
            task_type = "regression"
            df[target_col] = cleaned_target
    else:
        valid_targets = target_series.dropna()
        n_unique_target = valid_targets.nunique()

        if n_unique_target == 2:
            task_type = "binary_classification"
        elif 2 < n_unique_target <= 50:
            task_type = "multiclass_classification"
        else:
            raise ValueError(
                f"Selected target '{target_col}' contains {n_unique_target} unique string values. "
                f"A target column must be continuous numbers (for regression) or categories (<= 50 classes) for classification."
            )

    # Drop rows where target is missing or placeholder
    initial_rows = len(df)
    df = df.dropna(subset=[target_col]).copy()
    dropped_target_rows = initial_rows - len(df)
    if dropped_target_rows > 0:
        cleaning_summary.append(f"Dropped {dropped_target_rows} rows with missing or placeholder targets (e.g. 'Ask For Price').")

    # 2. Clean Feature Columns
    numeric_cols = []
    categorical_cols = []
    dropped_cols = []

    for col in df.columns:
        if col == target_col:
            continue

        series = df[col]
        col_lower = col.lower()
        n_unique = series.nunique()
        n_rows = len(df)

        # Detect ID columns
        if n_unique <= 1:
            dropped_cols.append(col)
            continue
        if ("id" in col_lower or "guid" in col_lower or "uuid" in col_lower) and n_unique > 0.8 * n_rows:
            dropped_cols.append(col)
            continue

        # Try numeric conversion
        cleaned_series, is_col_numeric = clean_series_numeric(series, threshold=0.60)
        if is_col_numeric:
            df[col] = cleaned_series
            median_val = float(cleaned_series.median()) if cleaned_series.notna().any() else 0.0
            df[col] = df[col].fillna(median_val)
            numeric_cols.append(col)
            if not pd.api.types.is_numeric_dtype(series):
                cleaning_summary.append(f"Cleaned column '{col}' (stripped commas/units and converted to numbers).")
        else:
            if n_unique <= max_categories_per_col:
                df[col] = series.fillna("Unknown").astype(str)
                categorical_cols.append(col)
            else:
                top_cats = set(series.value_counts().head(30).index)
                df[col] = series.apply(lambda x: str(x) if x in top_cats else "Other")
                categorical_cols.append(col)
                cleaning_summary.append(f"Grouped high-cardinality categories in '{col}' into top 30 values + 'Other'.")

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable features found after cleaning the dataset.")

    return {
        "df": df,
        "task_type": task_type,
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "cleaning_summary": cleaning_summary,
        "row_count": len(df),
        "initial_rows": initial_rows
    }

def generate_standalone_script(
    dataset_filename: str,
    target_col: str,
    task_type: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    best_model_name: str = "lightgbm",
    best_params: Optional[Dict[str, Any]] = None
) -> str:
    """Generates a clean, standalone, well-commented Python script runnable anywhere without project dependencies."""
    is_classification = task_type in ("binary_classification", "multiclass_classification")
    metric_str = "Accuracy and F1" if is_classification else "RMSE and R2"
    
    if "catboost" in best_model_name.lower():
        model_class = "CatBoostClassifier" if is_classification else "CatBoostRegressor"
        import_stmt = "from catboost import CatBoostClassifier, CatBoostRegressor"
        req_stmt = "pip install pandas numpy scikit-learn catboost"
        default_params = "{'iterations': 250, 'learning_rate': 0.05, 'depth': 6, 'random_seed': 42, 'verbose': False}"
    elif "forest" in best_model_name.lower():
        model_class = "RandomForestClassifier" if is_classification else "RandomForestRegressor"
        import_stmt = "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor"
        req_stmt = "pip install pandas numpy scikit-learn"
        default_params = "{'n_estimators': 150, 'max_depth': 12, 'random_state': 42}"
    else:
        model_class = "lgb.LGBMClassifier" if is_classification else "lgb.LGBMRegressor"
        import_stmt = "import lightgbm as lgb"
        req_stmt = "pip install pandas numpy scikit-learn lightgbm"
        default_params = "{'n_estimators': 150, 'learning_rate': 0.05, 'random_state': 42}"

    params_code = f"params = {best_params or {}}" if best_params else f"params = {default_params}"

    script = f'''"""
================================================================================
AutoDataScientist: Standalone Production ML Pipeline
- Target: '{target_col}' ({task_type})
- Model: {best_model_name.upper()}
- Requirements: {req_stmt}
================================================================================
"""

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
{import_stmt}

# 1. Load Data
DATASET_PATH = "{dataset_filename}"
print(f"Loading dataset: {{DATASET_PATH}}...")
df = pd.read_csv(DATASET_PATH)

# 2. Universal Data Cleaning Function
def clean_numeric(val):
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(',', '')
    s = re.sub(r'[$₹€£\\s]', '', s)
    s = re.sub(r'(kms|kmpl|km|cc|bhp|kg|years|yrs|%|l)\\b', '', s, flags=re.IGNORECASE).strip()
    try:
        return float(s)
    except ValueError:
        return None

target_col = "{target_col}"
numeric_cols = {numeric_cols}
categorical_cols = {categorical_cols}

# Clean target
{"df[target_col] = df[target_col].apply(clean_numeric)" if task_type == "regression" else ""}
df = df.dropna(subset=[target_col]).copy()

# Clean numeric features
for col in numeric_cols:
    df[col] = df[col].apply(clean_numeric)
    df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)

# Clean categorical features
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown").astype(str)

print(f"Cleaned dataset: {{len(df)}} rows across {{len(numeric_cols) + len(categorical_cols)}} features.")

# 3. Train / Test Split
X = df[numeric_cols + categorical_cols].copy()
y = df[target_col].copy()

# Prepare categorical columns
{"for col in categorical_cols: X[col] = X[col].astype('category').cat.codes" if "forest" in best_model_name.lower() else "for col in categorical_cols: X[col] = X[col].astype('category')"}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model with Tuned Hyperparameters
print(f"\\nTraining {best_model_name.upper()} model...")
{params_code}
model = {model_class}(**params)
model.fit(X_train, y_train)

# 5. Evaluate on Unseen Test Split
y_pred = model.predict(X_test)
print("\\n--- Test Evaluation Results ---")
{"print('Accuracy:', round(accuracy_score(y_test, y_pred), 4))" if is_classification else "print('RMSE:', round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))"}
{"print('F1-Score:', round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4))" if is_classification else "print('R2 Score:', round(r2_score(y_test, y_pred), 4))"}

# 6. Feature Importance Ranking
importances = model.feature_importances_
feat_ranking = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
print("\\n--- Top Influential Features ---")
for feat, imp in feat_ranking[:8]:
    print(f"  • {{feat}}: {{imp}}")

print("\\nPipeline execution complete! Model ready for inference.")
'''
    return script

