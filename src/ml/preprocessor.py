"""
Automated tabular preprocessing and dataset wrapping for PyTorch and GBDTs.
Prevents data leakage by fitting exclusively on training folds.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class PyTorchTabularDataset(Dataset):
    """PyTorch Dataset wrapper for mixed continuous and categorical tabular data."""
    def __init__(self, x_cont: np.ndarray, x_cat: np.ndarray, y: Optional[np.ndarray] = None, task_type: str = "binary_classification"):
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32) if x_cont is not None and x_cont.size > 0 else torch.empty((len(x_cat), 0), dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long) if x_cat is not None and x_cat.size > 0 else torch.empty((len(x_cont), 0), dtype=torch.long)
        if y is not None:
            if task_type == "multiclass_classification":
                self.y = torch.tensor(y, dtype=torch.long)
            else:
                self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.x_cont) if self.x_cont.shape[1] > 0 else len(self.x_cat)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        target = self.y[idx] if self.y is not None else torch.tensor([])
        return self.x_cont[idx], self.x_cat[idx], target

class TabularPreprocessor:
    def __init__(
        self,
        target_col: str,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        task_type: str = "binary_classification",
        test_size: float = 0.2,
        val_size: float = 0.15,
        random_state: int = 42
    ):
        self.target_col = target_col
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
        self.task_type = task_type
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.num_imputer_values: Dict[str, float] = {}
        self.scaler = StandardScaler()
        self.cat_mappings: Dict[str, Dict[Any, int]] = {}
        self.cat_cardinalities: List[int] = []
        self.target_mapping: Optional[Dict[Any, int]] = None
        self.is_fitted: bool = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Fits strictly on train split and returns (train, val, test) data dictionaries."""
        def _clean_numeric_series(s: pd.Series) -> pd.Series:
            if pd.api.types.is_numeric_dtype(s):
                return s
            return pd.to_numeric(
                s.astype(str)
                .str.replace(r'[\$,₹€£]', '', regex=True)
                .str.replace(',', '', regex=False)
                .str.replace(r'\s*(kms|km|kmpl|cc|bhp|kg|years|yrs|%)\b', '', regex=True, case=False)
                .str.strip(),
                errors='coerce'
            )

        # Pre-clean target if regression and contains dirty strings/currency/commas
        if self.task_type == "regression" and not pd.api.types.is_numeric_dtype(df[self.target_col]):
            df = df.copy()
            df[self.target_col] = _clean_numeric_series(df[self.target_col])

        df_clean = df.dropna(subset=[self.target_col]).copy()

        # Extract target
        y_raw = df_clean[self.target_col]
        X_raw = df_clean.drop(columns=[self.target_col])

        # Stratified or standard split (safe against rare classes with < 4 samples)
        stratify = None
        if self.task_type in ("binary_classification", "multiclass_classification"):
            min_class_count = y_raw.value_counts().min()
            if min_class_count >= 4:
                stratify = y_raw

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_raw, y_raw, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )

        val_rel_size = self.val_size / (1.0 - self.test_size)
        stratify_val = None
        if stratify is not None:
            min_val_class_count = y_train_val.value_counts().min()
            if min_val_class_count >= 2:
                stratify_val = y_train_val

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_rel_size, random_state=self.random_state, stratify=stratify_val
        )

        # 1. Fit numerical features
        for col in self.numeric_cols:
            s_num = _clean_numeric_series(X_train[col])
            median_val = float(s_num.median()) if not s_num.dropna().empty else 0.0
            self.num_imputer_values[col] = median_val

        X_train_num = self._transform_num(X_train)
        if self.numeric_cols:
            self.scaler.fit(X_train_num)
            X_train_num = self.scaler.transform(X_train_num)

        # 2. Fit categorical features
        self.cat_cardinalities = []
        for col in self.categorical_cols:
            # 0 is reserved for unknown / missing
            categories = [str(c).strip() for c in X_train[col].dropna().unique().tolist()]
            mapping = {cat: idx + 1 for idx, cat in enumerate(categories)}
            self.cat_mappings[col] = mapping
            # Total unique values = len(categories) + 1 (for unknown)
            self.cat_cardinalities.append(len(categories) + 1)

        X_train_cat = self._transform_cat(X_train)

        # 3. Fit target
        y_train_proc = self._fit_transform_target(y_train)

        self.is_fitted = True

        # Transform validation and test
        X_val_num = self.scaler.transform(self._transform_num(X_val)) if self.numeric_cols else np.empty((len(X_val), 0))
        X_val_cat = self._transform_cat(X_val)
        y_val_proc = self._transform_target(y_val)

        X_test_num = self.scaler.transform(self._transform_num(X_test)) if self.numeric_cols else np.empty((len(X_test), 0))
        X_test_cat = self._transform_cat(X_test)
        y_test_proc = self._transform_target(y_test)

        train_data = {"X_cont": X_train_num, "X_cat": X_train_cat, "y": y_train_proc, "df_raw": X_train}
        val_data = {"X_cont": X_val_num, "X_cat": X_val_cat, "y": y_val_proc, "df_raw": X_val}
        test_data = {"X_cont": X_test_num, "X_cat": X_test_cat, "y": y_test_proc, "df_raw": X_test}

        return train_data, val_data, test_data

    def _transform_num(self, df: pd.DataFrame) -> np.ndarray:
        if not self.numeric_cols:
            return np.empty((len(df), 0), dtype=np.float32)
        arr = np.empty((len(df), len(self.numeric_cols)), dtype=np.float32)
        for i, col in enumerate(self.numeric_cols):
            s = df[col]
            if not pd.api.types.is_numeric_dtype(s):
                s = pd.to_numeric(
                    s.astype(str)
                    .str.replace(r'[\$,₹€£]', '', regex=True)
                    .str.replace(',', '', regex=False)
                    .str.replace(r'\s*(kms|km|kmpl|cc|bhp|kg|years|yrs|%)\b', '', regex=True, case=False)
                    .str.strip(),
                    errors='coerce'
                )
            val = s.fillna(self.num_imputer_values.get(col, 0.0)).to_numpy(dtype=np.float32)
            arr[:, i] = val
        return arr

    def _transform_cat(self, df: pd.DataFrame) -> np.ndarray:
        if not self.categorical_cols:
            return np.empty((len(df), 0), dtype=np.int64)
        arr = np.zeros((len(df), len(self.categorical_cols)), dtype=np.int64)
        for i, col in enumerate(self.categorical_cols):
            mapping = self.cat_mappings.get(col, {})
            # Standardize string representations to match fitted dictionary
            clean_s = df[col].astype(str).str.strip().replace({'nan': '', 'None': '', '<NA>': ''})
            mapped = clean_s.map(mapping).fillna(0).astype(np.int64).to_numpy()
            arr[:, i] = mapped
        return arr

    def _fit_transform_target(self, y: pd.Series) -> np.ndarray:
        if self.task_type == "binary_classification":
            unique_vals = sorted(y.unique().tolist())
            if set(unique_vals).issubset({0, 1}):
                return y.to_numpy(dtype=np.float32)
            # Map classes to 0 and 1
            self.target_mapping = {val: idx for idx, val in enumerate(unique_vals)}
            return y.map(self.target_mapping).to_numpy(dtype=np.float32)
        elif self.task_type == "multiclass_classification":
            unique_vals = sorted(y.unique().tolist())
            self.target_mapping = {val: idx for idx, val in enumerate(unique_vals)}
            return y.map(self.target_mapping).to_numpy(dtype=np.int64)
        else:
            if not pd.api.types.is_numeric_dtype(y):
                y = pd.to_numeric(
                    y.astype(str)
                    .str.replace(r'[\$,₹€£]', '', regex=True)
                    .str.replace(',', '', regex=False)
                    .str.replace(r'\s*(kms|km|kmpl|cc|bhp|kg|years|yrs|%)\b', '', regex=True, case=False)
                    .str.strip(),
                    errors='coerce'
                )
            median_val = float(y.median()) if y.notna().any() else 0.0
            return y.fillna(median_val).to_numpy(dtype=np.float32)

    def _transform_target(self, y: pd.Series) -> np.ndarray:
        if self.target_mapping:
            return y.map(self.target_mapping).fillna(0).to_numpy(dtype=np.float32 if self.task_type == "binary_classification" else np.int64)
        if not pd.api.types.is_numeric_dtype(y):
            y = pd.to_numeric(
                y.astype(str)
                .str.replace(r'[\$,₹€£]', '', regex=True)
                .str.replace(',', '', regex=False)
                .str.replace(r'\s*(kms|km|kmpl|cc|bhp|kg|years|yrs|%)\b', '', regex=True, case=False)
                .str.strip(),
                errors='coerce'
            )
        median_val = float(y.median()) if y.notna().any() else 0.0
        return y.fillna(median_val).to_numpy(dtype=np.float32)

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Transforms new input DataFrame into scaled continuous and encoded categorical arrays."""
        x_num = self._transform_num(df)
        if self.numeric_cols and hasattr(self.scaler, "mean_"):
            x_num = self.scaler.transform(x_num)
        x_cat = self._transform_cat(df)
        return {"X_cont": x_num, "X_cat": x_cat}

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "TabularPreprocessor":
        return joblib.load(filepath)
