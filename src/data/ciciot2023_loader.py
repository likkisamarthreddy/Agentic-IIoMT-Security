import os
import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger("ciciot2023.loader")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATASET_DIR = _PROJECT_ROOT / "datasets" / "CICIOT2023"

@dataclass
class CICIotData:
    X_train: np.ndarray          # (N, 1, F) float32
    X_test: np.ndarray           # (M, 1, F) float32
    y_train: np.ndarray          # (N,) int64
    y_test: np.ndarray           # (M,) int64
    feature_names: List[str]
    label_mapping: Dict[str, int]
    scaler: StandardScaler
    num_features: int
    num_classes: int

class CICIot2023Loader:
    def __init__(self, test_ratio: float = 0.2, max_rows_per_file: int = None):
        self.test_ratio = test_ratio
        self.max_rows_per_file = max_rows_per_file

    def prepare(self) -> CICIotData:
        csv_files = glob.glob(str(_DATASET_DIR / "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {_DATASET_DIR}")
        
        train_dfs = []
        test_dfs = []
        
        for csv_file in csv_files:
            logger.info(f"Loading {csv_file}...")
            # Use engine="c" for speed
            df = pd.read_csv(csv_file, low_memory=False, engine="c")
            
            # The first column is usually an unnamed index, let's drop it if it has no name
            if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
                df = df.drop(columns=[df.columns[0]])
                
            if self.max_rows_per_file:
                df = df.head(self.max_rows_per_file)
                
            # Clean NaN/Inf
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            # Temporal / Sequential Split (80/20) for each file
            split_idx = int(len(df) * (1 - self.test_ratio))
            train_dfs.append(df.iloc[:split_idx])
            test_dfs.append(df.iloc[split_idx:])
            
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        logger.info(f"Train set shape: {train_df.shape}, Test set shape: {test_df.shape}")
        
        # Label encode target
        label_column = "Label"
        if label_column not in train_df.columns:
            raise KeyError(f"Column '{label_column}' not found. Available: {train_df.columns}")
            
        y_train_raw = train_df[label_column].astype(str).values
        y_test_raw = test_df[label_column].astype(str).values
        
        target_le = LabelEncoder()
        # Fit on both to ensure all classes are captured
        all_y = np.concatenate([y_train_raw, y_test_raw])
        target_le.fit(all_y)
        
        y_train = target_le.transform(y_train_raw).astype(np.int64)
        y_test = target_le.transform(y_test_raw).astype(np.int64)
        
        label_mapping = {cls: int(i) for i, cls in enumerate(target_le.classes_)}
        
        # Process features
        train_df = train_df.drop(columns=[label_column])
        test_df = test_df.drop(columns=[label_column])
        
        # Convert any remaining objects to numeric if necessary
        for c in train_df.columns:
            if train_df[c].dtype == object:
                le = LabelEncoder()
                # Need to fit on both to avoid un-seen labels
                all_vals = pd.concat([train_df[c], test_df[c]]).astype(str)
                le.fit(all_vals)
                train_df[c] = le.transform(train_df[c].astype(str))
                test_df[c] = le.transform(test_df[c].astype(str))
                
        train_df = train_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        test_df = test_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        
        feature_names = list(train_df.columns)
        X_train = train_df.values.astype(np.float32)
        X_test = test_df.values.astype(np.float32)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        
        X_train = np.expand_dims(X_train, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        
        return CICIotData(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            feature_names=feature_names, label_mapping=label_mapping,
            scaler=scaler, num_features=X_train.shape[2], num_classes=len(label_mapping)
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = CICIot2023Loader(max_rows_per_file=10000)
    data = loader.prepare()
    print("Done.", data.X_train.shape, data.num_classes)
