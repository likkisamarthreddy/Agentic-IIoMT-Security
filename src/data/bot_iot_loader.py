# -*- coding: utf-8 -*-
"""
Data Preprocessor for BoT-IoT dataset.
Handles specific BoT-IoT label parsing and feature encoding.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from data.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)

class BotIoTLoader(DataPreprocessor):
    """
    DataPreprocessor tailored for the BoT-IoT dataset.
    Automatically handles categorical columns and maps the target label.
    """
    
    def __init__(self, config_path: Optional[Path] = None) -> None:
        super().__init__(config_path)
    
    def prepare_pipeline(
        self,
        df: pd.DataFrame,
        test_ratio: Optional[float] = None,
        window_size: Optional[int] = None,
        label_col: str = "category",  # Default for BoT-IoT
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Execute the full preprocessing pipeline with BoT-IoT specific handling.
        """
        logger.info("Starting BotIoT prepare_pipeline")
        
        # Determine actual label column if 'category' isn't present
        if label_col not in df.columns:
            if 'attack' in df.columns:
                label_col = 'attack'
            else:
                logger.warning(f"Default label '{label_col}' not found. Using last column.")
                label_col = df.columns[-1]

        # 1. Clean
        df = self.clean(df)
        
        # 1.5 Handle Categorical Features
        # Drop identifiers if they exist
        drop_cols = ['pkSeqID', 'seq', 'stime', 'ltime', 'flgs', 'flgs_number', 'state_number']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if label_col in categorical_cols:
            categorical_cols.remove(label_col)
            
        if categorical_cols:
            logger.info(f"One-hot encoding categorical columns: {categorical_cols}")
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # 2. Encode Labels
        df, label_mapping = self.encode_labels(df, label_col=label_col)

        # Use base class pipeline from Split onwards
        test_ratio = test_ratio or self._test_ratio
        window_size = window_size or self._window_size

        feature_cols = [c for c in df.columns if c != label_col]
        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values.astype(np.int64)

        from sklearn.model_selection import train_test_split
        
        class_counts = pd.Series(y).value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        valid_mask = np.isin(y, valid_classes)
        
        if not valid_mask.all():
            X = X[valid_mask]
            y = y[valid_mask]
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=self._random_seed, stratify=y
        )
        
        X_train, X_test = self.scale_features(X_train, X_test)
        
        X_train, y_train = self.create_sliding_windows(X_train, y_train, window_size)
        X_test, y_test = self.create_sliding_windows(X_test, y_test, window_size)

        return X_train, X_test, y_train, y_test, label_mapping
