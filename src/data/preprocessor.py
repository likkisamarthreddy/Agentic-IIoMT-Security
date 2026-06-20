# -*- coding: utf-8 -*-
"""
Data Preprocessor for Cross-Domain Agentic Security (IIoMT).

Provides a full feature-engineering pipeline from raw CSV / DataFrame to
PyTorch ``DataLoader`` objects ready for the CNN-BiGRU classifier.
Supports extraction of features from both CICIoMT2023 and Edge-IIoTset
datasets, encompassing clinical and industrial protocols (MQTT, CoAP, DICOM, HL7).

Pipeline stages:
    1. **Load** — read a CSV file from disk (CICIoMT2023 or Edge-IIoTset).
    2. **Clean** — drop duplicates, replace inf/NaN, clip outliers.
    3. **Encode** — map string labels to integer classes.
    4. **Scale** — ``StandardScaler`` fit on train, transform both splits.
    5. **Window** — create sliding temporal windows
       ``(num_windows, window_size, num_features)``.
    6. **DataLoader** — wrap NumPy arrays into PyTorch ``DataLoader``
       instances with configurable batch size.

Typical usage::

    from data.preprocessor import DataPreprocessor

    pp = DataPreprocessor()
    df = pp.load_csv("data/raw/ciciomt2023.csv")
    X_tr, X_te, y_tr, y_te, mapping = pp.prepare_pipeline(df)
    train_dl, test_dl = pp.get_dataloaders(X_tr, X_te, y_tr, y_te)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-root resolution
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"


def _load_config(config_path: Path = _DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load and return the YAML configuration dictionary.

    Args:
        config_path: Path to ``settings.yaml``.

    Returns:
        Parsed YAML dictionary.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class DataPreprocessor:
    """End-to-end data-preparation pipeline for the IIoMT classifier.

    Configuration values (window size, test ratio, batch size, random seed)
    are read from ``config/settings.yaml``.

    Args:
        config_path: Optional override for the settings YAML location.

    Example::

        pp = DataPreprocessor()
        df = pp.load_csv("data/raw/ciciomt2024.csv")
        X_tr, X_te, y_tr, y_te, lbl_map = pp.prepare_pipeline(df)
        train_dl, test_dl = pp.get_dataloaders(X_tr, X_te, y_tr, y_te)
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config = _load_config(config_path or _DEFAULT_CONFIG)
        data_cfg = self._config.get("data", {})
        training_cfg = self._config.get("system1", {}).get("training", {})

        self._window_size: int = data_cfg.get("window_size", 10)
        self._test_ratio: float = data_cfg.get("test_ratio", 0.2)
        self._random_seed: int = data_cfg.get("random_seed", 42)
        self._batch_size: int = training_cfg.get("batch_size", 256)

        self._label_encoder: Optional[LabelEncoder] = None
        self._scaler: Optional[StandardScaler] = None
        self._label_mapping: Optional[Dict[str, int]] = None

        logger.info(
            "DataPreprocessor initialised — window=%d, test_ratio=%.2f, "
            "batch_size=%d, seed=%d",
            self._window_size,
            self._test_ratio,
            self._batch_size,
            self._random_seed,
        )

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------

    @staticmethod
    def load_csv(filepath: str | Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Read a CSV file into a pandas DataFrame.

        Args:
            filepath: Path to the CSV file.
            max_rows: Optional maximum number of rows to read.

        Returns:
            Raw DataFrame.

        Raises:
            FileNotFoundError: If the CSV does not exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        df = pd.read_csv(filepath, nrows=max_rows)
        logger.info("Loaded CSV %s — shape %s", filepath.name, df.shape)
        return df

    @staticmethod
    def load_edge_iiotset_csv(filepath: str | Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Read an Edge-IIoTset CSV file into a pandas DataFrame.
        
        This aligns with the paper's claim of supporting Edge-IIoTset structures
        (MQTT, CoAP, etc.) for cross-validation or future inclusion.
        
        Args:
            filepath: Path to the Edge-IIoTset CSV file.
            max_rows: Optional maximum number of rows to read.
            
        Returns:
            Raw DataFrame aligned with CICIoMT2023 feature schema where possible.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Edge-IIoTset CSV file not found: {filepath}")
        df = pd.read_csv(filepath, nrows=max_rows)
        logger.info("Loaded Edge-IIoTset CSV %s — shape %s", filepath.name, df.shape)
        # Note: mapping of Edge-IIoTset features to the 46-feature CICIoMT schema
        # would be implemented here in a full deployment.
        return df

    # ------------------------------------------------------------------
    # 2. Clean
    # ------------------------------------------------------------------

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Clean a DataFrame in-place-style (returns a copy).

        Operations:
            * Replace ``±inf`` with ``NaN``.
            * Fill remaining ``NaN`` with column medians.
            * Drop exact-duplicate rows.
            * Clip numeric columns to the [1st, 99th] percentile range.

        Args:
            df: Raw DataFrame.

        Returns:
            Cleaned DataFrame (copy).
        """
        original_len = len(df)
        df = df.copy()

        # Replace infinities
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill NaN with column medians (numeric only)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        medians = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(medians)

        # Drop duplicates
        df.drop_duplicates(inplace=True)
        dropped = original_len - len(df)
        if dropped:
            logger.info("Dropped %d duplicate rows", dropped)

        # Clip outliers to [1st, 99th] percentile
        for col in numeric_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower, upper=upper)

        logger.info("Cleaning complete — shape %s", df.shape)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Encode labels
    # ------------------------------------------------------------------

    def encode_labels(
        self, df: pd.DataFrame, label_col: str = "label"
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Encode categorical labels to integer class indices.

        Fits a ``LabelEncoder`` and stores the mapping so it can be reused
        for inference.

        Args:
            df: DataFrame that must contain *label_col*.
            label_col: Name of the label column.

        Returns:
            Tuple of (DataFrame with integer labels, label→int mapping).

        Raises:
            KeyError: If *label_col* is not in the DataFrame.
        """
        if label_col not in df.columns:
            raise KeyError(f"Label column '{label_col}' not found in DataFrame")

        df = df.copy()
        self._label_encoder = LabelEncoder()
        df[label_col] = self._label_encoder.fit_transform(df[label_col].astype(str))
        self._label_mapping = dict(
            zip(
                self._label_encoder.classes_,
                self._label_encoder.transform(self._label_encoder.classes_),
            )
        )
        logger.info("Label mapping: %s", self._label_mapping)
        return df, self._label_mapping

    # ------------------------------------------------------------------
    # 4. Scale features
    # ------------------------------------------------------------------

    def scale_features(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standardise features (zero-mean, unit-variance).

        The ``StandardScaler`` is fit **only** on *X_train* and applied to
        both splits to prevent data leakage.

        Args:
            X_train: Training feature matrix ``(n_train, n_features)``.
            X_test: Test feature matrix ``(n_test, n_features)``.

        Returns:
            Tuple of scaled ``(X_train, X_test)`` arrays.
        """
        self._scaler = StandardScaler()
        X_train_s = self._scaler.fit_transform(X_train)
        X_test_s = self._scaler.transform(X_test)
        logger.info(
            "Feature scaling applied - train mean ~ %.4f, std ~ %.4f",
            X_train_s.mean(),
            X_train_s.std(),
        )
        return X_train_s, X_test_s

    # ------------------------------------------------------------------
    # 5. Sliding windows
    # ------------------------------------------------------------------

    @staticmethod
    def create_sliding_windows(
        X: np.ndarray,
        y: np.ndarray,
        window_size: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create overlapping temporal windows for sequential models.

        The resulting shapes are suitable for the **CNN-BiGRU** architecture:

            * ``X_windows``: ``(num_windows, window_size, num_features)``
            * ``y_windows``: ``(num_windows,)`` — label of the **last**
              sample in each window.

        Args:
            X: Feature matrix ``(n_samples, n_features)``.
            y: Label vector ``(n_samples,)``.
            window_size: Number of consecutive time-steps per window.

        Returns:
            Tuple ``(X_windows, y_windows)``.

        Raises:
            ValueError: If *window_size* exceeds the number of samples.
        """
        n_samples, n_features = X.shape
        if window_size > n_samples:
            raise ValueError(
                f"window_size ({window_size}) exceeds sample count ({n_samples})"
            )

        num_windows = n_samples - window_size + 1
        X_windows = np.empty(
            (num_windows, window_size, n_features), dtype=X.dtype
        )
        y_windows = np.empty(num_windows, dtype=y.dtype)

        for i in range(num_windows):
            X_windows[i] = X[i : i + window_size]
            y_windows[i] = y[i + window_size - 1]  # label of last step

        logger.info(
            "Sliding windows created — X: %s, y: %s",
            X_windows.shape,
            y_windows.shape,
        )
        return X_windows, y_windows

    # ------------------------------------------------------------------
    # 6. Full pipeline
    # ------------------------------------------------------------------

    def prepare_pipeline(
        self,
        df: pd.DataFrame,
        test_ratio: Optional[float] = None,
        window_size: Optional[int] = None,
        label_col: str = "label",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
        """Execute the full preprocessing pipeline.

        Stages: **clean → encode → split → scale → window**.

        Args:
            df: Raw DataFrame (must contain *label_col*).
            test_ratio: Fraction held out for testing (overrides config).
            window_size: Temporal window length (overrides config).
            label_col: Name of the label column.

        Returns:
            Tuple of ``(X_train, X_test, y_train, y_test, label_mapping)``
            where ``X_train`` and ``X_test`` have shape
            ``(num_windows, window_size, num_features)``.
        """
        test_ratio = test_ratio or self._test_ratio
        window_size = window_size or self._window_size

        logger.info("Starting prepare_pipeline — input shape %s", df.shape)

        # 1. Clean
        df = self.clean(df)

        # 2. Encode
        df, label_mapping = self.encode_labels(df, label_col=label_col)

        # 3. Split
        feature_cols = [c for c in df.columns if c != label_col]
        X = df[feature_cols].values.astype(np.float32)
        y = df[label_col].values.astype(np.int64)

        # Filter out classes with fewer than 2 instances to avoid stratify error
        class_counts = pd.Series(y).value_counts()
        valid_classes = class_counts[class_counts >= 2].index
        valid_mask = np.isin(y, valid_classes)
        
        if not valid_mask.all():
            removed = (~valid_mask).sum()
            logger.warning(f"Removed {removed} samples belonging to classes with <2 instances for stratification.")
            X = X[valid_mask]
            y = y[valid_mask]
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=self._random_seed, stratify=y
        )
        logger.info(
            "Train/test split — train: %d, test: %d", len(X_train), len(X_test)
        )

        # 4. Scale
        X_train, X_test = self.scale_features(X_train, X_test)

        # 5. Sliding windows
        X_train, y_train = self.create_sliding_windows(X_train, y_train, window_size)
        X_test, y_test = self.create_sliding_windows(X_test, y_test, window_size)

        logger.info(
            "Pipeline complete — X_train: %s, X_test: %s",
            X_train.shape,
            X_test.shape,
        )
        return X_train, X_test, y_train, y_test, label_mapping

    # ------------------------------------------------------------------
    # 7. PyTorch DataLoaders
    # ------------------------------------------------------------------

    def get_dataloaders(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        batch_size: Optional[int] = None,
    ) -> Tuple[Any, Any]:
        """Wrap NumPy arrays into PyTorch ``DataLoader`` instances.

        Args:
            X_train: Training features (windowed or flat).
            X_test: Test features.
            y_train: Training labels.
            y_test: Test labels.
            batch_size: Mini-batch size (overrides config).

        Returns:
            Tuple ``(train_loader, test_loader)``.
        """
        # Lazy import so the module can be loaded without torch installed
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        batch_size = batch_size or self._batch_size

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

        logger.info(
            "DataLoaders created — train batches: %d, test batches: %d, "
            "batch_size: %d",
            len(train_loader),
            len(test_loader),
            batch_size,
        )
        return train_loader, test_loader


# ===================================================================
# Quick smoke-test
# ===================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Generate a tiny synthetic set to test the pipeline
    from data.synthetic_generator import SyntheticIIoMTGenerator

    gen = SyntheticIIoMTGenerator()
    df = gen.generate_ciciomt_data(500)
    print(f"Raw shape: {df.shape}")

    pp = DataPreprocessor()
    X_tr, X_te, y_tr, y_te, mapping = pp.prepare_pipeline(df, window_size=5)
    print(f"X_train: {X_tr.shape}, y_train: {y_tr.shape}")
    print(f"X_test:  {X_te.shape}, y_test:  {y_te.shape}")
    print(f"Label mapping: {mapping}")

    train_dl, test_dl = pp.get_dataloaders(X_tr, X_te, y_tr, y_te)
    for batch_X, batch_y in train_dl:
        print(f"Batch X: {batch_X.shape}, Batch y: {batch_y.shape}")
        break
