# ==========================================
# Kaggle Training Script for IIoMT CNN-BiGRU
# ==========================================
# 1. Upload this script to a Kaggle Notebook.
# 2. Attach the CICIoMT2024 dataset.
# 3. Change DATASET_PATH below to point to the CSV.
# 4. Run the cell!

import os
import time
import copy
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("kaggle_trainer")

# --- Default Configurations (Hardcoded to remove YAML dependency) ---
CONFIG = {
    "data": {
        "window_size": 20,
        "test_ratio": 0.2,
        "random_seed": 42
    },
    "system1": {
        "training": {
            "batch_size": 2048,
            "epochs": 100,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "patience": 10,
            "scheduler_factor": 0.5,
            "scheduler_patience": 5
        },
        "model": {
            "conv1_filters": 64,
            "conv2_filters": 128,
            "kernel_size": 3,
            "pool_size": 2,
            "gru_hidden_size": 64,
            "gru_num_layers": 2,
            "fc1_size": 64,
            "dropout_rate": 0.3
        }
    }
}

_DEFAULT_CONFIG = None

def _load_config(config_path=None):
    return CONFIG

# =====================================================================

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
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        # Directory loading logic: glob and recursively load shards
        if filepath.is_dir():
            logger.info("Path %s is a directory. Loading and concatenating all CSV shards inside...", filepath.name)
            csv_files = sorted(filepath.glob("**/*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found inside directory: {filepath}")
            
            chunks = []
            total_rows = 0
            for csv_file in csv_files:
                shard_max = (max_rows - total_rows) if max_rows else None
                if max_rows and total_rows >= max_rows:
                    break
                
                try:
                    shard_df = DataPreprocessor.load_csv(csv_file, max_rows=shard_max)
                    if not shard_df.empty:
                        chunks.append(shard_df)
                        total_rows += len(shard_df)
                except Exception as e:
                    logger.error(f"Error loading shard {csv_file.name}: {e}")
            
            if not chunks:
                return pd.DataFrame()
            
            df = pd.concat(chunks, ignore_index=True)
            logger.info("Successfully concatenated %d shards into shape %s", len(chunks), df.shape)
            return df

        # Single-file chunked read logic
        logger.info("Parsing CSV %s (max_rows=%s) via chunked read...", filepath.name, max_rows)

        label_candidates = {"label", "Attack_type", "Label", "attack_type", "class", "Class", "category", "Category", "attack", "Attack", "traffic_type", "Traffic Type"}
        CHUNK_SIZE = 50_000
        chunks = []
        rows_read = 0
        keep_cols = None
        numeric_cols = None

        for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE, low_memory=False):
            # Identify columns to keep from first chunk only
            if keep_cols is None:
                # Strip spaces and cast to lowercase/compare for robust mapping
                label_col = next((c for c in chunk.columns if c.strip() in label_candidates or c.strip().lower() in label_candidates), None)
                if label_col is None:
                    logger.warning("No label column found in candidates! Available columns: %s", list(chunk.columns))
                numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
                keep_cols = numeric_cols + ([label_col] if label_col and label_col not in numeric_cols else [])
                dropped = [c for c in chunk.columns if c not in keep_cols]
                if dropped:
                    logger.info("Dropping %d non-numeric columns: %s...", len(dropped), dropped[:5])

            # Drop string columns immediately before accumulating
            chunk = chunk[keep_cols].copy()

            # Safe cast to float32 after column filter
            for col in numeric_cols:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype(np.float32)

            chunks.append(chunk)
            rows_read += len(chunk)

            if max_rows and rows_read >= max_rows:
                break

        df = pd.concat(chunks, ignore_index=True)
        if max_rows:
            df = df.iloc[:max_rows]

        logger.info(
            "Loaded %s — final shape %s, memory: %.1f MB",
            filepath.name, df.shape,
            df.memory_usage(deep=True).sum() / 1e6,
        )
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
            raise KeyError(f"Label column '{label_col}' not found in DataFrame. Available columns: {list(df.columns)}")

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
class CNNBiGRU(nn.Module):
    """Hybrid CNN–BiGRU network for multi-class traffic classification.

    The model first applies two 1-D convolutional blocks to extract
    local spatial features, then feeds the resulting sequence into a
    bidirectional GRU to capture temporal dependencies.  A two-layer
    fully-connected classifier head produces the final logits.

    Args:
        num_features: Number of input features per sample.
        num_classes: Number of output classes.
        config_path: Path to ``settings.yaml``.  When *None*, the
            default project-root config is used.

    Example::

        model = CNNBiGRU(num_features=46, num_classes=6)
        x = torch.randn(32, 1, 46)
        logits = model(x)          # (32, 6)
        scores = model.get_anomaly_score(x)  # (32,)
    """

    def __init__(
        self,
        num_features: int = 46,
        num_classes: int = 6,
        config_path: Optional[Path] = None,
    ) -> None:
        super().__init__()

        cfg = _load_config(config_path or _DEFAULT_CONFIG)
        s1 = cfg["system1"]["model"]

        self.num_features = num_features
        self.num_classes = num_classes

        # Quantization Stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # ---- Hyper-parameters from config --------------------------------
        conv_filters: List[int] = s1.get("conv_filters", [64, 128])
        kernel_size: int = s1.get("conv_kernel_size", 3)
        gru_hidden: int = s1.get("gru_hidden_size", 64)
        gru_layers: int = s1.get("gru_num_layers", 2)
        gru_dropout: float = s1.get("gru_dropout", 0.3)
        cnn_dropout: float = s1.get("cnn_dropout", 0.25)
        fc_hidden: int = s1.get("fc_hidden", 64)
        fc_dropout: float = s1.get("fc_dropout", 0.3)

        # ---- CNN Feature Extractor ---------------------------------------
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=conv_filters[0],
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(conv_filters[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(cnn_dropout),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=conv_filters[0],
                out_channels=conv_filters[1],
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(conv_filters[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(cnn_dropout),
        )

        # ---- BiGRU Temporal Encoder --------------------------------------
        self.gru = nn.GRU(
            input_size=conv_filters[1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )

        # BiGRU output dimension = 2 * gru_hidden (fwd + bwd)
        gru_output_size = gru_hidden * 2
        
        # ---- Attention Layer ---------------------------------------------
        self.attention = nn.Linear(gru_output_size, 1)

        # ---- Classifier Head ---------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, num_classes),
        )

        logger.info(
            "CNNBiGRU initialised — features=%d, classes=%d, "
            "conv_filters=%s, gru_hidden=%d, params=%s",
            num_features,
            num_classes,
            conv_filters,
            gru_hidden,
            f"{self.count_parameters():,}",
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Args:
            x: Input tensor of shape ``(batch, 1, num_features)``.

        Returns:
            Logits tensor of shape ``(batch, num_classes)``.
        """
        x = self.quant(x)

        # Input: (B, Seq, Features)
        # Transpose for CNN: (B, Features, Seq)
        out = x.permute(0, 2, 1)
        out = self.conv_block1(out)
        out = self.conv_block2(out)

        # Transpose for GRU: (B, Channels, Seq) → (B, Seq, Channels)
        out = out.permute(0, 2, 1)

        # BiGRU: (B, Seq, Channels) → (B, Seq, 2*H)
        gru_out, _ = self.gru(out)

        # Attention scoring
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = (attn_weights * gru_out).sum(dim=1)  # (B, 2*H)

        # Classifier
        logits = self.classifier(context)  # (B, num_classes)
        
        logits = self.dequant(logits)
        return logits

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute an anomaly score as ``1 − max(softmax(logits))``.

        Higher values indicate more anomalous inputs (lower classifier
        confidence).

        Args:
            x: Input tensor of shape ``(batch, 1, num_features)``.

        Returns:
            Tensor of shape ``(batch,)`` with anomaly scores in [0, 1].
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            max_probs, _ = torch.max(probabilities, dim=1)
            anomaly_scores = 1.0 - max_probs
        return anomaly_scores

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_summary(self) -> Dict[str, Any]:
        """Return a summary dict with parameter count and estimated size.

        The estimated size is computed as
        ``total_params × 4 bytes`` (FP32) converted to megabytes.

        Returns:
            Dictionary with keys ``total_params``, ``trainable_params``,
            ``non_trainable_params``, ``estimated_size_mb``,
            and ``layer_details``.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable

        # Estimated FP32 model size in MB
        estimated_mb = (total * 4) / (1024 ** 2)

        layer_details: List[Dict[str, Any]] = []
        for name, param in self.named_parameters():
            layer_details.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "params": param.numel(),
                    "requires_grad": param.requires_grad,
                }
            )

        summary = {
            "total_params": total,
            "trainable_params": trainable,
            "non_trainable_params": non_trainable,
            "estimated_size_mb": round(estimated_mb, 4),
            "layer_details": layer_details,
        }

        logger.info(
            "Model summary — total=%s, trainable=%s, size≈%.4f MB",
            f"{total:,}",
            f"{trainable:,}",
            estimated_mb,
        )
        return summary

    def get_estimated_size_mb(self) -> float:
        """Return the estimated FP32 model size in megabytes.

        Returns:
            Model size in MB (FP32, 4 bytes per parameter).
        """
        total_params = sum(p.numel() for p in self.parameters())
        size_mb = (total_params * 4) / (1024 ** 2)
        return round(size_mb, 4)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

class ModelTrainer:
    """Full training pipeline for the CNN-BiGRU model.

    Args:
        model: The PyTorch model to train (e.g. ``CNNBiGRU``).
        config: Training-specific configuration dict with keys
            ``learning_rate``, ``weight_decay``, ``patience``,
            ``scheduler_factor``, ``scheduler_patience``.
    """

    def __init__(self, model: nn.Module, config: dict) -> None:
        self.model = model
        self.config = config

        self.lr: float = float(config.get("learning_rate", 1e-3))
        self.weight_decay: float = float(config.get("weight_decay", 1e-4))
        self.patience: int = int(config.get("patience", 10))
        self.scheduler_factor: float = float(config.get("scheduler_factor", 0.5))
        self.scheduler_patience: int = int(config.get("scheduler_patience", 5))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self._best_state: Optional[dict] = None
        self._best_val_loss: float = float("inf")

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[float]]:
        """Run the full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation / test data loader.
            epochs: Number of training epochs.
            class_weights: Optional per-class weight tensor for
                imbalanced datasets.  Computed automatically from
                ``train_loader`` if *None*.

        Returns:
            Dictionary with per-epoch ``train_loss``, ``train_acc``,
            ``val_loss``, ``val_acc`` histories.
        """
        # --- Compute class weights if not provided ---
        if class_weights is None:
            class_weights = self._compute_class_weights(train_loader)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        no_improve_count = 0
        self._best_val_loss = float("inf")
        self._best_state = None

        logger.info(
            "Starting training for %d epochs (lr=%.1e, wd=%.1e, patience=%d)",
            epochs, self.lr, self.weight_decay, self.patience,
        )

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()

            # --- Train one epoch ---
            train_loss, train_acc = self._train_one_epoch(
                train_loader, criterion, optimizer
            )

            # --- Validate ---
            val_loss, val_acc = self._validate(val_loader, criterion)

            # --- Scheduler step ---
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            elapsed = time.perf_counter() - t0

            logger.info(
                "Epoch %d/%d - Loss: %.4f - Acc: %.4f - Val_Loss: %.4f - "
                "Val_Acc: %.4f - LR: %.1e - %.1fs",
                epoch, epochs, train_loss, train_acc,
                val_loss, val_acc, current_lr, elapsed,
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # --- Early stopping ---
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_state = copy.deepcopy(self.model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch, self.patience,
                    )
                    break

        # Restore best weights
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            logger.info("Restored best model (val_loss=%.4f)", self._best_val_loss)

        logger.info("Training complete.")
        return history

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, model: nn.Module, test_loader: DataLoader
    ) -> Dict[str, Any]:
        """Evaluate the model and return a full classification report.

        Returns:
            Dict with ``accuracy``, ``f1_macro``, ``confusion_matrix``,
            and per-class ``classification_report``.
        """
        model.to(self.device)
        model.eval()

        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                logits = model(X_batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(y_batch.numpy().tolist())

        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)

        accuracy = float(np.mean(all_preds_np == all_labels_np))

        # Confusion matrix
        classes = sorted(set(all_labels))
        n_classes = len(classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for true, pred in zip(all_labels_np, all_preds_np):
            cm[true][pred] += 1

        # Per-class precision, recall, F1
        report: Dict[str, Dict[str, float]] = {}
        f1_scores: List[float] = []
        for i, c in enumerate(classes):
            tp = cm[i][i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            report[str(c)] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": int(cm[i, :].sum()),
            }
            f1_scores.append(f1)

        f1_macro = float(np.mean(f1_scores))

        logger.info("Evaluating model...")
        logger.info(
            "  Accuracy: %.4f | Macro-F1: %.4f | Classes: %d",
            accuracy, f1_macro, n_classes,
        )

        return {
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1_macro, 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

    def evaluate_per_attack(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        label_mapping: Dict[str, int],
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-attack-type accuracy and FPR (Table 1 metrics).

        Args:
            model: The model to evaluate.
            test_loader: Test data loader.
            label_mapping: ``{label_name: label_index}`` mapping.

        Returns:
            ``{attack_name: {"accuracy": ..., "fpr": ...}}``
        """
        model.to(self.device)
        model.eval()

        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                logits = model(X_batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(y_batch.numpy().tolist())

        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)

        # Invert mapping: index -> name
        idx_to_name = {v: k for k, v in label_mapping.items()}

        results: Dict[str, Dict[str, float]] = {}
        unique_classes = sorted(set(all_labels))

        for cls_idx in unique_classes:
            name = idx_to_name.get(cls_idx, f"class_{cls_idx}")
            if name == "Benign":
                continue  # skip benign for attack metrics

            # True positive: correctly identified as this attack
            mask_true = all_labels_np == cls_idx
            if mask_true.sum() == 0:
                continue

            accuracy = float(
                np.mean(all_preds_np[mask_true] == cls_idx)
            )

            # FPR: fraction of non-attack samples misclassified as this attack
            mask_neg = all_labels_np != cls_idx
            if mask_neg.sum() == 0:
                fpr = 0.0
            else:
                fpr = float(
                    np.mean(all_preds_np[mask_neg] == cls_idx)
                )

            results[name] = {
                "accuracy": round(accuracy, 4),
                "fpr": round(fpr, 6),
            }

        return results

    # ------------------------------------------------------------------
    #  Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, model: nn.Module, filepath: str | Path) -> None:
        """Save the model state dict to disk.

        Args:
            model: Model whose weights to save.
            filepath: Destination file path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), filepath)
        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(
            "Saving model checkpoint to %s (%.2f MB)", filepath, size_mb
        )

    def load_checkpoint(self, model: nn.Module, filepath: str | Path) -> None:
        """Load a model checkpoint from disk.

        Args:
            model: Model to load weights into.
            filepath: Path to the saved state dict.
        """
        filepath = Path(filepath)
        state_dict = torch.load(filepath, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint from %s", filepath)

    # ------------------------------------------------------------------
    #  Full pipeline
    # ------------------------------------------------------------------

    def full_pipeline(self, data_path: str) -> Dict[str, Any]:
        raise NotImplementedError("Not used in Kaggle training.")


    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        """Run one training epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).long()

            optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def _validate(
        self, loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, float]:
        """Run validation pass. Returns (avg_loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).long()

                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)

                total_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += X_batch.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def _compute_class_weights(self, loader: DataLoader) -> Optional[torch.Tensor]:
        """Compute inverse-frequency class weights from the data loader."""
        label_counts: Counter = Counter()
        for _, y_batch in loader:
            for label in y_batch.numpy().tolist():
                label_counts[label] += 1

        if not label_counts:
            return None

        # Get num_classes from model's final linear layer if available
        # The output of CNNBiGRU is (batch, num_classes)
        # We can extract it from self.model if possible, else fallback to max label + 1
        n_classes = getattr(self.model, 'num_classes', None)
        if n_classes is None:
            # Fallback for models without explicit num_classes
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    n_classes = module.out_features
        if n_classes is None:
            n_classes = max(label_counts.keys()) + 1

        n_samples = sum(label_counts.values())
        weights = []
        for i in range(n_classes):
            count = label_counts.get(i, 0)
            if count > 0:
                weights.append(n_samples / (n_classes * count))
            else:
                weights.append(0.0)

        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        logger.info(
            "Class weights computed: %s",
            ", ".join(f"{i}:{w:.2f}" for i, w in enumerate(weights) if w > 0),
        )
        return weight_tensor


# =====================================================================
# KAGGLE EXECUTION BLOCK
# =====================================================================

# Attack family mapping — collapse 46 subtypes into ~8 families
CICIOMT_LABEL_MAP = {
    "Benign": "Benign",
    "ARP_Spoofing": "Spoofing",
    "MQTT_Malformed_Data": "MQTT",
    "MQTT-Malformed_Data": "MQTT",
    "MQTT_DDoS_Connect_Flood": "DDoS",
    "MQTT-DDoS-Connect_Flood": "DDoS",
    "MQTT_DDoS_Publish_Flood": "DDoS",
    "MQTT-DDoS-Publish_Flood": "DDoS",
    "MQTT_DoS_Connect_Flood": "DoS",
    "MQTT-DoS-Connect_Flood": "DoS",
    "MQTT_DoS_Publish_Flood": "DoS",
    "MQTT-DoS-Publish_Flood": "DoS",
    "Recon_OS_Scan": "Recon",
    "Recon-OS_Scan": "Recon",
    "Recon_Ping_Sweep": "Recon",
    "Recon-Ping_Sweep": "Recon",
    "Recon_Port_Scan": "Recon",
    "Recon-Port_Scan": "Recon",
    "Recon_VulScan": "Recon",
    "Recon-VulScan": "Recon",
}

def load_ciciomt2024(train_dir: str, test_dir: Optional[str] = None, max_rows_per_file: int = 5_000) -> pd.DataFrame:
    all_dfs = []

    dirs_to_load = [("train", train_dir)]
    if test_dir and os.path.exists(test_dir):
        dirs_to_load.append(("test", test_dir))

    for split, directory in dirs_to_load:
        csv_files = sorted(Path(directory).glob("*.csv"))
        logger.info("Loading %d CSVs from %s split (max %d rows each)...", len(csv_files), split, max_rows_per_file)

        for csv_file in csv_files:
            try:
                stem = csv_file.stem  # e.g. "ARP_Spoofing_train.pcap"
                raw_label = stem.replace(".pcap", "").replace(f"_{split}", "")

                # Collapse to coarse family; fallback to TCP_IP family detection
                if raw_label in CICIOMT_LABEL_MAP:
                    label = CICIOMT_LABEL_MAP[raw_label]
                elif "DDoS" in raw_label:
                    label = "DDoS"
                elif "DoS" in raw_label:
                    label = "DoS"
                elif "Recon" in raw_label:
                    label = "Recon"
                elif "Benign" in raw_label:
                    label = "Benign"
                else:
                    label = raw_label  # keep as-is if unrecognized

                df_shard = pd.read_csv(csv_file, nrows=max_rows_per_file, low_memory=False)
                numeric_cols = df_shard.select_dtypes(include=[np.number]).columns.tolist()
                df_shard = df_shard[numeric_cols].copy()
                for col in numeric_cols:
                    df_shard[col] = pd.to_numeric(df_shard[col], errors="coerce").astype(np.float32)
                df_shard["label"] = label

                all_dfs.append(df_shard)
                logger.info("  %s — %d rows, label='%s'", csv_file.name, len(df_shard), label)

            except Exception as e:
                logger.error("Error loading %s: %s", csv_file.name, e)

    if not all_dfs:
        raise RuntimeError("No CICIoMT2024 data loaded.")

    df = pd.concat(all_dfs, ignore_index=True)
    logger.info("CICIoMT2024 combined shape: %s | Classes: %d", df.shape, df["label"].nunique())
    logger.info("Class distribution:\n%s", df["label"].value_counts().to_string())
    return df


if __name__ == "__main__":
    CICIOMT_TRAIN = "/kaggle/input/datasets/likkisamarthreddy/ciciomt2024/csv/train"
    CICIOMT_TEST  = "/kaggle/input/datasets/likkisamarthreddy/ciciomt2024/csv/test"
    EDGE_IIOT_CSV = "/kaggle/input/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv"

    DATASETS = [
        {"name": "CICIoMT2024",  "mode": "sharded", "train_dir": CICIOMT_TRAIN, "test_dir": CICIOMT_TEST,  "max_rows": None,    "label_col": "label"},
        {"name": "Edge-IIoTset", "mode": "single",  "path": EDGE_IIOT_CSV,      "test_path": None,         "max_rows": 500_000, "label_col": "Attack_type"},
    ]

    os.makedirs("checkpoints", exist_ok=True)
    prep = DataPreprocessor(config_path=None)

    for ds in DATASETS:
        ds_name = ds["name"]
        logger.info("\n%s\nStarting Training for Dataset: %s\n%s", "="*60, ds_name, "="*60)

        try:
            # ── Load ──────────────────────────────────────────────────
            if ds["mode"] == "sharded":
                if not os.path.exists(ds["train_dir"]):
                    logger.error("Train dir not found: %s. Skipping...", ds["train_dir"])
                    continue
                df = load_ciciomt2024(ds["train_dir"], ds.get("test_dir"), max_rows_per_file=50_000)

            else:  # single CSV
                csv_path = ds["path"]
                if not os.path.exists(csv_path):
                    logger.error("CSV not found: %s. Skipping...", csv_path)
                    continue
                df = prep.load_csv(csv_path, max_rows=ds.get("max_rows"))

            label_col = ds["label_col"]
            if label_col not in df.columns:
                logger.error("Label column '%s' not in DataFrame. Columns: %s", label_col, list(df.columns))
                continue

            logger.info("Label distribution:\n%s", df[label_col].value_counts().to_string())

            # ── Preprocess ────────────────────────────────────────────
            X_train, X_test, y_train, y_test, label_mapping = prep.prepare_pipeline(
                df,
                test_ratio=CONFIG["data"]["test_ratio"],
                window_size=CONFIG["data"]["window_size"],
                label_col=label_col,
            )

            train_loader, test_loader = prep.get_dataloaders(
                X_train, X_test, y_train, y_test,
                batch_size=CONFIG["system1"]["training"]["batch_size"],
            )

            # ── Model ─────────────────────────────────────────────────
            num_classes  = len(label_mapping)
            num_features = X_train.shape[2]
            logger.info("Initializing model — features=%d, classes=%d", num_features, num_classes)

            model = CNNBiGRU(num_features=num_features, num_classes=num_classes)
            model.model_summary()

            # ── Train ─────────────────────────────────────────────────
            trainer = ModelTrainer(model, CONFIG["system1"]["training"])
            history = trainer.train(train_loader, test_loader, epochs=CONFIG["system1"]["training"]["epochs"])

            # ── Evaluate ──────────────────────────────────────────────
            eval_results = trainer.evaluate(model, test_loader)
            logger.info("Results: Accuracy=%.4f | Macro-F1=%.4f", eval_results["accuracy"], eval_results["f1_macro"])

            per_attack = trainer.evaluate_per_attack(model, test_loader, label_mapping)
            logger.info("Per-attack:\n%s", json.dumps(per_attack, indent=2))

            # ── Save ──────────────────────────────────────────────────
            prefix = ds_name.lower().replace("-", "").replace(" ", "")
            trainer.save_checkpoint(model, f"checkpoints/cnn_bigru_{prefix}_fp32.pt")

            with open(f"checkpoints/label_mapping_{prefix}.json", "w") as f:
                json.dump({k: int(v) for k, v in label_mapping.items()}, f, indent=2)

            with open(f"checkpoints/history_{prefix}.json", "w") as f:
                json.dump(history, f, indent=2)

            # ── INT8 Post-Training Static Quantization ─────────────────────
            logger.info("Running INT8 Post-Training Static Quantization...")

            model.eval()
            model.cpu()

            # Fuse Conv+BN+ReLU for better quantization
            model_fused = torch.quantization.fuse_modules(model, [
                ['conv_block1.0', 'conv_block1.1', 'conv_block1.2'],
                ['conv_block2.0', 'conv_block2.1', 'conv_block2.2'],
            ])

            model_fused.qconfig = torch.quantization.get_default_qconfig('x86')
            model_fused.gru.qconfig = None
            torch.quantization.prepare(model_fused, inplace=True)

            # Calibrate on a subset of test data (CPU)
            logger.info("Calibrating quantization observer on test set...")
            model_fused.eval()
            calib_batches = 0
            with torch.no_grad():
                for X_batch, _ in test_loader:
                    model_fused(X_batch.cpu())
                    calib_batches += 1
                    if calib_batches >= 20:  # ~40k samples for calibration
                        break

            torch.quantization.convert(model_fused, inplace=True)
            logger.info("INT8 quantization complete.")

            # Save INT8 model
            int8_path = f"checkpoints/cnn_bigru_{prefix}_int8.pt"
            torch.save(model_fused.state_dict(), int8_path)
            logger.info("Saved INT8 model to %s", int8_path)

            # Evaluate INT8
            logger.info("Evaluating INT8 model...")

            # Temporarily move test_loader to CPU for INT8 eval
            int8_all_preds, int8_all_labels = [], []
            model_fused.eval()
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    logits = model_fused(X_batch.cpu())
                    preds = torch.argmax(logits, dim=1).numpy()
                    int8_all_preds.extend(preds.tolist())
                    int8_all_labels.extend(y_batch.numpy().tolist())

            int8_preds_np = np.array(int8_all_preds)
            int8_labels_np = np.array(int8_all_labels)
            int8_accuracy = float(np.mean(int8_preds_np == int8_labels_np))

            # Per-class metrics for the table
            idx_to_name = {v: k for k, v in label_mapping.items()}
            logger.info("\n%-20s %-15s %-15s %-10s", "Attack", "FP32_Acc", "INT8_Acc", "FPR")
            logger.info("-" * 60)

            fp32_results = eval_results["classification_report"]
            classes = sorted(set(int8_all_labels))
            int8_f1_scores = []

            for cls_idx in classes:
                name = idx_to_name.get(cls_idx, f"class_{cls_idx}")
                
                # INT8 per-class accuracy
                mask_true = int8_labels_np == cls_idx
                mask_neg  = int8_labels_np != cls_idx
                int8_acc  = float(np.mean(int8_preds_np[mask_true] == cls_idx)) if mask_true.sum() > 0 else 0.0
                int8_fpr  = float(np.mean(int8_preds_np[mask_neg]  == cls_idx)) if mask_neg.sum()  > 0 else 0.0

                # FP32 per-class accuracy from earlier eval
                fp32_acc = fp32_results.get(str(cls_idx), {}).get("recall", 0.0)

                logger.info("%-20s %-15.4f %-15.4f %-10.6f", name, fp32_acc, int8_acc, int8_fpr)

            logger.info("\nOverall INT8 Accuracy: %.4f", int8_accuracy)

            # Save comparison JSON
            comparison = {
                "fp32_accuracy": eval_results["accuracy"],
                "int8_accuracy": round(int8_accuracy, 4),
                "fp32_f1_macro": eval_results["f1_macro"],
                "per_class": {}
            }
            for cls_idx in classes:
                name = idx_to_name.get(cls_idx, f"class_{cls_idx}")
                mask_true = int8_labels_np == cls_idx
                mask_neg  = int8_labels_np != cls_idx
                comparison["per_class"][name] = {
                    "fp32_recall": fp32_results.get(str(cls_idx), {}).get("recall", 0.0),
                    "int8_accuracy": round(float(np.mean(int8_preds_np[mask_true] == cls_idx)), 4) if mask_true.sum() > 0 else 0.0,
                    "fpr": round(float(np.mean(int8_preds_np[mask_neg] == cls_idx)), 6) if mask_neg.sum() > 0 else 0.0,
                }

            with open(f"checkpoints/quantization_comparison_{prefix}.json", "w") as f:
                json.dump(comparison, f, indent=2)
            logger.info("Saved quantization comparison to checkpoints/quantization_comparison_%s.json", prefix)

            logger.info("Finished %s!\n", ds_name)

        except Exception as e:
            logger.exception("Fatal error on dataset %s: %s", ds_name, e)
            continue

    logger.info("All datasets processed.")

