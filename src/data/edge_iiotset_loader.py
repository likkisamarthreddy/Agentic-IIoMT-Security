# -*- coding: utf-8 -*-
"""
Edge-IIoTset Loader / Preprocessor  (INDUSTRIAL DOMAIN)
=======================================================

Standalone loader for the **Edge-IIoTset** dataset. This module is kept
completely independent from the CICIoMT2024 (medical) pipeline — the two
domains are *never* merged. This realises the paper's "Cross-Domain"
objective as two separately trained, separately evaluated domain models.

It targets **File 3.1 — ``DNN-EdgeIIoT-dataset.csv``**, the curated
deep-learning dataset shipped with Edge-IIoTset (the official ML/DL split).
The cleaning recipe follows the dataset authors' reference preprocessing:

1. Drop known leakage / identifier columns (IPs, raw payloads, timestamps,
   ephemeral ports) so the model learns behaviour, not addresses.
2. Drop duplicate rows and NaN/Inf.
3. Label-encode the remaining categorical (object) columns.
4. ``StandardScaler`` fit on train, applied to both splits.
5. Stratified train/test split.

Returns NumPy arrays ready for the CNN-BiGRU classifier (with a unit
sequence axis so the runtime contract matches ``sequence_length: 1``).

Usage::

    from data.edge_iiotset_loader import EdgeIIoTLoader
    loader = EdgeIIoTLoader()
    data = loader.prepare()
    # data.X_train (N, 1, F), data.y_train, ... data.label_mapping
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger("edge_iiotset.loader")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "settings.yaml"


@dataclass
class EdgeData:
    """Container for prepared Edge-IIoTset tensors and metadata."""

    X_train: np.ndarray          # (N, 1, F) float32
    X_test: np.ndarray           # (M, 1, F) float32
    y_train: np.ndarray          # (N,) int64  — multiclass labels
    y_test: np.ndarray           # (M,) int64
    feature_names: List[str]
    label_mapping: Dict[str, int]
    scaler: StandardScaler
    num_features: int
    num_classes: int


class EdgeIIoTLoader:
    """Loads and preprocesses the Edge-IIoTset DNN dataset (industrial)."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        with open(config_path or _DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self.cfg: Dict[str, Any] = cfg.get("edge_iiotset", {})
        self.drop_columns: List[str] = self.cfg.get("drop_columns", [])
        self.label_column: str = self.cfg.get("label_column", "Attack_type")
        self.binary_column: str = self.cfg.get("binary_column", "Attack_label")
        self.test_ratio: float = float(self.cfg.get("test_ratio", 0.2))
        self.seed: int = int(self.cfg.get("random_seed", 42))
        self.max_rows: Optional[int] = self.cfg.get("max_rows")
        self.preferred_csv: str = self.cfg.get("preferred_csv", "DNN-EdgeIIoT-dataset.csv")
        self.dataset_path: Optional[str] = self.cfg.get("dataset_path")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def find_csv(self) -> Path:
        """Locate the curated DNN-EdgeIIoT CSV (File 3.1).

        Search order:
          1. ``edge_iiotset.dataset_path`` if it points directly to a CSV.
          2. Recursive search for ``preferred_csv`` under ``dataset_path``.
          3. Recursive search under common local folders.
        """
        # 1. Explicit path to a CSV
        if self.dataset_path:
            p = Path(self.dataset_path)
            if p.is_file() and p.suffix.lower() == ".csv":
                return p
            if p.is_dir():
                hit = self._search_dir(p)
                if hit:
                    return hit

        # 2/3. Search common roots
        search_roots = [
            _PROJECT_ROOT / "datasets",
            _PROJECT_ROOT / "datasets" / "edge_iiotset",
            _PROJECT_ROOT / "Edge-IIoTset",
            _PROJECT_ROOT / "EdgeIIoTset",
            _PROJECT_ROOT,
            # Kaggle mounted-dataset locations (read-only, no local disk used)
            Path("/kaggle/input/datasets/mohamedamineferrag/"
                 "edgeiiotset-cyber-security-dataset-of-iot-iiot/Edge-IIoTset dataset"),
            Path("/kaggle/input/edge-iiotset-cyber-security-dataset-of-iot-iiot"),
            Path("/kaggle/input/edgeiiotset-cyber-security-dataset-of-iot-iiot"),
            Path("/kaggle/input"),
        ]
        for root in search_roots:
            if root.exists():
                hit = self._search_dir(root)
                if hit:
                    return hit

        raise FileNotFoundError(
            f"Could not locate '{self.preferred_csv}'. Set 'edge_iiotset.dataset_path' "
            f"in config/settings.yaml to the extracted Edge-IIoTset folder or directly "
            f"to the {self.preferred_csv} file."
        )

    def _search_dir(self, root: Path) -> Optional[Path]:
        # Prefer the exact curated filename first.
        for cand in root.rglob(self.preferred_csv):
            logger.info("Found preferred Edge-IIoTset CSV: %s", cand)
            return cand
        # Fallback: any DNN-*EdgeIIoT*.csv
        for cand in root.rglob("*DNN*EdgeIIoT*.csv"):
            logger.info("Found Edge-IIoTset DNN CSV (fallback): %s", cand)
            return cand
        return None

    # ------------------------------------------------------------------
    # Loading & cleaning
    # ------------------------------------------------------------------
    def load_raw(self, csv_path: Optional[Path] = None) -> pd.DataFrame:
        csv_path = csv_path or self.find_csv()
        logger.info("Loading Edge-IIoTset from %s ...", csv_path)
        # Force the plain-Python string backend. pandas' optional PyArrow string
        # backend can segfault (access violation) on Windows during read_csv, so
        # we explicitly avoid it here. This is also fast for the large CSV.
        try:
            pd.set_option("mode.string_storage", "python")
        except Exception:
            pass
        try:
            pd.set_option("future.infer_string", False)
        except Exception:
            pass
        df = pd.read_csv(csv_path, low_memory=False, engine="c")
        if self.max_rows:
            df = df.sample(n=min(self.max_rows, len(df)), random_state=self.seed)
        logger.info("Raw shape: %s", df.shape)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the official Edge-IIoTset cleaning recipe."""
        # Normalise possible label-column name variants
        rename = {}
        for c in df.columns:
            lc = c.strip()
            if lc.lower() in ("attack_type",):
                rename[c] = "Attack_type"
            elif lc.lower() in ("attack_label",):
                rename[c] = "Attack_label"
        df = df.rename(columns=rename)

        # Drop leakage / identifier columns (ignore if absent)
        present_drop = [c for c in self.drop_columns if c in df.columns]
        df = df.drop(columns=present_drop, errors="ignore")
        logger.info("Dropped %d leakage columns: %s", len(present_drop), present_drop)

        # Clean NaN / Inf, drop duplicates
        df = df.replace([np.inf, -np.inf], np.nan)
        before = len(df)
        df = df.dropna()
        df = df.drop_duplicates()
        logger.info("Dropped %d rows via dropna/dedup (%d -> %d).",
                    before - len(df), before, len(df))
        df = df.reset_index(drop=True)
        return df

    def encode_and_split(self, df: pd.DataFrame) -> EdgeData:
        if self.label_column not in df.columns:
            raise KeyError(
                f"Label column '{self.label_column}' not found. Available: {list(df.columns)[:20]} ..."
            )

        # Separate labels; drop the binary column so it can't leak the answer.
        y_raw = df[self.label_column].astype(str).values
        feature_df = df.drop(columns=[self.label_column], errors="ignore")
        feature_df = feature_df.drop(columns=[self.binary_column], errors="ignore")

        # Label-encode remaining categorical (object) feature columns.
        cat_cols = [c for c in feature_df.columns if feature_df[c].dtype == object]
        for c in cat_cols:
            feature_df[c] = LabelEncoder().fit_transform(feature_df[c].astype(str))
        if cat_cols:
            logger.info("Label-encoded %d categorical feature cols: %s", len(cat_cols), cat_cols)

        feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        feature_names = list(feature_df.columns)
        X = feature_df.values.astype(np.float32)

        # Encode target labels.
        target_le = LabelEncoder()
        y = target_le.fit_transform(y_raw).astype(np.int64)
        label_mapping = {cls: int(i) for i, cls in enumerate(target_le.classes_)}
        logger.info("Classes (%d): %s", len(label_mapping), label_mapping)

        # Stratified split.
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.test_ratio, random_state=self.seed, stratify=y
        )

        # Scale (fit on train only).
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_te = scaler.transform(X_te).astype(np.float32)

        # Add unit sequence axis -> (N, 1, F) to match sequence_length: 1.
        X_tr = np.expand_dims(X_tr, axis=1)
        X_te = np.expand_dims(X_te, axis=1)

        return EdgeData(
            X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
            feature_names=feature_names, label_mapping=label_mapping,
            scaler=scaler, num_features=X_tr.shape[2], num_classes=len(label_mapping),
        )

    def prepare(self, csv_path: Optional[Path] = None) -> EdgeData:
        """Full pipeline: locate -> load -> clean -> encode/split/scale."""
        df = self.load_raw(csv_path)
        df = self.clean(df)
        data = self.encode_and_split(df)
        logger.info(
            "Edge-IIoTset prepared: X_train=%s X_test=%s features=%d classes=%d",
            data.X_train.shape, data.X_test.shape, data.num_features, data.num_classes,
        )
        return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    loader = EdgeIIoTLoader()
    d = loader.prepare()
    print("OK:", d.X_train.shape, d.num_classes, "classes")
