# -*- coding: utf-8 -*-
"""
Model Trainer
=============

Full training pipeline for the CNN-BiGRU model with:
- CrossEntropyLoss with optional class-weight balancing
- Adam optimiser with weight decay
- ReduceLROnPlateau learning-rate scheduler
- Early stopping
- Per-epoch accuracy, loss, and macro-F1 logging
- Real evaluation with confusion matrix and classification report
- Per-attack accuracy and false-positive rate (Table 1 metrics)
- Checkpoint saving via ``torch.save``
"""

from __future__ import annotations

import copy
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("iimt.system1.training.trainer")


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
        """End-to-end: load data -> preprocess -> train -> evaluate -> save.

        Args:
            data_path: Path to CSV dataset or ``"synthetic"`` to generate.

        Returns:
            Dictionary with training history and evaluation metrics.
        """
        from data.preprocessor import DataPreprocessor
        from data.synthetic_generator import SyntheticIIoMTGenerator
        import yaml

        config_path = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)

        if data_path == "synthetic":
            gen = SyntheticIIoMTGenerator(full_config)
            df = gen.generate_combined_dataset(
                full_config["data"]["synthetic"]["num_samples"]
            )
        else:
            prep = DataPreprocessor(full_config)
            df = prep.load_csv(data_path)

        prep = DataPreprocessor(full_config)
        X_train, X_test, y_train, y_test, label_mapping = prep.prepare_pipeline(
            df, test_ratio=full_config["data"]["test_ratio"], window_size=1
        )
        train_loader, test_loader = prep.get_dataloaders(
            X_train, X_test, y_train, y_test,
            batch_size=full_config["system1"]["training"]["batch_size"],
        )

        epochs = full_config["system1"]["training"]["epochs"]
        history = self.train(train_loader, test_loader, epochs)
        metrics = self.evaluate(self.model, test_loader)
        per_attack = self.evaluate_per_attack(self.model, test_loader, label_mapping)

        logger.info("Pipeline complete. Accuracy: %.4f", metrics["accuracy"])
        return {"history": history, "metrics": metrics, "per_attack": per_attack}

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
