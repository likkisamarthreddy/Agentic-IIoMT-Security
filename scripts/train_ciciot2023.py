import os
import sys
from pathlib import Path
import logging

import torch
import torch.nn as nn
import numpy as np

# Setup paths
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_PROJECT_ROOT))

from src.data.ciciot2023_loader import CICIot2023Loader
from src.system1.models.cnn_bigru import CNNBiGRU
from src.evaluation.metrics_collector import compute_ecr, compute_fer, compute_gci, compute_ri2, compute_cas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_ciciot2023")

def main():
    logger.info("Loading CICIoT2023 Dataset with Temporal Splitting...")
    # Restrict rows for testing speed (100k per file)
    loader = CICIot2023Loader(max_rows_per_file=100000)
    data = loader.prepare()
    
    input_shape = (1, data.num_features)  # (1, 39)
    num_classes = data.num_classes
    
    logger.info(f"Building Model: Input Shape: {input_shape}, Classes: {num_classes}")
    
    model = CNNBiGRU(
        num_features=input_shape[1],
        num_classes=num_classes
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert data to tensors
    X_train_t = torch.tensor(data.X_train, dtype=torch.float32)
    y_train_t = torch.tensor(data.y_train, dtype=torch.long)
    X_test_t = torch.tensor(data.X_test, dtype=torch.float32)
    y_test_t = torch.tensor(data.y_test, dtype=torch.long)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    test_dataset = torch.utils.data.TensorDataset(X_test_t, y_test_t)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 3
    logger.info(f"Starting Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
        
    logger.info("Evaluating on Test Set...")
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    accuracy = np.mean(all_preds == all_targets)
    logger.info(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Identify the benign class. In CICIoT2023, there's "benign.csv", so class might be "Benign"
    benign_cls = data.label_mapping.get("Benign", 0)
    
    # Simulate Agentic Decision Loop metrics
    policy_ok_count = 0
    total_constrained = len(all_preds)
    total_escalations = 0
    false_escalation_count = 0
    
    # Let's split into 5 temporal segments to calculate RI2
    segments = np.array_split(np.arange(total_constrained), 5)
    ecr_segments = []
    
    for seg_indices in segments:
        seg_policy_ok = 0
        for i in seg_indices:
            pred = all_preds[i]
            true_label = all_targets[i]
            conf = np.max(all_probs[i])
            
            # Rule 1: Confidence < 0.85 requires human review (violates autonomous policy)
            if conf >= 0.85:
                seg_policy_ok += 1
                policy_ok_count += 1
            
            # Rule 2: False Escalation (predicted malicious, but was benign)
            if pred != benign_cls:
                total_escalations += 1
                if true_label == benign_cls:
                    false_escalation_count += 1
                    
        ecr_segments.append(compute_ecr(seg_policy_ok, len(seg_indices)))

    ecr = compute_ecr(policy_ok_count, total_constrained)
    fer = compute_fer(false_escalation_count, total_escalations)
    ri2 = compute_ri2(ecr_segments)
    
    # GCI -> weighted average of ECR and (1-FER)
    gci = compute_gci([ecr, 1.0 - fer], [0.6, 0.4])
    
    # CAS -> ECR * (1 - FER) (no previous ECR so delta_t=0)
    cas = compute_cas(ecr, fer, previous_ecr=None, delta_t=0)
    
    print("\n" + "="*50)
    print("Agentic Governance Metrics (CICIoT2023)")
    print("="*50)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Ethical Compliance Rate (ECR): {ecr:.4f}")
    print(f"False Escalation Rate (FER): {fer:.4f}")
    print(f"Governance Compliance Index (GCI): {gci:.4f}")
    print(f"Resilience Index (RI2): {ri2:.4f}")
    print(f"Cyber-Adaptive Score (CAS): {cas:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
