import os
import numpy as np
import torch
import joblib
from sklearn.neighbors import KernelDensity

# We mock the structure of EdgeCNN_BiGRU here to load the quantized/FP32 model
# In a real environment, you'd import it from binary_classifier
from binary_classifier import EdgeCNN_BiGRU

def get_anomaly_score(model, packet_sequence):
    """
    Returns anomaly score = 1 - Confidence(highest prob)
    """
    model.eval()
    with torch.no_grad():
        out = model(packet_sequence)
        probs = torch.softmax(out, dim=1)
        return 1.0 - torch.max(probs).item()

def train_kde_threshold(benign_loader, model_path="checkpoints/binary_Man_in_the_Middle_fp32.pt"):
    print(f"Loading EdgeCNN_BiGRU from {model_path}...")
    # NOTE: num_features must match your data. Assuming 48 for CICIoMT or 41 for EdgeIIoT.
    # We will instantiate with 48 as default. This may need adjustment based on specific dataset.
    model = EdgeCNN_BiGRU(num_features=48, num_classes=2)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        print(f"Warning: {model_path} not found. Using untrained model for KDE mock.")
    
    benign_scores = []
    print("Evaluating benign traffic for KDE...")
    
    # We mock benign_loader here. In practice, this would be a DataLoader of only benign windows.
    for X_batch, y_batch in benign_loader:
        # Filter for benign
        benign_mask = (y_batch == 0)
        X_benign = X_batch[benign_mask]
        
        for i in range(len(X_benign)):
            # Pass individual sequences (unsqueeze batch dim)
            seq = X_benign[i].unsqueeze(0)
            score = get_anomaly_score(model, seq)
            benign_scores.append(score)
            
    benign_scores = np.array(benign_scores).reshape(-1, 1)
    
    print("Fitting Kernel Density Estimator...")
    kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
    kde.fit(benign_scores)
    
    # Set dynamic threshold: 99.5th percentile
    threshold = np.percentile(benign_scores, 99.5)
    print(f"Computed Emergency Brake Threshold: {threshold:.5f}")
    
    # Save the KDE and threshold
    joblib.dump({"kde": kde, "threshold": threshold}, "emergency_brake_kde.pkl")
    print("Saved KDE model to emergency_brake_kde.pkl")
    
    return threshold

if __name__ == "__main__":
    # Mock data loader for demonstration
    # In real usage, you'd pass a DataLoader containing only normal/benign traffic
    print("Starting KDE Emergency Brake Training Pipeline...")
    
    mock_X = torch.randn(100, 5, 48) # 100 sequences, 5 packets, 48 features
    mock_y = torch.zeros(100)        # All benign
    
    class MockLoader:
        def __iter__(self):
            yield mock_X, mock_y
            
    train_kde_threshold(MockLoader(), "checkpoints/binary_Man_in_the_Middle_fp32.pt")
