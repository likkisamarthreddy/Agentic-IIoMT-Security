# --- repo bootstrap: make src/ importable + anchor CWD to repo root ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _os.path.join(_ROOT, "src") not in _sys.path:
    _sys.path.insert(0, _os.path.join(_ROOT, "src"))
_os.chdir(_ROOT)
# --- end bootstrap ---

import torch
from pathlib import Path
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
from system1.models.cnn_bigru import CNNBiGRU
import glob

def export_and_quantize():
    pt_files = glob.glob("checkpoints/*_fp32.pt")
    if not pt_files:
        print("Waiting for trained models in checkpoints/... (looking for checkpoints/*_fp32.pt)")
        return
        
    for model_path_str in pt_files:
        model_path = Path(model_path_str)
        print(f"\nProcessing {model_path.name}...")
        
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        
        # Auto-detect features and classes from the saved weights
        num_features = state_dict["conv_block1.0.weight"].shape[1]
        num_classes = state_dict["classifier.3.weight"].shape[0]
        
        print(f"Detected {num_features} features and {num_classes} classes.")
        
        model = CNNBiGRU(num_features=num_features, num_classes=num_classes)
        model.load_state_dict(state_dict)
        model.eval()

        prefix = model_path.stem.replace("_fp32", "")
        onnx_path_fp32 = Path(f"checkpoints/{prefix}_fp32.onnx")

        # Sequence length must match the training window and the runtime
        # EdgeAgent contract. Read it from config so there is a single
        # source of truth (falls back to 1 = per-flow window).
        seq_len = 1
        try:
            import yaml
            _cfg = yaml.safe_load(open("config/settings.yaml", "r", encoding="utf-8"))
            seq_len = int(
                _cfg.get("system1", {}).get("model", {}).get("sequence_length", 1)
            )
        except Exception as _e:
            print(f"Could not read sequence_length from config ({_e}); using {seq_len}.")

        dummy_input = torch.randn(1, seq_len, num_features)
        print(f"Exporting FP32 ONNX with dummy input shape {dummy_input.shape}...")
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path_fp32,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence_len'},
                'output': {0: 'batch_size'}
            }
        )

        onnx_path_int8 = Path(f"checkpoints/{prefix}_int8.onnx")
        print("Running Post-Training Quantization (INT8)...")
        quantize_dynamic(
            model_input=str(onnx_path_fp32),
            model_output=str(onnx_path_int8),
            weight_type=QuantType.QInt8
        )
        print(f"Quantized model saved to {onnx_path_int8}")

if __name__ == "__main__":
    export_and_quantize()
