import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
from net import TinyUNet

def export_to_onnx():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pth_path = os.path.join(current_dir, "model.pth")
    onnx_path = os.path.join(current_dir, "model.onnx")
    
    if not os.path.exists(pth_path):
        print(f"Weights file not found at {pth_path}. Run train.py first to train a model.")
        return False
        
    print(f"Loading weights from {pth_path}...")
    model = TinyUNet()
    model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Create dummy input of appropriate shape (1, 3, 180, 320)
    dummy_input = torch.randn(1, 3, 180, 320)
    
    print(f"Exporting model to ONNX at {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None # Static shapes are preferred for optimal TensorRT compilation on Jetson Nano
    )
    
    # Verify model integrity
    print("Verifying ONNX model graph structure...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX structure check: OK")
    
    # Test inference with ONNX Runtime
    print("Testing ONNX Runtime inference execution...")
    ort_session = ort.InferenceSession(onnx_path)
    
    # Input data
    np_input = np.random.randn(1, 3, 180, 320).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: np_input}
    
    ort_outs = ort_session.run(None, ort_inputs)
    print("ONNX Runtime execution check: OK")
    print(f"Output shape from ONNX Runtime: {ort_outs[0].shape}")
    
    print("Model successfully exported and verified.")
    return True

if __name__ == "__main__":
    export_to_onnx()
