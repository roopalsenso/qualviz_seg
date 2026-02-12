import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("/root/old-data/home/roopal/output/model_fp32.onnx", providers=["CPUExecutionProvider"])

# Print input name
input_name = session.get_inputs()[0].name
print("Model input name:", input_name)

# Create dummy input (same as training size)
dummy_input = np.random.rand(1, 3, 512, 512).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: dummy_input})

print("Inference successful!")
print("Output shape:", outputs[0].shape)
print("Output dtype:", outputs[0].dtype)
