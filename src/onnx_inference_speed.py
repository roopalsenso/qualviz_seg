# ------------------ USER CONFIG ------------------
models = [
    "/root/old-data/home/roopal/models/fp32/model_fp32.onnx",
    "/root/old-data/home/roopal/models/fp16/model_fp16.onnx",
    "/root/old-data/home/roopal/models/int8/model_int8.onnx"
]

image_path = "/root/old-data/home/roopal/datasets/test_offline_dataset/images/24.jpg"
device_type = "CPU"

# ------------------ LOAD AND PREPROCESS IMAGE ONCE ------------------
import cv2
import numpy as np

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ------------------ IMPORT ONNX AND TIME ------------------
import onnxruntime as ort
import time
import torch

for model_path in models:
    # ------------------ SETUP SESSION ------------------
    providers = ["CUDAExecutionProvider"] if device_type.upper() == "GPU" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    input_name = session.get_inputs()[0].name
    input_shape = tuple(session.get_inputs()[0].shape)
    input_shape = tuple([1 if (d is None or d < 0) else d for d in input_shape])

    # Resize image to model input
    H, W = input_shape[2], input_shape[3]
    img_resized = cv2.resize(img, (W, H))
    # Convert image to float32 first, then normalize
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Transpose HWC → CHW
    img_transposed = np.transpose(img_normalized, (2, 0, 1))

    # Add batch dimension: 1 x C x H x W
    input_tensor = np.expand_dims(img_transposed, axis=0)

    # Convert to the type expected by the model
    input_type = session.get_inputs()[0].type
    if input_type == 'tensor(float16)':
        input_tensor = input_tensor.astype(np.float16)
    elif input_type == 'tensor(float32)':
        input_tensor = input_tensor.astype(np.float32)
    elif input_type == 'tensor(int8)':
        input_tensor = input_tensor.astype(np.int8)




    # ------------------ WARM-UP ------------------
    for _ in range(5):
        session.run(None, {input_name: input_tensor})

    # ------------------ MEASURE INFERENCE ------------------
    if device_type.upper() == "GPU":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.time()
    session.run(None, {input_name: input_tensor})
    if device_type.upper() == "GPU":
        torch.cuda.synchronize()

    inference_time_ms = (time.time() - start_time) * 1000
    gpu_memory = torch.cuda.max_memory_allocated() / (1024*1024) if device_type.upper() == "GPU" else 0

    quant_type = "INT8" if "int8" in model_path.lower() else "FP16" if "fp16" in model_path.lower() else "FP32"
    print(f"Model Path       : {model_path}")
    print(f"Quant Type       : {quant_type}")
    print(f"Input Shape      : {input_shape}")
    print(f"Device           : {device_type.upper()}")
    print(f"Inference Time   : {inference_time_ms:.2f} ms")
    print(f"GPU Memory Used  : {gpu_memory:.2f} MB\n")

