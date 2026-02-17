# ------------------ USER CONFIG ------------------
models = [
   "/root/old-data/home/roopal/models/fp32/model_fp32.onnx",
   "/root/old-data/home/roopal/models/fp16/model_fp16.onnx",
   "/root/old-data/home/roopal/models/int8/model_int8.onnx"
]

image_path = "/root/old-data/home/roopal/datasets/test_offline_dataset/images/24.jpg"
device_type = "GPU"

# ------------------ LOAD AND PREPROCESS IMAGE ONCE ------------------
import cv2
import numpy as np
import subprocess
import onnxruntime as ort
import time

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ------------------ FUNCTION TO GET GPU MEMORY (MB) ------------------
def get_full_gpu_memory():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        return float(result.decode().strip())
    except:
        return 0

# ------------------ LOOP OVER MODELS ------------------
for model_path in models:
    # ------------------ SETUP SESSION ------------------
    providers = ["CUDAExecutionProvider"] if device_type.upper() == "GPU" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    # ------------------ GET GPU MEMORY AFTER SESSION CREATION ------------------
    gpu_mem_used = get_full_gpu_memory() if device_type.upper() == "GPU" else 0

    # ------------------ PREP IMAGE FOR MODEL ------------------
    input_name = session.get_inputs()[0].name
    input_shape = tuple(session.get_inputs()[0].shape)
    input_shape = tuple([1 if (d is None or d < 0) else d for d in input_shape])

    H, W = input_shape[2], input_shape[3]
    img_resized = cv2.resize(img, (W, H))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    input_tensor = np.expand_dims(img_transposed, axis=0)

    input_type = session.get_inputs()[0].type
    if input_type == 'tensor(float16)':
        input_tensor = input_tensor.astype(np.float16)
    elif input_type == 'tensor(float32)':
        input_tensor = np.float32(input_tensor)
    elif input_type == 'tensor(int8)':
        input_tensor = np.int8(input_tensor)

    # ------------------ WARM-UP ------------------
    for _ in range(5):
        session.run(None, {input_name: input_tensor})

    # ------------------ MEASURE INFERENCE ------------------
    start_time = time.time()
    session.run(None, {input_name: input_tensor})
    elapsed_ms = (time.time() - start_time) * 1000

    # ------------------ PRINT RESULTS ------------------
    quant_type = "INT8" if "int8" in model_path.lower() else "FP16" if "fp16" in model_path.lower() else "FP32"
    provider_used = session.get_providers()[0]

    print(f"Model Path       : {model_path}")
    print(f"Quant Type       : {quant_type}")
    print(f"Provider Used    : {provider_used}")
    print(f"Input Shape      : {input_shape}")
    print(f"Device           : {device_type.upper()}")
    print(f"Inference Time   : {elapsed_ms:.2f} ms")
    print(f"GPU Memory Used  : {gpu_mem_used:.2f} MB\n")
