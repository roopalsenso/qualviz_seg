import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import subprocess

# ------------------ CONFIG ------------------
trt_engines = [
    "/root/old-data/home/roopal/engines/model_fp32.trt",
    "/root/old-data/home/roopal/engines/model_fp16.trt",
    "/root/old-data/home/roopal/engines/model_int8.trt"
]

# ONNX tensor names (from your FP32 model)
input_name = "x"
output_name = "save_infer_model/scale_0.tmp_0"

# Hardcoded shapes (from your ONNX model)
input_shape = (1, 3, 512, 512)
output_shape = (1, 512, 512)

image_path = "/root/old-data/home/roopal/datasets/test_offline_dataset/images/24.jpg"

# ------------------ LOAD AND PREPROCESS IMAGE ------------------
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (input_shape[3], input_shape[2]))
img_normalized = img_resized.astype(np.float32) / 255.0
img_transposed = np.transpose(img_normalized, (2, 0, 1))
input_tensor = np.ascontiguousarray(np.expand_dims(img_transposed, axis=0))

# ------------------ GPU MEMORY UTILITY ------------------
def get_gpu_memory():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        return float(result.decode().strip())
    except:
        return 0

# ------------------ LOOP OVER TRT ENGINES ------------------
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

for engine_path in trt_engines:
    quant_type = "INT8" if "int8" in engine_path.lower() else "FP16" if "fp16" in engine_path.lower() else "FP32"

    # Load engine and create context
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    # Allocate GPU buffers
    d_input = cuda.mem_alloc(input_tensor.nbytes)
    d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.float32().nbytes))
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    stream = cuda.Stream()

    # Copy input and run inference
    cuda.memcpy_htod_async(d_input, input_tensor, stream)

    start_time = time.time()
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000
    gpu_mem = get_gpu_memory()

    # Print results
    print(f"Model Path       : {engine_path}")
    print(f"Quant Type       : {quant_type}")
    print(f"Input Shape      : {input_shape}")
    print(f"Device           : GPU")
    print(f"Inference Time   : {elapsed_ms:.2f} ms")
    print(f"GPU Memory Used  : {gpu_mem:.2f} MB\n")
