import onnxruntime as ort
import numpy as np
import time


def load_model(model_path):
    """
    Load ONNX model using ONNX Runtime
    """
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]  # Change if using GPU
    )
    return session


def get_input_details(session):
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    input_type = input_meta.type
    return input_name, input_shape, input_type


def run_inference(session, input_tensor):
    input_name = session.get_inputs()[0].name

    start_time = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    end_time = time.time()

    inference_time = (end_time - start_time) * 1000  # ms

    return outputs, inference_time


def main():
    model_path = "/root/old-data/home/roopal/output/model_fp16.onnx"  # Change path if needed

    print("Loading model...")
    session = load_model(model_path)

    input_name, input_shape, input_type = get_input_details(session)

    print(f"Input Name  : {input_name}")
    print(f"Input Shape : {input_shape}")
    print(f"Input Type  : {input_type}")

    # Replace dynamic dims (None) with 1
    input_shape = [dim if isinstance(dim, int) else 1 for dim in input_shape]

    # Create dummy input
    input_tensor = np.random.randn(*input_shape).astype(np.float16)

    print("Running inference...")
    outputs, inference_time = run_inference(session, input_tensor)

    print(f"Inference Time: {inference_time:.2f} ms")

    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")
        print(f"Output {i} dtype: {output.dtype}")


if __name__ == "__main__":
    main()
