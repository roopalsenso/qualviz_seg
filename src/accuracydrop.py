# accuracydrop_segmentation.py
import os
import glob
import cv2
import numpy as np
import onnxruntime as ort

# ---------------- CONFIG ----------------
models = [
    "/root/old-data/home/roopal/models/fp32/model_fp32.onnx",
    "/root/old-data/home/roopal/models/fp16/model_fp16.onnx",
    "/root/old-data/home/roopal/models/int8/model_int8.onnx"
]

dataset_images = "/root/old-data/home/roopal/datasets/test_offline_dataset/images/*.jpg"
dataset_labels = "/root/old-data/home/roopal/datasets/test_offline_dataset/annotations/*.png"

device_type = "CPU"
num_classes = 2  # your dataset classes (0 and 1)

# ---------------- HELPER: IoU ----------------
def compute_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        iou = intersection / union if union != 0 else 1.0
        ious.append(iou)
    return np.mean(ious)

# ---------------- GET DATA FILES ----------------
image_files = sorted(glob.glob(dataset_images))
label_files = sorted(glob.glob(dataset_labels))
assert len(image_files) == len(label_files), "Mismatch between images and labels"

# ---------------- LOAD FP32 BASELINE ----------------
print("Running FP32 baseline...")
fp32_session = ort.InferenceSession(models[0], providers=["CPUExecutionProvider"])
input_name = fp32_session.get_inputs()[0].name
H, W = fp32_session.get_inputs()[0].shape[2:4]

baseline_outputs = []

for img_path, lbl_path in zip(image_files, label_files):
    # Read and preprocess image
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H)).astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))[None,...]  # 1xCxHxW

    # Run FP32 model
    pred = fp32_session.run(None, {input_name: img.astype(np.float32)})[0]  # [1, H, W]
    pred = pred[0]  # remove batch dim → [H, W]

    # Resize label to match prediction
    target = cv2.imread(lbl_path, 0)  # original label
    target_resized = cv2.resize(target, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

    baseline_outputs.append((pred, target_resized))

# Compute baseline average IoU
baseline_iou = np.mean([compute_iou(p, t, num_classes) for p, t in baseline_outputs])
print(f"FP32 Average IoU: {baseline_iou:.4f}\n")

# ---------------- RUN QUANTIZED MODELS ----------------
for model_path in models[1:]:
    print(f"Running {model_path} ...")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_type = session.get_inputs()[0].type

    total_iou = 0
    for img_path, lbl_path in zip(image_files, label_files):
        # Read and preprocess image
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W,H)).astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None,...]

        # Convert input type if needed
        if input_type == 'tensor(int8)':
            img_input = (img * 255).astype(np.int8)
        elif input_type == 'tensor(float16)':
            img_input = img.astype(np.float16)
        else:
            img_input = img.astype(np.float32)

        # Run model
        pred = session.run(None, {input_name: img_input})[0]  # [1, H, W]
        pred = pred[0]  # remove batch dim → [H, W]

        # Resize label to match prediction
        target = cv2.imread(lbl_path, 0)
        target_resized = cv2.resize(target, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

        total_iou += compute_iou(pred, target_resized, num_classes)

    avg_iou = total_iou / len(image_files)
    accuracy_drop = (baseline_iou - avg_iou) / baseline_iou * 100

    quant_type = "INT8" if "int8" in model_path.lower() else "FP16"
    print(f"{quant_type} Model Average IoU: {avg_iou:.4f}")
    print(f"Accuracy drop vs FP32: {accuracy_drop:.2f}%\n")
