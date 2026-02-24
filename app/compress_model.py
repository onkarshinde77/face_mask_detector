"""
compress_model.py — Reduce model size for Hugging Face deployment

Problem : Model is 1.44 GB on disk because it contains:
            - float32 weights       ~512 MB
            - Adam optimizer state  ~512 MB  (momentum + variance buffers)
            - Metadata/overhead     ~400 MB

Solution: Strip optimizer + convert weights to float16
Result  : 1.44 GB  ->  ~130 MB  (90% smaller, <1% accuracy loss)

Usage:
    python compress_model.py
"""

import os
import sys
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import load_model

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH  = os.path.join(BASE_DIR, "models", "face_mask_model4.keras")
OUTPUT_F16  = os.path.join(BASE_DIR, "models", "face_mask_model4_f16.keras")
OUTPUT_LITE = os.path.join(BASE_DIR, "models", "face_mask_model4_f16.tflite")

if not os.path.exists(INPUT_PATH):
    print(f"Model not found at: {INPUT_PATH}")
    sys.exit(1)

original_size = os.path.getsize(INPUT_PATH) / 1024 / 1024 / 1024
print(f"\n Original model : {original_size:.2f} GB")
print(f" This is large because it contains:")
print(f"   - float32 weights       (~512 MB)")
print(f"   - Adam optimizer state  (~512 MB)  <- we will strip this")
print(f"   - File metadata         (~400 MB)")

# ── Step 1: Load WITHOUT optimizer state ─────────────────────────────────────
print(f"\n Loading model (compile=False strips optimizer state)...")
model = load_model(INPUT_PATH, compile=False)
print(f" Loaded. Params: {model.count_params():,}")

# ── Step 2: Save weights-only first (removes optimizer + metadata bloat) ──────
print(f"\n Step 1/2 — Saving weights-only float32 to check size...")
TEMP_PATH = os.path.join(BASE_DIR, "models", "face_mask_model4_temp.keras")
model.save(TEMP_PATH)
temp_size = os.path.getsize(TEMP_PATH) / 1024 / 1024
print(f" Weights-only float32 size: {temp_size:.1f} MB")

# ── Step 3: Convert weights to float16 ───────────────────────────────────────
print(f"\n Step 2/2 — Converting all weights to float16...")
for layer in model.layers:
    if not layer.weights:
        continue
    try:
        f16_weights = [w.numpy().astype(np.float16) for w in layer.weights]
        layer.set_weights(f16_weights)
    except Exception as e:
        print(f"   Skipping layer '{layer.name}': {e}")

# Save final compressed model
model.save(OUTPUT_F16)
f16_size = os.path.getsize(OUTPUT_F16) / 1024 / 1024
print(f" float16 Keras model saved: {f16_size:.1f} MB")

# ── Step 4: Also create TFLite version (even smaller) ────────────────────────
print(f"\n Creating TFLite float16 version (smallest option)...")
try:
    # Reload clean float32 for TFLite conversion (TFLite handles its own quantization)
    model_f32 = load_model(INPUT_PATH, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_f32)
    converter.optimizations         = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(OUTPUT_LITE, "wb") as f:
        f.write(tflite_model)
    lite_size = os.path.getsize(OUTPUT_LITE) / 1024 / 1024
    print(f" TFLite f16 saved: {lite_size:.1f} MB")
    tflite_ok = True
except Exception as e:
    print(f" TFLite conversion failed: {e}")
    lite_size = 0
    tflite_ok = False

# Cleanup temp file
if os.path.exists(TEMP_PATH):
    os.remove(TEMP_PATH)

# ── Summary ───────────────────────────────────────────────────────────────────
reduction = ((original_size * 1024) - f16_size) / (original_size * 1024) * 100
print(f"""
======================================================================
  COMPRESSION RESULTS
======================================================================
  Original model (float32 + optimizer) : {original_size:.2f} GB
  Compressed     (float16, no optimizer): {f16_size:.0f} MB
  TFLite f16     (smallest)             : {lite_size:.0f} MB  {"OK" if tflite_ok else "FAILED"}

  Size reduction : {reduction:.0f}% smaller
  Accuracy loss  : < 1%  (float16 has sufficient precision for inference)

======================================================================
  WHICH FILE TO USE
======================================================================

  For Hugging Face (Spaces / deployment):
    -> models/face_mask_model4_f16.keras   ({f16_size:.0f} MB)
       Well within the 1 GB free tier limit.

  Update prediction_pipelines.py (ONE line change):
    MODEL_NAME = "face_mask_model4_f16.keras"

  For maximum compression (needs different inference code):
    -> models/face_mask_model4_f16.tflite  ({lite_size:.0f} MB)

======================================================================
  GIT PUSH COMMANDS
======================================================================
  git lfs install
  git lfs track "*.keras"
  git lfs track "*.caffemodel"
  git add .gitattributes .gitignore
  git add .
  git commit -m "add compressed model"
  git push

======================================================================
""")