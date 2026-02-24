"""
Model Inspector — shows every layer, activation function,
output shape, and parameter count from a saved .keras model.

Usage:
    python inspect_model.py
"""

import os
import sys

# ── 1. Locate the model ───────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_mask_model4.keras")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at: {MODEL_PATH}")
    print("   Edit MODEL_PATH in this script to point to your .keras file.")
    sys.exit(1)

# ── 2. Load model ─────────────────────────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # CPU only
import tensorflow as tf
from tensorflow.keras.models import load_model

print(f"\n📦 Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH,compile=False)
print("✅ Model loaded successfully\n")

# ── 3. Basic info ─────────────────────────────────────────────────────────────
print("=" * 70)
print("  MODEL SUMMARY")
print("=" * 70)
model.summary()

# ── 4. Activation functions per layer ────────────────────────────────────────
print("\n" + "=" * 70)
print("  ACTIVATION FUNCTIONS PER LAYER")
print("=" * 70)
print(f"{'#':<5} {'Layer Name':<35} {'Layer Type':<25} {'Activation'}")
print("-" * 70)

for i, layer in enumerate(model.layers):
    layer_type = layer.__class__.__name__

    # Get activation — different layers store it differently
    activation = "—"

    # Conv2D, Dense — have .activation attribute
    if hasattr(layer, 'activation') and layer.activation is not None:
        activation = layer.activation.__name__

    # Standalone Activation layer (e.g. Activation('relu'))
    elif layer_type == 'Activation':
        cfg = layer.get_config()
        activation = cfg.get('activation', '—')
        if isinstance(activation, dict):
            activation = activation.get('class_name', '—')

    # BatchNormalization uses no activation itself
    elif layer_type == 'BatchNormalization':
        activation = "(batch norm)"

    # Dropout, Flatten, Pooling layers
    elif layer_type in ('Dropout', 'Flatten', 'MaxPooling2D',
                        'AveragePooling2D', 'GlobalAveragePooling2D',
                        'GlobalMaxPooling2D'):
        activation = f"({layer_type.lower()})"

    print(f"{i:<5} {layer.name:<35} {layer_type:<25} {activation}")

# ── 5. Output layer detail ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  OUTPUT LAYER DETAIL")
print("=" * 70)
output_layer = model.layers[-1]
out_cfg      = output_layer.get_config()
out_shape    = model.output_shape
out_units    = out_cfg.get('units', 'N/A')
out_act_raw  = out_cfg.get('activation', 'N/A')
out_act      = out_act_raw if isinstance(out_act_raw, str) else out_act_raw.get('class_name', str(out_act_raw))

print(f"  Layer name   : {output_layer.name}")
print(f"  Layer type   : {output_layer.__class__.__name__}")
print(f"  Output shape : {out_shape}")
print(f"  Units        : {out_units}")
print(f"  Activation   : {out_act}")

# ── 6. Interpret output layer ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  WHAT THIS MEANS FOR YOUR PREDICTION CODE")
print("=" * 70)

if isinstance(out_units, int):
    if out_units == 1:
        print("  ✅ Single neuron output  →  SIGMOID")
        print("     score > 0.5  → No Mask")
        print("     score ≤ 0.5  → Mask")
        print("     confidence   = score if No Mask else 1 - score")
    elif out_units == 2:
        print("  ✅ Two neuron output  →  SOFTMAX")
        print("     pred[0]  → Mask probability")
        print("     pred[1]  → No Mask probability")
        print("     label    = argmax([pred[0], pred[1]])")
    else:
        print(f"  ⚠️  {out_units} output neurons — check class mapping manually")
else:
    print("  ⚠️  Could not determine unit count — inspect manually")

# ── 7. Total parameters ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PARAMETER COUNT")
print("=" * 70)
total       = model.count_params()
trainable   = sum(tf.size(w).numpy() for w in model.trainable_weights)
untrainable = total - trainable
print(f"  Total params       : {total:,}")
print(f"  Trainable params   : {trainable:,}")
print(f"  Frozen params      : {untrainable:,}")
print(f"  Model size (est.)  : {total * 4 / 1024 / 1024:.1f} MB  (float32)")

# ── 8. All unique activations used ───────────────────────────────────────────
print("\n" + "=" * 70)
print("  ALL UNIQUE ACTIVATION FUNCTIONS USED IN MODEL")
print("=" * 70)
activations_found = set()
for layer in model.layers:
    if hasattr(layer, 'activation') and layer.activation is not None:
        activations_found.add(layer.activation.__name__)
    elif layer.__class__.__name__ == 'Activation':
        cfg = layer.get_config()
        act = cfg.get('activation', None)
        if act:
            activations_found.add(act if isinstance(act, str) else act.get('class_name', '?'))

if activations_found:
    for act in sorted(activations_found):
        descriptions = {
            'relu'    : 'ReLU      — max(0, x)  — hidden layers, avoids vanishing gradient',
            'sigmoid' : 'Sigmoid   — 1/(1+e^-x) — binary output, squashes to 0–1',
            'softmax' : 'Softmax   — exp(x)/sum  — multi-class output, probs sum to 1',
            'tanh'    : 'Tanh      — (e^x-e^-x)/(e^x+e^-x) — zero-centered, -1 to 1',
            'linear'  : 'Linear    — f(x) = x   — regression output, no squashing',
            'elu'     : 'ELU       — smooth ReLU variant, handles negative values',
            'selu'    : 'SELU      — self-normalizing variant of ELU',
            'swish'   : 'Swish     — x * sigmoid(x) — smooth, non-monotonic',
        }
        print(f"  • {descriptions.get(act, act)}")
else:
    print("  No standard activations detected (may be in nested layers)")

print("\n" + "=" * 70 + "\n")