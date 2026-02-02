#!/usr/bin/env bash
set -euo pipefail

# A simple test model (input -> ReLU -> output) for running examples.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/../models"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/test_model.xml" ]; then
    echo "==> Model already exists at $MODEL_DIR/test_model.xml"
    exit 0
fi

echo "==> Generating test model..."

python3 - <<'PYEOF'
import sys, os
try:
    import openvino as ov
    import numpy as np
except ImportError:
    print("Error: openvino python package not found. Run: pip install openvino", file=sys.stderr)
    sys.exit(1)

from openvino.runtime import opset13 as opset

param = opset.parameter([1, 3, 224, 224], np.float32, name="input")
relu = opset.relu(param, name="relu")
result = opset.result(relu, name="output")
model = ov.Model([result], [param], "test_model")

out = os.path.join(os.environ.get("MODEL_DIR", "models"), "test_model.xml")
ov.save_model(model, out)
print(f"==> Model saved to {out}")
PYEOF
