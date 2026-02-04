#!/bin/bash
#
# Benchmark Setup Script
# Sets up the environment for running OpenVINO vs ONNX Runtime benchmarks.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/../models"
DEFAULT_MODEL="all-MiniLM-L6-v2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "========================================"
echo "  OpenVINO vs ONNX Runtime Benchmarks"
echo "  Setup Script"
echo "========================================"
echo

# -----------------------------------------------------------------------------
# Check OpenVINO
# -----------------------------------------------------------------------------
info "Checking OpenVINO installation..."

if python3 -c "import openvino; print(f'OpenVINO version: {openvino.__version__}')" 2>/dev/null; then
    info "OpenVINO Python bindings found"
else
    warn "OpenVINO Python bindings not found (optional, used for version detection)"
fi

# Check for OpenVINO C library
if [ -n "$INTEL_OPENVINO_DIR" ]; then
    info "INTEL_OPENVINO_DIR is set: $INTEL_OPENVINO_DIR"
elif ldconfig -p 2>/dev/null | grep -q libopenvino; then
    info "OpenVINO C library found in system path"
elif [ -f "/opt/intel/openvino/runtime/lib/intel64/libopenvino.so" ]; then
    info "OpenVINO found at /opt/intel/openvino"
    echo
    warn "You may need to source the OpenVINO environment:"
    echo "  source /opt/intel/openvino/setupvars.sh"
else
    error "OpenVINO C library not found!"
    echo "  Please install OpenVINO and ensure libopenvino.so is in your library path."
    echo "  See: https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html"
    exit 1
fi

# -----------------------------------------------------------------------------
# Check ONNX Runtime
# -----------------------------------------------------------------------------
info "Checking ONNX Runtime installation..."

ORT_LIB=""
ORT_PATHS=(
    "/usr/lib/libonnxruntime.so"
    "/usr/local/lib/libonnxruntime.so"
    "/usr/local/lib/onnxruntime/libonnxruntime.so"
    "$ONNXRUNTIME_LIB_PATH"
)

# Also check ldconfig
LDCONFIG_ORT=$(ldconfig -p 2>/dev/null | grep "libonnxruntime.so" | awk '{print $NF}' | head -1)
if [ -n "$LDCONFIG_ORT" ]; then
    ORT_PATHS+=("$LDCONFIG_ORT")
fi

for path in "${ORT_PATHS[@]}"; do
    if [ -n "$path" ] && [ -f "$path" ]; then
        ORT_LIB="$path"
        break
    fi
    # Also check for versioned .so files
    for versioned in "${path}"* ; do
        if [ -f "$versioned" ]; then
            ORT_LIB="$versioned"
            break 2
        fi
    done
done

if [ -n "$ORT_LIB" ]; then
    info "ONNX Runtime library found: $ORT_LIB"
    
    # Try to detect version
    ORT_VERSION=$(strings "$ORT_LIB" 2>/dev/null | grep -oE '^1\.[0-9]+\.[0-9]+$' | head -1)
    if [ -n "$ORT_VERSION" ]; then
        info "ONNX Runtime version: $ORT_VERSION"
    fi
else
    error "ONNX Runtime library not found!"
    echo "  Please install ONNX Runtime:"
    echo "  - Download from: https://github.com/microsoft/onnxruntime/releases"
    echo "  - Or: pip install onnxruntime (for Python, but we need the C library)"
    echo
    echo "  After installing, set ONNXRUNTIME_LIB_PATH environment variable:"
    echo "    export ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.so"
    exit 1
fi

# -----------------------------------------------------------------------------
# Check/Download Test Model
# -----------------------------------------------------------------------------
info "Checking for benchmark model..."

MODEL_PATH="${MODELS_DIR}/${DEFAULT_MODEL}/onnx/model.onnx"

if [ -f "$MODEL_PATH" ]; then
    info "Model found: $MODEL_PATH"
else
    info "Model not found. Downloading ${DEFAULT_MODEL}..."
    
    mkdir -p "${MODELS_DIR}/${DEFAULT_MODEL}/onnx"
    
    # Download from Hugging Face
    if command -v curl &>/dev/null; then
        curl -L "https://huggingface.co/sentence-transformers/${DEFAULT_MODEL}/resolve/main/onnx/model.onnx" \
            -o "$MODEL_PATH"
    elif command -v wget &>/dev/null; then
        wget -O "$MODEL_PATH" \
            "https://huggingface.co/sentence-transformers/${DEFAULT_MODEL}/resolve/main/onnx/model.onnx"
    else
        error "Neither curl nor wget found. Please install one of them."
        exit 1
    fi
    
    if [ -f "$MODEL_PATH" ]; then
        info "Model downloaded successfully!"
    else
        error "Failed to download model"
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Check Go
# -----------------------------------------------------------------------------
info "Checking Go installation..."

if command -v go &>/dev/null; then
    GO_VERSION=$(go version)
    info "Go found: $GO_VERSION"
else
    error "Go not found! Please install Go 1.21 or later."
    exit 1
fi

# -----------------------------------------------------------------------------
# Build benchmark tools
# -----------------------------------------------------------------------------
info "Building benchmark runner..."

cd "${SCRIPT_DIR}/cmd/runbench"
go build -o runbench .
info "Benchmark runner built: ${SCRIPT_DIR}/cmd/runbench/runbench"

# -----------------------------------------------------------------------------
# Download Go dependencies
# -----------------------------------------------------------------------------
info "Downloading Go dependencies..."

cd "${SCRIPT_DIR}"
go mod download
info "Dependencies downloaded"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo
echo "To run benchmarks:"
echo
echo "  Option 1: Use the benchmark runner (recommended)"
echo "    cd ${SCRIPT_DIR}/cmd/runbench"
echo "    ./runbench"
echo
echo "  Option 2: Use go test directly"
echo "    cd ${SCRIPT_DIR}"
echo "    go test -bench=. -benchmem -benchtime=5x"
echo
echo "Environment variables:"
echo "  BENCH_MODEL    - Path to ONNX model (default: ${MODEL_PATH})"
echo "  MODEL_PATH     - Alternative model path variable"
echo
echo "Examples:"
echo "  ./runbench 10x              # Run all benchmarks 10 times"
echo "  ./runbench 5x Infer         # Run only inference benchmarks"
echo "  ./runbench --help           # Show help"
echo
