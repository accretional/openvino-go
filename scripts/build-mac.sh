#!/usr/bin/env bash
set -euo pipefail

OPENVINO_VERSION="2026.1.0"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CW_DIR="$PROJECT_ROOT/internal/cwrapper"
PREBUILT_DIR="$CW_DIR/prebuilt"

# Resolve OPENVINO_ROOT
if [ -n "${OPENVINO_ROOT:-}" ] && [ -d "${OPENVINO_ROOT}/runtime/include" ]; then
    echo "==> Using OpenVINO from \$OPENVINO_ROOT: $OPENVINO_ROOT"
elif [ -d "$HOME/.local/openvino-${OPENVINO_VERSION}/runtime/include" ]; then
    export OPENVINO_ROOT="$HOME/.local/openvino-${OPENVINO_VERSION}"
    echo "==> Using OpenVINO from default install: $OPENVINO_ROOT"
else
    echo "Error: OpenVINO not found. Run: make setup  (or set OPENVINO_ROOT)" >&2
    exit 1
fi

INCLUDE_FLAGS="-I${OPENVINO_ROOT}/runtime/include"
LIB_FLAGS="-L${OPENVINO_ROOT}/runtime/lib/arm64/Release -Wl,-rpath,${OPENVINO_ROOT}/runtime/lib/arm64/Release"

echo "==> Building C++ wrapper (macOS arm64)..."

clang++ -std=c++17 -fPIC -O2 -Wall -Wno-deprecated-declarations \
    $INCLUDE_FLAGS \
    -c "$CW_DIR/core_wrapper.cpp" \
    -o "$CW_DIR/core_wrapper.o"

mkdir -p "$PREBUILT_DIR"

clang++ -dynamiclib \
    -install_name "@rpath/libopenvino_wrapper.dylib" \
    -o "$PREBUILT_DIR/libopenvino_wrapper.dylib" \
    "$CW_DIR/core_wrapper.o" \
    $LIB_FLAGS -lopenvino

rm -f "$CW_DIR/core_wrapper.o"
echo "==> Built $PREBUILT_DIR/libopenvino_wrapper.dylib"
