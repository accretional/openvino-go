#!/usr/bin/env bash
set -euo pipefail

# Build the C++ wrapper shared library required by CGO

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CW_DIR="$PROJECT_ROOT/internal/cwrapper"

OPENVINO_ROOT="${OPENVINO_ROOT:-}"
ARCH="${ARCH:-intel64}"

# Determine include/lib paths
if [ -n "$OPENVINO_ROOT" ] && [ -d "$OPENVINO_ROOT/runtime/include" ]; then
    # Tarball / manual install
    INCLUDE_FLAGS="-I${OPENVINO_ROOT}/runtime/include"
    LIB_FLAGS="-L${OPENVINO_ROOT}/runtime/lib/${ARCH} -Wl,-rpath,${OPENVINO_ROOT}/runtime/lib/${ARCH}"
    echo "==> Using OpenVINO from $OPENVINO_ROOT"
elif [ -f /usr/include/openvino/openvino.hpp ]; then
    # APT / system install
    INCLUDE_FLAGS=""
    LIB_FLAGS=""
    echo "==> Using system-installed OpenVINO"
else
    echo "Error: OpenVINO not found. Run scripts/setup.sh or set OPENVINO_ROOT." >&2
    exit 1
fi

echo "==> Building C++ wrapper..."

g++ -std=c++17 -fPIC -O2 -Wall -Wno-deprecated-declarations \
    $INCLUDE_FLAGS \
    -c "$CW_DIR/core_wrapper.cpp" \
    -o "$CW_DIR/core_wrapper.o"

g++ -shared \
    -o "$CW_DIR/libopenvino_wrapper.so" \
    "$CW_DIR/core_wrapper.o" \
    $LIB_FLAGS -lopenvino

echo "==> Built $CW_DIR/libopenvino_wrapper.so"
