#!/bin/bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

OPENVINO_VERSION="2026.1.0"
OPENVINO_BUILD="21367.63e31528c62"
OPENVINO_ARCHIVE="openvino_toolkit_macos_12_6_${OPENVINO_VERSION}.${OPENVINO_BUILD}_arm64.tgz"
OPENVINO_URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.1/macos/${OPENVINO_ARCHIVE}"
OPENVINO_INSTALL_DIR="$HOME/.local/openvino-${OPENVINO_VERSION}"

echo -e "${YELLOW}Setting up openvino-go dependencies for macOS (arm64)...${NC}"

# Xcode CLT
if ! xcode-select -p &>/dev/null; then
    echo -e "${YELLOW}Installing Xcode Command Line Tools...${NC}"
    xcode-select --install
    echo -e "${RED}Please complete the Xcode CLT installation prompt, then re-run this script.${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Xcode Command Line Tools found${NC}"
fi

# Homebrew
if ! command -v brew &>/dev/null; then
    echo -e "${RED}✗ Homebrew not found. Install from https://brew.sh then re-run.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Homebrew found${NC}"

# Go
if command -v go &>/dev/null; then
    echo -e "${GREEN}✓ Go already installed: $(go version)${NC}"
else
    echo -e "${YELLOW}Installing Go...${NC}"
    brew install go
    echo -e "${GREEN}✓ Go installed${NC}"
fi

# protoc
if command -v protoc &>/dev/null; then
    echo -e "${GREEN}✓ protoc already installed${NC}"
else
    echo -e "${YELLOW}Installing protobuf...${NC}"
    brew install protobuf
    echo -e "${GREEN}✓ protoc installed${NC}"
fi

# protoc-gen-go / protoc-gen-go-grpc
export PATH="$PATH:$(go env GOPATH)/bin"

if ! command -v protoc-gen-go &>/dev/null; then
    echo -e "${YELLOW}Installing protoc-gen-go...${NC}"
    go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
    echo -e "${GREEN}✓ protoc-gen-go installed${NC}"
else
    echo -e "${GREEN}✓ protoc-gen-go already installed${NC}"
fi

if ! command -v protoc-gen-go-grpc &>/dev/null; then
    echo -e "${YELLOW}Installing protoc-gen-go-grpc...${NC}"
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
    echo -e "${GREEN}✓ protoc-gen-go-grpc installed${NC}"
else
    echo -e "${GREEN}✓ protoc-gen-go-grpc already installed${NC}"
fi

# OpenVINO
if [ -f "${OPENVINO_INSTALL_DIR}/runtime/include/openvino/openvino.hpp" ]; then
    echo -e "${GREEN}✓ OpenVINO ${OPENVINO_VERSION} already installed at ${OPENVINO_INSTALL_DIR}${NC}"
else
    echo -e "${YELLOW}Downloading OpenVINO ${OPENVINO_VERSION} for macOS arm64...${NC}"
    TMP_DIR=$(mktemp -d)
    trap "rm -rf '$TMP_DIR'" EXIT

    curl -L --progress-bar -H "Referer: https://storage.openvinotoolkit.org/" "${OPENVINO_URL}" -o "${TMP_DIR}/${OPENVINO_ARCHIVE}"

    DOWNLOADED_SIZE=$(wc -c < "${TMP_DIR}/${OPENVINO_ARCHIVE}")
    if [ "$DOWNLOADED_SIZE" -lt 1000000 ]; then
        echo -e "${RED}✗ Download appears incomplete or invalid (${DOWNLOADED_SIZE} bytes). Check the URL or network.${NC}"
        exit 1
    fi

    echo -e "${YELLOW}Extracting OpenVINO...${NC}"
    mkdir -p "$HOME/.local"
    tar -xzf "${TMP_DIR}/${OPENVINO_ARCHIVE}" -C "$HOME/.local/"

    EXTRACTED=$(find "$HOME/.local" -maxdepth 1 -name "openvino_toolkit_macos*" -type d | head -1)
    if [ -z "$EXTRACTED" ]; then
        echo -e "${RED}✗ Could not find extracted OpenVINO directory under ~/.local${NC}"
        exit 1
    fi

    mv "$EXTRACTED" "$OPENVINO_INSTALL_DIR"
    echo -e "${GREEN}✓ OpenVINO ${OPENVINO_VERSION} installed to ${OPENVINO_INSTALL_DIR}${NC}"
fi

# Update ~/.zshrc
ZSHRC="$HOME/.zshrc"
UPDATED=false

if ! grep -q "# openvino-go environment" "$ZSHRC" 2>/dev/null; then
    echo "" >> "$ZSHRC"
    echo "# openvino-go environment" >> "$ZSHRC"
    UPDATED=true
fi

if ! grep -q "OPENVINO_ROOT" "$ZSHRC" 2>/dev/null; then
    echo "export OPENVINO_ROOT=\"${OPENVINO_INSTALL_DIR}\"" >> "$ZSHRC"
    echo "export DYLD_LIBRARY_PATH=\"\${OPENVINO_ROOT}/runtime/lib/arm64/Release\${DYLD_LIBRARY_PATH:+:\$DYLD_LIBRARY_PATH}\"" >> "$ZSHRC"
    UPDATED=true
fi

GO_USER_BIN="$(go env GOPATH)/bin"
if ! grep -q "PATH.*$(go env GOPATH)/bin" "$ZSHRC" 2>/dev/null; then
    echo "export PATH=\"\$PATH:${GO_USER_BIN}\"" >> "$ZSHRC"
    UPDATED=true
fi

# Apply to current session
export OPENVINO_ROOT="${OPENVINO_INSTALL_DIR}"
export DYLD_LIBRARY_PATH="${OPENVINO_ROOT}/runtime/lib/arm64${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export PATH="$PATH:$GO_USER_BIN"

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
if [ "$UPDATED" = true ]; then
    echo -e "${YELLOW}To apply environment changes in your current shell:${NC}"
    echo -e "${GREEN}  source ${ZSHRC}${NC}"
fi
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  Build the C++ wrapper: ./scripts/build-mac.sh"
