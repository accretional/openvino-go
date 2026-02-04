#!/usr/bin/env bash
set -euo pipefail

# Install dependencies for openvino-go

echo "==> Checking system dependencies..."

# Install build tools if missing
if ! command -v g++ &>/dev/null; then
    echo "==> Installing g++..."
    sudo apt-get update && sudo apt-get install -y g++
fi

if ! command -v curl &>/dev/null; then
    echo "==> Installing curl..."
    sudo apt-get update && sudo apt-get install -y curl
fi

if ! command -v go &>/dev/null; then
    echo "==> Installing Go..."
    GO_VERSION="${GO_VERSION:-1.25.7}"
    wget -q "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -O /tmp/go.tar.gz
    sudo tar -C /usr/local -xzf /tmp/go.tar.gz
    rm /tmp/go.tar.gz
    export PATH=$PATH:/usr/local/go/bin
fi

# Install OpenVINO if not already present
if [ -f /usr/include/openvino/openvino.hpp ]; then
    echo "==> OpenVINO already installed"
else
    # Detect Ubuntu version
    UBUNTU_VER=$(. /etc/os-release && echo "$VERSION_ID" | cut -d. -f1)
    echo "==> Detected Ubuntu ${UBUNTU_VER}"

    # Pick compatible version and repo codename
    case "$UBUNTU_VER" in
        24)
            APT_CODENAME="ubuntu24"
            OPENVINO_VERSION="${OPENVINO_VERSION:-2024.6.0}"
            ;;
        22)
            APT_CODENAME="ubuntu22"
            OPENVINO_VERSION="${OPENVINO_VERSION:-2024.4.0}"
            ;;
        *)
            echo "Error: Unsupported Ubuntu version ${UBUNTU_VER}. Need 22.04 or 24.04." >&2
            exit 1
            ;;
    esac

    echo "==> Installing OpenVINO ${OPENVINO_VERSION} (${APT_CODENAME})..."

    sudo apt-get update
    sudo apt-get install -y gnupg ca-certificates

    # Add Intel APT repository
    curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | sudo gpg --dearmor -o /usr/share/keyrings/intel-openvino.gpg
    echo "deb [signed-by=/usr/share/keyrings/intel-openvino.gpg] https://apt.repos.intel.com/openvino/2024 ${APT_CODENAME} main" \
        | sudo tee /etc/apt/sources.list.d/intel-openvino.list

    sudo apt-get update
    sudo apt-get install -y "openvino-${OPENVINO_VERSION}"

    echo "==> OpenVINO ${OPENVINO_VERSION} installed"
fi

echo "==> Setup complete. You can now run 'go build ./...'."
