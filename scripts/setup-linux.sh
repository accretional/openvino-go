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
    UBUNTU_VER=$(. /etc/os-release 2>/dev/null && echo "$VERSION_ID" | cut -d. -f1)
    UBUNTU_VER="${UBUNTU_VER:-22}"
    echo "==> Detected Ubuntu ${UBUNTU_VER}"

    # Pick repo codename for OpenVINO 2025 APT repository
    case "$UBUNTU_VER" in
        20)
            APT_CODENAME="ubuntu20"
            ;;
        22)
            APT_CODENAME="ubuntu22"
            ;;
        24)
            APT_CODENAME="ubuntu24"
            ;;
        *)
            echo "Warning: Unsupported Ubuntu version ${UBUNTU_VER}, using ubuntu22." >&2
            APT_CODENAME="ubuntu22"
            ;;
    esac

    OPENVINO_VERSION="${OPENVINO_VERSION:-2026.1.0}"
    echo "==> Installing OpenVINO ${OPENVINO_VERSION} (${APT_CODENAME})..."

    sudo apt-get update
    sudo apt-get install -y gnupg ca-certificates wget

    # Step 1: Download the Intel GPG key
    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

    # Step 2: Add the key to the system keyring
    sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

    # Step 3: Add the OpenVINO APT repository for the detected Ubuntu version
    echo "deb https://apt.repos.intel.com/openvino ${APT_CODENAME} main" \
        | sudo tee /etc/apt/sources.list.d/intel-openvino.list

    # Step 4: Update the package list
    sudo apt update

    # Step 5: Verify available OpenVINO packages
    apt-cache search openvino

    # Step 6: Install OpenVINO Runtime
    sudo apt install -y "openvino-${OPENVINO_VERSION}"

    echo "==> OpenVINO ${OPENVINO_VERSION} installed"
fi

echo "==> Setup complete. Run scripts/build.sh next."
