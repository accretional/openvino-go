# openvino-go

Go bindings for Intel OpenVINO Runtime.

## Prerequisites

- Linux (x86-64)
- Go 1.21+
- g++ with C++17 support
- Intel OpenVINO Runtime 2024.x+

### Install g++

```bash
sudo apt-get update && sudo apt-get install -y g++
```

### Install OpenVINO (Ubuntu 22.04 / 24.04)

```bash
sudo apt-get install -y gnupg ca-certificates curl

curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | sudo gpg --dearmor -o /usr/share/keyrings/intel-openvino.gpg

# Use "ubuntu22" for 22.04, "ubuntu24" for 24.04
echo "deb [signed-by=/usr/share/keyrings/intel-openvino.gpg] https://apt.repos.intel.com/openvino/2025 ubuntu24 main" \
  | sudo tee /etc/apt/sources.list.d/intel-openvino.list

sudo apt-get update && sudo apt-get install -y openvino
```

For other install methods, see the [OpenVINO install guide](https://docs.openvino.ai/2025/get-started/install-openvino.html).

## Installation

```bash
go get github.com/accretional/openvino-go
```

## Usage

```go
import "github.com/accretional/openvino-go/pkg/openvino"
```

CGo compiles the C++ wrapper automatically during `go build`. If OpenVINO is not in the default system paths, point CGo at it:

```bash
export CGO_CXXFLAGS="-I/path/to/openvino/runtime/include"
export CGO_LDFLAGS="-L/path/to/openvino/runtime/lib/intel64"
```

## Testing

```bash
go test ./...
```

Tests that require a model need `OPENVINO_TEST_MODEL` set:

```bash
go run cmd/ovmodel/main.go -model test-model
export OPENVINO_TEST_MODEL=models/test_model.onnx
go test ./... -v
```

See the `examples/` directory for working demos.

## Features

- Synchronous and asynchronous inference
- Tensor operations (input/output tensor management)
- Device enumeration and selection
- Performance optimizations (performance hints, stream configuration)
- Model I/O introspection
