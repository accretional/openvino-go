# openvino-go

Go bindings for Intel OpenVINO Runtime.

## Prerequisites

- Linux (x86-64)
- Go 1.21+
- g++ with C++17 support
- Intel OpenVINO Runtime 2024.x+

Run the script to have all the required installations inclusing OpenVINO:

```bash
scripts/setup.sh
```

Or install manually via Intel's [APT repository](https://docs.openvino.ai/2024/get-started/install-openvino.html).

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
