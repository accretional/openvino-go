# openvino-go

Go bindings for Intel OpenVINO Runtime.

## Prerequisites

- Linux (x86-64)
- Go 1.21+
- g++ with C++17 support
- Intel OpenVINO Runtime 2024.x+

## Setup

Install OpenVINO and build tools:

```bash
scripts/setup.sh
```

## Build

```bash
scripts/build.sh
```

Then use the package normally with `go build`, `go run`, etc. CGO links against the wrapper automatically.

## Test

Run all tests:

```bash
scripts/download-model.sh # Generates a small model using OpenVINO Python pkg. Needs "pip install openvino"
export OPENVINO_TEST_MODEL=models/test_model.xml
go test ./... -v
```

## Example

```bash
scripts/download-model.sh
go run examples/hello-world/main.go models/test_model.xml
```

## Status

**Early Development** - This project is in active development. 
