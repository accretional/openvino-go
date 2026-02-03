# openvino-go

Go bindings for Intel OpenVINO Runtime.

## Install

This module uses **CGO** and requires the **Intel OpenVINO Runtime** and a C++ toolchain to build. Add it to your project with:

```bash
go get github.com/accretional/openvino-go/pkg/openvino@latest
```

Import in your code:

```go
import "github.com/accretional/openvino-go/pkg/openvino"
```

Make sure CGO is enabled when building (`CGO_ENABLED=1`, which is the default when cgo is available). Your build environment must have OpenVINO installed and the C++ wrapper built (See Setup and Build below).

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

## Test

Run all tests:

```bash
scripts/download-model.sh   # Generates a small model (requires: pip install openvino)
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
