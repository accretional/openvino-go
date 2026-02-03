# openvino-go

Go bindings for Intel OpenVINO Runtime.

For now, the supported way to use this project is: **clone the repo, build it, then use it from here.** We are not yet promoting use as a standalone Go package (`go get`) until the build and distribution story is streamlined.

## How to use (for now)

1. **Clone** the repository.
2. **Setup** – run `scripts/setup.sh` to install OpenVINO and build tools.
3. **Build** – run `scripts/build.sh` to build the C++ wrapper.
4. **Use** – run the example, run tests, or use the package from another Go module by adding a `replace` in your `go.mod` pointing at this clone:

   ```go
   replace github.com/accretional/openvino-go => /path/to/openvino-go
   ```

   Then in your code: `import "github.com/accretional/openvino-go/pkg/openvino"` and build with `CGO_ENABLED=1`.

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

From the repo root (after Setup and Build):

```bash
go test ./...
```

To run tests that need a model (compile/inference), generate one and set `OPENVINO_TEST_MODEL`:

```bash
scripts/download-model.sh   # requires: pip install openvino
export OPENVINO_TEST_MODEL=models/test_model.xml
go test ./... -v
```

## Example

From the repo root (after Setup and Build):

```bash
scripts/download-model.sh
go run examples/hello-world/main.go models/test_model.xml
```

## Status

**Early Development** - This project is in active development. 
