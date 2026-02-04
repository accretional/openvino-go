# openvino-go

Go bindings for Intel OpenVINO Runtime.

For now, the supported way to use this project is: **clone the repo, build it, then use it from here.** 

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

To run tests that need a model (compile/inference), download one and set `OPENVINO_TEST_MODEL`:

```bash
go run cmd/ovmodel/main.go -model test-model
export OPENVINO_TEST_MODEL=models/test_model.onnx
go test ./... -v
```

## Examples

### Hello World Example

Basic inference pipeline demonstration:

```bash
# Download test model
go run cmd/ovmodel/main.go -model test-model

# Run the example (OpenVINO supports both .onnx and .xml)
go run examples/hello-world/main.go models/test_model.onnx
```

### Text Embedding Example

Text embedding inference with transformer models:

```bash
# Download a text embedding model
go run cmd/ovmodel/main.go -model all-MiniLM-L6-v2

# Run the example
go run examples/text-embedding/main.go models/sentence-transformers_all-MiniLM-L6-v2/model.onnx "Your text here"
```

See `examples/text-embedding/README.md` for more details on getting and using embedding models.

### Text Embedding Async Example

Asynchronous inference for batch text embedding processing:

```bash
# Download a text embedding model
go run cmd/ovmodel/main.go -model all-MiniLM-L6-v2

# Process multiple texts concurrently using async inference
go run examples/text-embedding-async/main.go \
  models/sentence-transformers_all-MiniLM-L6-v2/model.onnx \
  "Hello, world!" \
  "How are you?" \
  "Good morning!"
```

This example demonstrates:
- Async inference with `StartAsync()` and `Wait()`
- Parallel processing of multiple texts
- Request pooling for better performance
- Throughput optimization

See `examples/text-embedding-async/README.md` for more details.

## Troubleshooting

`go: creating work dir: ... permission denied` or `stat /tmp: no such file or directory`

Go uses `/tmp` (or `GOTMPDIR`) for its build cache when compiling. If `/tmp` does not exist or your user cannot write to it (e.g. in some containers or restricted environments), any `go run` or `go build` can fail with one of these errors.

**Fix:** Use a writable directory for Go’s temp files:

```bash
mkdir -p "$HOME/.tmp"
export TMPDIR=$HOME/.tmp GOTMPDIR=$HOME/.tmp
```

Then run your `go run` or `go test` as usual. To make this persistent, add the `export` lines to `~/.bashrc` or `~/.profile`.

## Features

- Synchronous and asynchronous inference
- Tensor operations (input/output tensor management)
- Device enumeration and selection
- Performance optimizations (performance hints, stream configuration)
- Model I/O introspection
