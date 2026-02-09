# openvino-go

Go bindings for Intel OpenVINO Runtime.

Use it with **go get** — no clone or build scripts required for typical use.

## Quick start

**Import path:** `github.com/accretional/openvino-go/pkg/openvino`

1. **Install OpenVINO**: 
   Follow [these steps to install OpenVINO](https://docs.openvino.ai/2025/get-started/install-openvino.html)

2. **Add the package** and build with CGO:
   ```bash
   go get github.com/accretional/openvino-go
   CGO_ENABLED=1 go build .
   ```

3. **In your code:** `import "github.com/accretional/openvino-go/pkg/openvino"`

A **prebuilt C++ wrapper** is included at `internal/cwrapper/prebuilt/libopenvino_wrapper.so` (Linux amd64, system OpenVINO). After `go get`, you only need OpenVINO installed and `CGO_ENABLED=1`; no need to clone the repo or run `scripts/build.sh`.

**Other platforms or custom OpenVINO:** run once to build the wrapper:
```bash
go generate ./...
```

## Using from this repo

1. **Clone** the repository.
2. **Setup** – run `scripts/setup.sh` to install OpenVINO and build tools (if not already installed).
3. **Build** – run `scripts/build.sh` (or `go generate ./...`) to build the C++ wrapper if you need to rebuild it.
4. **Use** – run the examples, run tests, or depend on the package from another module.

## Prerequisites

- Linux (x86-64) for the included prebuilt wrapper; other platforms use `go generate ./...` to build.
- Go 1.21+
- Intel OpenVINO Runtime 2024.x+ (must be installed for linking and runtime).

## Setup (when not using the prebuilt wrapper)

Install OpenVINO and build tools (for clone-and-build or to rebuild the wrapper):

```bash
scripts/setup.sh
```

## Build (when developing in this repo or on non-Linux)

Rebuild the C++ wrapper:

```bash
scripts/build.sh
# or
go generate ./...
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
