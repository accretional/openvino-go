# ovmodel - OpenVINO Model Downloader

A simple CLI tool to download models for use with the openvino-go bindings.

## Installation

```bash
go install ./cmd/ovmodel
```

Or run directly:
```bash
go run cmd/ovmodel/main.go [flags]
```

## Usage

### List available models

```bash
ovmodel -list
```

### Download a model by alias

```bash
ovmodel -model all-MiniLM-L6-v2
```

### Download a HuggingFace model directly

```bash
ovmodel -model sentence-transformers/all-MiniLM-L6-v2
```

### Download from a URL

```bash
ovmodel -url https://example.com/model.onnx
```

### Specify output directory

```bash
ovmodel -model all-MiniLM-L6-v2 -output ./my-models
```

### Force re-download

```bash
ovmodel -model all-MiniLM-L6-v2 -force
```

## Examples

```bash
# Download a text embedding model
ovmodel -model all-MiniLM-L6-v2

# Download from HuggingFace with full path
ovmodel -model sentence-transformers/all-mpnet-base-v2

# Download test model (simple ONNX model for testing)
ovmodel -model test-model
```

## Model Storage

Models are stored in the `models/` directory by default:
```
models/
  ├── sentence-transformers_all-MiniLM-L6-v2/
  │   └── model.onnx
  └── test_model.onnx
```

Note: The `test-model` downloads a simple ONNX model that works with both the hello-world example and tests. OpenVINO supports both `.onnx` and `.xml` formats.

## Supported Sources

- **HuggingFace Hub**: Automatically downloads ONNX models
- **Direct URLs**: Download any model file from a URL
- **OpenVINO Model Zoo**: (Coming soon)

## Notes

- Models are cached locally - re-running the same command won't re-download. Use `-force` to force re-download
- The tool automatically tries multiple paths for HuggingFace models (main/model.onnx, onnx/model.onnx, etc.)
