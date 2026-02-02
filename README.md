# openvino-go

Go bindings for Intel OpenVINO Runtime.

## Overview

`openvino-go` provides idiomatic Go bindings for Intel OpenVINO Runtime, enabling Go applications to leverage OpenVINO's optimized inference engine for all supported neural network architectures, including transformers, CNNs, RNNs, and Graph Neural Networks (GNNs).

## Features

- **Complete OpenVINO API Coverage**: Core, Model, CompiledModel, InferRequest, Tensor
- **All Neural Network Architectures**: Transformers, CNNs, RNNs, GNNs, Generative Models
- **Multiple Model Formats**: OpenVINO IR, ONNX, PyTorch, TensorFlow, PaddlePaddle, JAX/Flax
- **Device Support**: CPU, GPU, NPU, AUTO, HETERO, BATCH
- **Performance Optimizations**: Quantization, model caching, automatic batching
- **Idiomatic Go**: Type-safe, context-aware, proper resource management
- **Cross-Platform**: Linux, macOS, Windows (Intel x86-64, ARM64, Apple Silicon)

## Status

**Early Development** - This project is in active development. 

## Quick Start

```go
package main

import (
    "context"
    "github.com/accretional/openvino-go"
)

func main() {
    // Create core
    core, err := openvino.NewCore()
    if err != nil {
        panic(err)
    }
    defer core.Close()

    // Load model
    model, err := core.ReadModel("model.xml")
    if err != nil {
        panic(err)
    }
    defer model.Close()

    // Compile for CPU
    compiledModel, err := core.CompileModel(model, "CPU")
    if err != nil {
        panic(err)
    }
    defer compiledModel.Close()

    // Create infer request
    request, err := compiledModel.CreateInferRequest()
    if err != nil {
        panic(err)
    }
    defer request.Close()

    // Set input tensor
    inputData := []float32{1.0, 2.0, 3.0}
    err = request.SetInputTensorByIndex(0, inputData, []int64{1, 3})
    if err != nil {
        panic(err)
    }

    // Run inference
    err = request.Infer(context.Background())
    if err != nil {
        panic(err)
    }

    // Get output
    outputTensor, err := request.GetOutputTensorByIndex(0)
    if err != nil {
        panic(err)
    }
    defer outputTensor.Close()

    // Use results
    results := outputTensor.GetData()
    // ...
}
```

## Use Cases

### Text Embeddings
Generate embeddings from text using transformer models (BERT, sentence-transformers, etc.)

### Graph Neural Network Embeddings
Embed graph structures (code ASTs, knowledge graphs, collection relationships)

### Image Classification
Classify images using CNN models (ResNet, MobileNet, YOLO)

### Batch Inference
Process multiple inputs efficiently with automatic batching

## Requirements

- Go 1.21 or later
- OpenVINO 2024.x or later
- C compiler (for CGO)

## Installation

```bash
go get github.com/accretional/openvino-go
```

## Documentation

- [Implementation Plan](docs/openvino_go_bindings_plan.md) - Comprehensive development plan
- [API Reference](docs/api.md) - Complete API documentation (coming soon)
- [Examples](examples/) - Usage examples (coming soon)

## Performance

**Target Performance**:
- Latency: ≤ 100% of onnxruntime_go (equal or better)
- Throughput: ≥ 100% of onnxruntime_go (equal or better)
- Memory: ≤ 100% of onnxruntime_go (equal or better)

**On Intel CPUs**: OpenVINO optimizations should exceed onnxruntime_go performance (1.5-3x faster).

## Architecture

Three-layer approach:
```
Go API → CGO Bindings → C Wrapper → OpenVINO C++ Runtime
```

## Contributing

Contributions welcome! This is an open-source project designed for community contributions.

## License

MIT

## References

- [OpenVINO Documentation](https://docs.openvino.ai/)
- [OpenVINO C++ API Reference](https://docs.openvino.ai/2025/api/api_reference.html)
- [GraNNite Repository](https://github.com/arghadippurdue/GraNNite) - OpenVINO GNN support proof
- [ONNX Runtime Go Bindings](https://github.com/yalue/onnxruntime_go) - Reference implementation
