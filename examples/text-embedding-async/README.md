# Text Embedding Async Example

This example demonstrates how to use **asynchronous inference** with OpenVINO for batch text embedding processing. It shows how to process multiple texts concurrently using async inference to improve throughput.

## Key Features

- **Async Inference**: Uses `StartAsync()` and `Wait()` for non-blocking inference
- **Parallel Processing**: Processes multiple texts concurrently using a pool of inference requests
- **Performance Optimization**: Uses throughput mode and multiple streams for better performance
- **Request Pooling**: Reuses inference requests to minimize overhead

## Prerequisites

- OpenVINO Runtime installed and configured
- A text embedding model (e.g., sentence-transformers model in ONNX format)

### Getting a Model

Download a text embedding model using the `ovmodel` CLI:

```bash
go run cmd/ovmodel/main.go -model all-MiniLM-L6-v2
```

## Usage

```bash
# Process multiple texts concurrently
go run examples/text-embedding-async/main.go \
  models/sentence-transformers_all-MiniLM-L6-v2/model.onnx \
  "Hello, world!" \
  "How are you?" \
  "Good morning!" \
  "Nice to meet you!"
```

## Example Output

```
Text Embedding Async Example
============================

Model: models/sentence-transformers_all-MiniLM-L6-v2/model.onnx
Texts to process: 4

Creating OpenVINO Core...
Available devices: [CPU]

Loading model from: models/sentence-transformers_all-MiniLM-L6-v2/model.onnx

Compiling model for CPU with throughput optimization...
Model compiled successfully for device: CPU

Processing 4 texts using async inference...

=== Results ===

Text 1: "Hello, world!"
  Embedding dimension: 384
  First 5 values: [0.0123, -0.0234, 0.0345, -0.0456, 0.0567]
  Processing time: 45.2ms

Text 2: "How are you?"
  Embedding dimension: 384
  First 5 values: [0.0234, -0.0345, 0.0456, -0.0567, 0.0678]
  Processing time: 42.1ms

Text 3: "Good morning!"
  Embedding dimension: 384
  First 5 values: [0.0345, -0.0456, 0.0567, -0.0678, 0.0789]
  Processing time: 43.8ms

Text 4: "Nice to meet you!"
  Embedding dimension: 384
  First 5 values: [0.0456, -0.0567, 0.0678, -0.0789, 0.0890]
  Processing time: 44.5ms

=== Performance Summary ===
Total texts processed: 4
Total time: 125ms
Average time per text: 31.25ms
Throughput: 32.00 texts/second

Async inference example completed successfully!
```

## Tips for Best Performance

1. **Use Throughput Mode**: Compile model with `PerformanceModeThroughput` for better parallel performance
2. **Multiple Streams**: Use `NumStreams(4)` or more for CPU inference
3. **Request Pooling**: Reuse inference requests instead of creating new ones
4. **Batch Size**: Process multiple texts together when possible
5. **Device Selection**: Use GPU if available for better parallel performance

## See Also

- `examples/text-embedding/` - Synchronous text embedding example
- `examples/hello-world/` - Basic inference pipeline
- `pkg/openvino/doc.go` - API documentation
