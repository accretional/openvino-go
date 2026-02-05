# OpenVINO vs ONNX Runtime Benchmarks

Performance comparison benchmarks between OpenVINO and ONNX Runtime inference engines.

## Quick Start

```bash
# Setup (checks dependencies, downloads model, builds tools)
./setup.sh

# Run benchmarks with formatted output
cd cmd/runbench
./runbench
```

## Prerequisites

- **Go 1.21+**
- **OpenVINO** - Intel's inference toolkit
  - Install from: https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html
  - Ensure `libopenvino.so` is in your library path
- **ONNX Runtime** - Cross-platform inference engine
  - Install from: https://github.com/microsoft/onnxruntime/releases
  - Set `ONNXRUNTIME_LIB_PATH` to the library location

## Benchmark Types

| Benchmark | Description |
|-----------|-------------|
| **Load** | Model loading time |
| **Infer** | Single inference latency |
| **InferParallel** | Parallel inference using all CPU cores |
| **FirstInference** | Cold start latency (load + first inference) |
| **Throughput** | Maximum inferences per second |
| **BatchSize** | Performance scaling with batch sizes 1, 2, 4, 8 |
| **SeqLen** | Performance scaling with sequence lengths |
| **Threads** | Thread count scaling (1, 2, 4, 8 threads) |
| **Memory** | Memory consumption during inference |
| **ConcurrentSessions** | Multiple independent model instances (1, 2 sessions) |
| **AsyncInference** | OpenVINO async API with multiple InferRequests (recommended pattern) |

### Concurrency Patterns

The benchmarks test two different concurrency patterns:

1. **ConcurrentSessions**: Creates multiple independent model instances, each with its own
   compiled model. This pattern is common when you need isolated sessions but can suffer
   from thread oversubscription.

2. **AsyncInference** (OpenVINO only): Uses a single compiled model with multiple
   `InferRequest` objects and the async API (`StartAsync`/`Wait`). This is the recommended
   pattern for OpenVINO as it:
   - Shares resources efficiently
   - Scales better with increasing concurrency
   - Avoids thread oversubscription

## Running Benchmarks

### Using the Benchmark Runner (Recommended)

The benchmark runner provides formatted, readable output with comparisons:

```bash
cd cmd/runbench

# Run all benchmarks (5 iterations each)
./runbench

# Run with more iterations for stable results
./runbench 10x

# Run for a specific duration
./runbench 1s

# Filter to specific benchmarks
./runbench 5x Infer       # Only inference benchmarks
./runbench 5x Thread      # Only thread scaling
./runbench 5x Throughput  # Only throughput test

# Show help
./runbench --help
```

### Using Go Test Directly

For raw benchmark output:

```bash
cd benchmarks

# Run all benchmarks
go test -bench=. -benchmem -benchtime=5x

# Run specific benchmarks
go test -bench=BenchmarkOpenVINO_Infer -benchmem

# Run with more iterations
go test -bench=. -benchmem -benchtime=10x

# Run for duration instead of iterations
go test -bench=. -benchmem -benchtime=2s
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BENCH_MODEL` | Path to ONNX model file | `../models/test_model.onnx` |
| `MODEL_PATH` | Alternative model path | - |
| `ONNXRUNTIME_LIB_PATH` | Path to libonnxruntime.so | System default |

### Default Model

The benchmarks use the `all-MiniLM-L6-v2` sentence transformer model by default:
- **Inputs**: token IDs, attention mask, token type IDs (dynamic batch, sequence length)
- **Output**: 384-dimensional embeddings
- **Size**: ~23MB

To use a different model:
```bash
export BENCH_MODEL=/path/to/your/model.onnx
./runbench
```

## Output Format

The benchmark runner produces formatted output like:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║               OpenVINO vs ONNX Runtime Benchmark Comparison                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─ Benchmark Configuration ────────────────────────────────────────────────────
│  Model:       all-MiniLM-L6-v2
│  Platform:    linux/amd64
│  CPU Cores:   8
│  ...

┌─ Single Inference ───────────────────────────────────────────────────────────
│  OpenVINO:             5.18ms     ONNX Runtime:         8.10ms
│  Result: OpenVINO 1.6x faster

┌─ Thread Scaling ─────────────────────────────────────────────────────────────
│  Variant                  OpenVINO   ONNX Runtime         Comparison
│  ────────────────────────────────────────────────────────────────────
│  threads_1                  5.45ms         6.57ms     OV 1.2x faster
│  threads_2                  5.44ms         6.43ms     OV 1.2x faster
│  ...

╔══════════════════════════════════════════════════════════════════════════════╗
║                                   SUMMARY                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Test Results:  OpenVINO wins 10/14  |  ONNX Runtime wins 4/14
```

## Notes

- Thread scaling tests may show different patterns based on CPU architecture
- First run may be slower due to model compilation/optimization
