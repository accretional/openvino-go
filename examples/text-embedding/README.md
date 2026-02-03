# Text Embedding Example

This example demonstrates how to use OpenVINO for text embedding inference with transformer models (BERT, sentence-transformers, etc.).

## Prerequisites

- OpenVINO Runtime installed and configured
- A text embedding model (e.g., sentence-transformers model in ONNX format)

### Getting a Model

Download a pre-converted text embedding model using the `ovmodel` CLI:

```bash
# Download a text embedding model
go run cmd/ovmodel/main.go -model all-MiniLM-L6-v2

# Or download any HuggingFace model directly
go run cmd/ovmodel/main.go -model sentence-transformers/paraphrase-MiniLM-L6-v2
```

The `ovmodel` tool automatically downloads ONNX models from HuggingFace Hub. See `cmd/ovmodel/README.md` for more details.

Alternatively, you can download pre-converted models from the [OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo).

## Usage

```bash
# First, download a model
go run cmd/ovmodel/main.go -model all-MiniLM-L6-v2

# Then run the example
go run examples/text-embedding/main.go models/sentence-transformers_all-MiniLM-L6-v2/model.onnx "Hello, world! This is a test."
```

## Async Inference

For better throughput, you can use asynchronous inference:

```go
// Start async inference
err = request.StartAsync()
if err != nil {
    log.Fatal(err)
}

// Do other work while inference runs...

// Wait for completion
err = request.Wait()
if err != nil {
    log.Fatal(err)
}

// Or use convenience method
err = request.InferAsync()

// Or with timeout
completed, err := request.WaitFor(5000) // 5 seconds
if !completed {
    log.Fatal("inference timed out")
}
```

## Input Format

The example expects:
- **Model path**: Path to an OpenVINO IR (`.xml`) or ONNX (`.onnx`) model file
- **Text**: The text string to embed (can contain spaces, use quotes)

Common compatible models:
- `all-MiniLM-L6-v2` (384 dimensions)
- `all-mpnet-base-v2` (768 dimensions)
- `paraphrase-MiniLM-L6-v2` (384 dimensions)
- Any BERT-based sentence transformer model

## Example Output

```
Text Embedding Example
======================

Model: model.xml
Text: "Hello, world!"

Creating OpenVINO Core...
Available devices: [CPU]

Loading model from: model.xml

=== Model I/O Information ===
Model has 2 input(s):
  Input 0: name='input_ids', shape=[1 512], type=1
  Input 1: name='attention_mask', shape=[1 512], type=1
Model has 1 output(s):
  Output 0: name='last_hidden_state', shape=[1 512 384], type=0

Compiling model for CPU...
Model compiled successfully for device: CPU

Creating inference request...
Preparing input data...
Tokenized text: 4 tokens (max: 512)
First 10 token IDs: [101 7592 1010 2088 999 102 0 0 0 0]

Setting input tensors...
  Set input 'input_ids': shape=[1 512], type=1
  Set input 'attention_mask': shape=[1 512], type=1

Running inference...
Inference completed in 45.2ms

Extracting embedding...
Output shape: [1 512 384]
Output size: 196608 elements
Embedding dimension: 384
Normalized embedding dimension: 384

=== Results ===
First 10 embedding values:
  [0] = 0.012345
  [1] = -0.023456
  [2] = 0.034567
  ...

Embedding statistics:
  Mean: 0.000123
  Min:  -0.456789
  Max:  0.234567
  L2 norm: 1.000000

Text embedding example completed successfully!
```