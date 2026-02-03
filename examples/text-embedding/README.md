# Text Embedding Example

This example demonstrates how to use OpenVINO for text embedding inference with transformer models (BERT, sentence-transformers, etc.).

## Prerequisites

- OpenVINO Runtime installed and configured
- A text embedding model (e.g., sentence-transformers model converted to OpenVINO IR or ONNX)

### Getting a Model

You can download and convert a sentence-transformers model:

```bash
# Using Python with openvino and sentence-transformers
python3 << EOF
from sentence_transformers import SentenceTransformer
import openvino as ov

# Load a sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert to OpenVINO IR
ov_model = ov.convert_model(model, input=[1, 512])
ov.save_model(ov_model, "all-MiniLM-L6-v2.xml")
EOF
```

Or download a pre-converted model from the [OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo).

## Usage

```bash
cd /path/to/openvino-go
go run examples/text-embedding/main.go <model.xml|model.onnx> "<text>"
```

### Example

```bash
go run examples/text-embedding/main.go model.xml "Hello, world! This is a test."
```

## Input Format

The example expects:
- **Model path**: Path to an OpenVINO IR (`.xml`) or ONNX (`.onnx`) model file
- **Text**: The text string to embed (can contain spaces, use quotes)

## Model Requirements

The model should:
- Accept text token IDs as input (typically `input_ids`)
- Optionally accept `attention_mask` and `token_type_ids`
- Output embeddings (typically shape `[batch, sequence, embedding_dim]` or `[batch, embedding_dim]`)

Common compatible models:
- `all-MiniLM-L6-v2` (384 dimensions)
- `all-mpnet-base-v2` (768 dimensions)
- `paraphrase-MiniLM-L6-v2` (384 dimensions)
- Any BERT-based sentence transformer model

## Tokenization Note

⚠️ **Important**: This example uses a simplified tokenization approach for demonstration purposes. 

For production use, you should:
1. Use the model's actual tokenizer (e.g., HuggingFace tokenizer)
2. Load the tokenizer configuration from the model directory
3. Use proper subword tokenization (WordPiece, BPE, etc.)

The example includes a simple hash-based tokenization that works for demonstration but may not produce accurate results with real models.

## Output

The example outputs:
- Model I/O information
- Tokenization details
- Inference timing
- Embedding vector (first 10 values)
- Embedding statistics (mean, min, max, L2 norm)

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

## Next Steps

To use this in production:
1. Integrate a proper tokenizer (e.g., `github.com/sugarme/tokenizer`)
2. Load tokenizer configuration from model directory
3. Handle batch processing for multiple texts
4. Add caching for compiled models
5. Implement async inference for better throughput
