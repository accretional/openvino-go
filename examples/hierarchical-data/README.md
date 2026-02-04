# Hierarchical Data Processing Example

This example demonstrates how to use OpenVINO Go bindings for processing hierarchical data structures.

## Features Demonstrated

1. **Multi-level Processing**: Process data at multiple hierarchy levels (e.g., sentences → paragraphs → documents)
2. **Tensor Creation**: Create tensors for intermediate results and state management
3. **Zero-copy Output**: Pre-allocate output tensors to avoid memory copies
4. **Dynamic Shapes**: Handle variable-sized hierarchies with dynamic tensor shapes

## Usage

```bash
go run main.go <model_path>
```

Example:
```bash
go run main.go model.xml
```

## What We Have Here

### Example 1: Multi-level Processing
Demonstrates processing hierarchical data by:
- Processing each level (sentences) individually
- Aggregating results to higher levels (paragraphs)
- Using tensor creation for intermediate embeddings

### Example 2: Tensor Creation for Intermediate Results
Shows how to:
- Create empty tensors for intermediate processing
- Reshape tensors dynamically for different hierarchy levels
- Manage tensor lifecycle

### Example 3: Zero-copy Output Pre-allocation
Demonstrates:
- Pre-allocating output tensors before inference
- Avoiding memory copies by writing directly to pre-allocated memory
- Improving performance for repeated inference

### Example 4: Dynamic Shape Handling
Shows:
- Processing variable-sized inputs
- Handling dynamic shapes for different hierarchy depths
- Reshaping tensors when supported

## Key Concepts

### Hierarchical Data Structures

For hierarchical data (trees, multi-level graphs, nested structures), you typically:

1. **Process bottom-up**: Start with leaf nodes (e.g., words/tokens)
2. **Aggregate**: Combine results at each level (e.g., sentences → paragraphs)
3. **Maintain state**: Use VariableState for stateful models, or manual state management in Go
4. **Handle variable sizes**: Use dynamic shapes or multiple inference passes

### Tensor Creation

Use `NewTensor()` and `NewTensorWithData()` to:
- Create tensors for intermediate results
- Pre-allocate output tensors
- Manage state between inference passes

### State Management

- **For stateful models** (RNNs/LSTMs): Use `VariableState` API
- **For general hierarchical data**: Use multiple inference passes with manual state management in Go

## Notes

- This example uses a generic model for demonstration
- In practice, you would use models specifically designed for your hierarchical data structure
- Dynamic shape support depends on the model and OpenVINO device capabilities
- Batch tensor operations require models with fixed batch dimensions
