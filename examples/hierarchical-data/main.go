package main

import (
	"fmt"
	"log"
	"os"

	"github.com/accretional/openvino-go"
)

// This example demonstrates hierarchical data processing using OpenVINO Go bindings.
// It shows how to process data at multiple levels (e.g., sentences -> paragraphs -> documents)
// using tensor creation, multiple inference passes, and state management.

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <model_path>")
		fmt.Println("Example: go run main.go model.xml")
		os.Exit(1)
	}

	modelPath := os.Args[1]

	// Initialize OpenVINO
	core, err := openvino.NewCore()
	if err != nil {
		log.Fatalf("Failed to create core: %v", err)
	}
	defer core.Close()

	// Load model
	model, err := core.ReadModel(modelPath)
	if err != nil {
		log.Fatalf("Failed to read model: %v", err)
	}
	defer model.Close()

	// Compile model
	compiled, err := core.CompileModel(model, "CPU")
	if err != nil {
		log.Fatalf("Failed to compile model: %v", err)
	}
	defer compiled.Close()

	// Get model I/O information
	inputs, _ := model.GetInputs()
	outputs, _ := model.GetOutputs()

	if len(inputs) == 0 || len(outputs) == 0 {
		log.Fatal("Model must have at least one input and one output")
	}

	inputInfo := inputs[0]
	outputInfo := outputs[0]

	fmt.Printf("Model Input: %s, Shape: %v, Type: %v\n",
		inputInfo.Name, inputInfo.Shape, inputInfo.DataType)
	fmt.Printf("Model Output: %s, Shape: %v, Type: %v\n",
		outputInfo.Name, outputInfo.Shape, outputInfo.DataType)

	// Example 1: Process hierarchical data with multiple inference passes
	fmt.Println("\n=== Example 1: Multi-level Processing ===")
	processMultiLevel(core, compiled, inputInfo, outputInfo)

	// Example 2: Use tensor creation for intermediate results
	fmt.Println("\n=== Example 2: Tensor Creation for Intermediate Results ===")
	processWithIntermediateTensors(compiled, inputInfo, outputInfo)

	// Example 3: Pre-allocate output tensors for zero-copy
	fmt.Println("\n=== Example 3: Zero-copy Output Pre-allocation ===")
	processWithPreallocatedOutput(compiled, inputInfo, outputInfo)

	// Example 4: Dynamic shape handling
	fmt.Println("\n=== Example 4: Dynamic Shape Handling ===")
	processDynamicShapes(compiled, inputInfo, outputInfo)
}

// processMultiLevel demonstrates processing data at multiple hierarchy levels
func processMultiLevel(core *openvino.Core, compiled *openvino.CompiledModel,
	inputInfo, outputInfo openvino.PortInfo) {

	// Simulate hierarchical data: 3 sentences -> 1 paragraph
	sentences := [][]float32{
		{1.0, 2.0, 3.0, 4.0},
		{5.0, 6.0, 7.0, 8.0},
		{9.0, 10.0, 11.0, 12.0},
	}

	req, err := compiled.CreateInferRequest()
	if err != nil {
		log.Fatalf("Failed to create infer request: %v", err)
	}
	defer req.Close()

	// Process each sentence level
	sentenceEmbeddings := make([][]float32, len(sentences))
	for i, sentence := range sentences {
		// Create tensor for sentence
		sentenceTensor, err := openvino.NewTensorWithData(
			openvino.DataTypeFloat32,
			[]int64{1, int64(len(sentence))},
			sentence,
		)
		if err != nil {
			log.Fatalf("Failed to create sentence tensor: %v", err)
		}
		defer sentenceTensor.Close()

		// Set input (adjust shape if needed)
		inputShape := shapeToInt64(inputInfo.Shape)
		if len(inputShape) > 0 && inputShape[1] != int64(len(sentence)) {
			// Reshape tensor to match model input
			err = sentenceTensor.SetShape(inputShape)
			if err != nil {
				log.Printf("Warning: SetShape failed: %v", err)
			}
		}

		// For demonstration, we'll use SetInputTensor directly
		// In real scenarios, you might need to reshape or pad
		err = req.SetInputTensor(inputInfo.Name, sentence, inputShape, inputInfo.DataType)
		if err != nil {
			log.Printf("SetInputTensor failed: %v", err)
			continue
		}

		// Run inference
		err = req.Infer()
		if err != nil {
			log.Printf("Infer failed: %v", err)
			continue
		}

		// Get output
		outputTensor, err := req.GetOutputTensor(outputInfo.Name)
		if err != nil {
			log.Printf("GetOutputTensor failed: %v", err)
			continue
		}
		defer outputTensor.Close()

		embedding, err := outputTensor.GetDataAsFloat32()
		if err != nil {
			log.Printf("GetDataAsFloat32 failed: %v", err)
			continue
		}

		sentenceEmbeddings[i] = embedding
		fmt.Printf("  Sentence %d embedding length: %d\n", i+1, len(embedding))
	}

	// Aggregate to paragraph level (simple average for demonstration)
	if len(sentenceEmbeddings) > 0 && len(sentenceEmbeddings[0]) > 0 {
		paragraphEmbedding := make([]float32, len(sentenceEmbeddings[0]))
		for i := range paragraphEmbedding {
			for _, emb := range sentenceEmbeddings {
				if i < len(emb) {
					paragraphEmbedding[i] += emb[i]
				}
			}
			paragraphEmbedding[i] /= float32(len(sentenceEmbeddings))
		}
		fmt.Printf("  Paragraph embedding length: %d\n", len(paragraphEmbedding))
	}
}

// processWithIntermediateTensors demonstrates using tensor creation for intermediate results
func processWithIntermediateTensors(compiled *openvino.CompiledModel,
	inputInfo, outputInfo openvino.PortInfo) {

	req, err := compiled.CreateInferRequest()
	if err != nil {
		log.Fatalf("Failed to create infer request: %v", err)
	}
	defer req.Close()

	// Create intermediate tensor for processing
	intermediateShape := []int64{1, 128}
	intermediateTensor, err := openvino.NewTensor(openvino.DataTypeFloat32, intermediateShape)
	if err != nil {
		log.Fatalf("Failed to create intermediate tensor: %v", err)
	}
	defer intermediateTensor.Close()

	fmt.Printf("  Created intermediate tensor with shape: %v\n", intermediateShape)

	// Reshape for different hierarchy levels
	newShape := []int64{1, 256}
	err = intermediateTensor.SetShape(newShape)
	if err != nil {
		log.Printf("  Warning: SetShape failed (may not be supported): %v", err)
	} else {
		fmt.Printf("  Reshaped tensor to: %v\n", newShape)
	}
}

// processWithPreallocatedOutput demonstrates zero-copy output pre-allocation
func processWithPreallocatedOutput(compiled *openvino.CompiledModel,
	inputInfo, outputInfo openvino.PortInfo) {

	req, err := compiled.CreateInferRequest()
	if err != nil {
		log.Fatalf("Failed to create infer request: %v", err)
	}
	defer req.Close()

	// Pre-allocate output tensor
	outputShape := shapeToInt64(outputInfo.Shape)
	outputTensor, err := openvino.NewTensor(outputInfo.DataType, outputShape)
	if err != nil {
		log.Fatalf("Failed to create output tensor: %v", err)
	}
	defer outputTensor.Close()

	// Set pre-allocated output
	err = req.SetOutputTensor(outputInfo.Name, outputTensor)
	if err != nil {
		log.Printf("  Warning: SetOutputTensor failed: %v", err)
		return
	}

	fmt.Printf("  Pre-allocated output tensor with shape: %v\n", outputShape)

	// Set input
	inputShape := shapeToInt64(inputInfo.Shape)
	size := int64(1)
	for _, d := range inputShape {
		size *= d
	}
	inputData := make([]float32, size)
	err = req.SetInputTensor(inputInfo.Name, inputData, inputShape, inputInfo.DataType)
	if err != nil {
		log.Printf("  SetInputTensor failed: %v", err)
		return
	}

	// Run inference (results go directly to pre-allocated tensor)
	err = req.Infer()
	if err != nil {
		log.Printf("  Infer failed: %v", err)
		return
	}

	// Results are already in outputTensor, no copy needed
	result, err := outputTensor.GetDataAsFloat32()
	if err != nil {
		log.Printf("  GetDataAsFloat32 failed: %v", err)
		return
	}

	fmt.Printf("  Got results directly from pre-allocated tensor: %d elements\n", len(result))
}

// processDynamicShapes demonstrates handling dynamic shapes for variable-sized hierarchies
func processDynamicShapes(compiled *openvino.CompiledModel,
	inputInfo, outputInfo openvino.PortInfo) {

	req, err := compiled.CreateInferRequest()
	if err != nil {
		log.Fatalf("Failed to create infer request: %v", err)
	}
	defer req.Close()

	// Process different sized inputs (simulating variable hierarchy depths)
	sizes := []int{64, 128, 256}
	for _, size := range sizes {
		// Create tensor with dynamic size
		data := make([]float32, size)
		for i := range data {
			data[i] = float32(i)
		}

		tensor, err := openvino.NewTensorWithData(
			openvino.DataTypeFloat32,
			[]int64{1, int64(size)},
			data,
		)
		if err != nil {
			log.Printf("  Failed to create tensor for size %d: %v", size, err)
			continue
		}
		defer tensor.Close()

		shape, err := tensor.GetShape()
		if err != nil {
			log.Printf("  Failed to get shape: %v", err)
			continue
		}

		fmt.Printf("  Processed tensor with dynamic shape: %v\n", shape)

		// Try to reshape if model supports it
		newShape := []int64{1, int64(size * 2)}
		err = tensor.SetShape(newShape)
		if err != nil {
			// Expected for most models - dynamic reshape may not be supported
			fmt.Printf("  Note: Dynamic reshape not supported (expected)\n")
		} else {
			fmt.Printf("  Reshaped to: %v\n", newShape)
		}
	}
}

func shapeToInt64(s []int32) []int64 {
	out := make([]int64, len(s))
	for i, v := range s {
		out[i] = int64(v)
	}
	return out
}
