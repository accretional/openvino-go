// Hello World example for openvino-go
// This demonstrates the basic inference pipeline
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/accretional/openvino-go/pkg/openvino"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: hello-world <model.xml>")
		fmt.Println("\nNote: Provide the .xml file; the .bin file is loaded automatically.")
		os.Exit(1)
	}

	modelPath := os.Args[1]
	if !strings.HasSuffix(modelPath, ".xml") && !strings.HasSuffix(modelPath, ".onnx") {
		log.Fatalf("Model must be a .xml (OpenVINO IR) or .onnx file, got: %s", modelPath)
	}

	// Step 1: Create OpenVINO Core
	fmt.Println("Creating OpenVINO Core...")
	core, err := openvino.NewCore()
	if err != nil {
		log.Fatalf("Failed to create core: %v", err)
	}
	defer core.Close()

	// Step 2: Get available devices
	fmt.Println("Checking available devices...")
	devices, err := core.GetAvailableDevices()
	if err != nil {
		log.Fatalf("Failed to get devices: %v", err)
	}
	fmt.Printf("Available devices: %v\n", devices)

	// Step 3: Load model
	fmt.Printf("Loading model from: %s\n", modelPath)
	model, err := core.ReadModel(modelPath)
	if err != nil {
		log.Fatalf("Failed to read model: %v", err)
	}
	defer model.Close()

	// Step 3.5: Get model I/O information (NEW in Phase 2)
	fmt.Println("\n=== Model I/O Information ===")
	inputs, err := model.GetInputs()
	if err != nil {
		log.Printf("Warning: Failed to get input info: %v", err)
	} else {
		fmt.Printf("Model has %d input(s):\n", len(inputs))
		for i, input := range inputs {
			fmt.Printf("  Input %d: name='%s', shape=%v, type=%d\n", i, input.Name, input.Shape, input.DataType)
		}
	}

	outputs, err := model.GetOutputs()
	if err != nil {
		log.Printf("Warning: Failed to get output info: %v", err)
	} else {
		fmt.Printf("Model has %d output(s):\n", len(outputs))
		for i, output := range outputs {
			fmt.Printf("  Output %d: name='%s', shape=%v, type=%d\n", i, output.Name, output.Shape, output.DataType)
		}
	}
	fmt.Println()

	// Step 4: Compile model for CPU with optimizations (NEW in Phase 2)
	fmt.Println("Compiling model for CPU with performance optimizations...")
	device := "CPU"
	if len(devices) > 0 {
		// Use first available device (could be CPU, GPU, etc.)
		device = devices[0]
	}
	
	// Use property configuration options (NEW in Phase 2)
	compiledModel, err := core.CompileModel(model, device,
		openvino.PerformanceHint(openvino.PerformanceModeThroughput),
		openvino.NumStreams(4),
	)
	if err != nil {
		log.Fatalf("Failed to compile model: %v", err)
	}
	defer compiledModel.Close()
	fmt.Printf("Model compiled successfully for device: %s with optimizations\n", device)

	// Step 5: Create inference request
	fmt.Println("Creating inference request...")
	request, err := compiledModel.CreateInferRequest()
	if err != nil {
		log.Fatalf("Failed to create infer request: %v", err)
	}
	defer request.Close()

	// Step 6: Prepare input data using model I/O information (NEW in Phase 2)
	fmt.Println("Preparing input data...")
	
	var inputShape []int64
	var inputName string
	var inputDataType openvino.DataType
	
	// Use model I/O information if available
	if len(inputs) > 0 {
		inputName = inputs[0].Name
		inputDataType = inputs[0].DataType
		// Convert []int32 to []int64
		inputShape = make([]int64, len(inputs[0].Shape))
		for i, dim := range inputs[0].Shape {
			inputShape[i] = int64(dim)
		}
		fmt.Printf("Using model input info: name='%s', shape=%v, type=%d\n", inputName, inputShape, inputDataType)
	} else {
		// Fallback to default shape if I/O info not available
		inputShape = []int64{1, 3, 224, 224}
		inputName = "input"
		inputDataType = openvino.DataTypeFloat32
		fmt.Printf("Using default input shape: %v\n", inputShape)
	}
	
	inputSize := int64(1)
	for _, dim := range inputShape {
		inputSize *= dim
	}
	inputData := make([]float32, inputSize)
	// Initialize with dummy data (in practice, this would be preprocessed image data)
	for i := range inputData {
		inputData[i] = 0.5 // Dummy value
	}

	// Step 7: Set input tensor
	fmt.Println("Setting input tensor...")
	// Try by name first (using model I/O info), then by index
	err = request.SetInputTensor(inputName, inputData, inputShape, inputDataType)
	if err != nil {
		// If that fails, try by index
		fmt.Printf("Setting by name failed, trying by index 0...\n")
		err = request.SetInputTensorByIndex(0, inputData, inputShape, inputDataType)
		if err != nil {
			log.Fatalf("Failed to set input tensor: %v", err)
		}
	}

	// Step 8: Run inference with context support (NEW in Phase 2)
	fmt.Println("Running inference...")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	err = request.InferWithContext(ctx)
	if err != nil {
		if err == context.DeadlineExceeded {
			log.Fatalf("Inference timed out after 30 seconds")
		}
		log.Fatalf("Failed to run inference: %v", err)
	}

	// Step 9: Get output tensor using model I/O information (NEW in Phase 2)
	fmt.Println("Getting output tensor...")
	var outputTensor *openvino.Tensor
	if len(outputs) > 0 {
		// Use model I/O info to get output by name
		outputTensor, err = request.GetOutputTensor(outputs[0].Name)
		if err != nil {
			// Fallback to index
			outputTensor, err = request.GetOutputTensorByIndex(0)
			if err != nil {
				log.Fatalf("Failed to get output tensor: %v", err)
			}
		}
	} else {
		// Fallback to index if I/O info not available
		outputTensor, err = request.GetOutputTensorByIndex(0)
		if err != nil {
			// Try by common name
			outputTensor, err = request.GetOutputTensor("output")
			if err != nil {
				log.Fatalf("Failed to get output tensor: %v", err)
			}
		}
	}
	defer outputTensor.Close()

	// Step 10: Extract results
	outputData, err := outputTensor.GetDataAsFloat32()
	if err != nil {
		log.Fatalf("Failed to get output data: %v", err)
	}

	outputShape, err := outputTensor.GetShape()
	if err != nil {
		log.Fatalf("Failed to get output shape: %v", err)
	}

	fmt.Printf("Inference completed successfully!\n")
	fmt.Printf("Output shape: %v\n", outputShape)
	fmt.Printf("Output size: %d elements\n", len(outputData))
	
	// Print first few values (for classification, this might be class probabilities)
	fmt.Println("First 10 output values:")
	for i := 0; i < len(outputData) && i < 10; i++ {
		fmt.Printf("  [%d] = %f\n", i, outputData[i])
	}

	// For classification models, find the class with highest probability
	if len(outputData) > 0 {
		maxIdx := 0
		maxVal := outputData[0]
		for i, val := range outputData {
			if val > maxVal {
				maxVal = val
				maxIdx = i
			}
		}
		fmt.Printf("\nPredicted class index: %d (confidence: %f)\n", maxIdx, maxVal)
	}

	fmt.Println("\nHello World example completed successfully!")
}
