// Hello World example for openvino-go
// This demonstrates the basic inference pipeline
package main

import (
	"fmt"
	"log"
	"os"
	"strings"

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

	// Step 4: Compile model for CPU
	fmt.Println("Compiling model for CPU...")
	device := "CPU"
	if len(devices) > 0 {
		// Use first available device (could be CPU, GPU, etc.)
		device = devices[0]
	}
	compiledModel, err := core.CompileModel(model, device)
	if err != nil {
		log.Fatalf("Failed to compile model: %v", err)
	}
	defer compiledModel.Close()
	fmt.Printf("Model compiled successfully for device: %s\n", device)

	// Step 5: Create inference request
	fmt.Println("Creating inference request...")
	request, err := compiledModel.CreateInferRequest()
	if err != nil {
		log.Fatalf("Failed to create infer request: %v", err)
	}
	defer request.Close()

	// Step 6: Prepare input data
	// Note: This is a placeholder. In a real scenario, you would:
	// - Know the model's input shape and type
	// - Preprocess your data accordingly
	// - Set the appropriate input tensor
	fmt.Println("Preparing input data...")
	
	// Example: Create dummy input data (float32, shape [1, 3, 224, 224] for image classification)
	// In practice, you would load and preprocess an actual image
	inputShape := []int64{1, 3, 224, 224}
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
	// Try by index first (most models use index 0 for the first input)
	fmt.Println("Setting input tensor...")
	err = request.SetInputTensorByIndex(0, inputData, inputShape, openvino.DataTypeFloat32)
	if err != nil {
		// If that fails, try by name (common names: "input", "data", "image", etc.)
		fmt.Printf("Setting by index failed, trying by name 'input'...\n")
		err = request.SetInputTensor("input", inputData, inputShape, openvino.DataTypeFloat32)
		if err != nil {
			log.Fatalf("Failed to set input tensor: %v\nNote: You may need to adjust the input name or shape based on your model", err)
		}
	}

	// Step 8: Run inference
	fmt.Println("Running inference...")
	err = request.Infer()
	if err != nil {
		log.Fatalf("Failed to run inference: %v", err)
	}

	// Step 9: Get output tensor
	fmt.Println("Getting output tensor...")
	outputTensor, err := request.GetOutputTensorByIndex(0)
	if err != nil {
		// Try by name
		outputTensor, err = request.GetOutputTensor("output")
		if err != nil {
			log.Fatalf("Failed to get output tensor: %v\nNote: You may need to adjust the output name based on your model", err)
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
