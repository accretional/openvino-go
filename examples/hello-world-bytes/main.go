// Hello World Bytes example for openvino-go
// This demonstrates loading models from []byte memory buffers (both single-file like ONNX and dual-file like OpenVINO IR XML+BIN)
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/accretional/openvino-go/pkg/openvino"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: hello-world-bytes <model.onnx | model.xml>")
		fmt.Println("\nExamples:")
		fmt.Println("  1. ONNX model (single buffer):")
		fmt.Println("     hello-world-bytes model.onnx")
		fmt.Println("  2. OpenVINO IR model (XML + BIN buffers):")
		fmt.Println("     hello-world-bytes model.xml")
		os.Exit(1)
	}

	modelPath := os.Args[1]
	ext := strings.ToLower(filepath.Ext(modelPath))
	if ext != ".xml" && ext != ".onnx" {
		log.Fatalf("Model must be a .xml (OpenVINO IR) or .onnx file, got: %s", modelPath)
	}

	// Read primary model buffer from disk
	modelBytes, err := os.ReadFile(modelPath)
	if err != nil {
		log.Fatalf("Failed to read model file %s: %v", modelPath, err)
	}
	fmt.Printf("Loaded model file into memory (%d bytes)\n", len(modelBytes))

	var weightsBytes []byte
	if ext == ".xml" {
		// For OpenVINO IR (.xml), check if a matching .bin weights file exists
		binPath := strings.TrimSuffix(modelPath, filepath.Ext(modelPath)) + ".bin"
		if bData, err := os.ReadFile(binPath); err == nil {
			weightsBytes = bData
			fmt.Printf("Loaded weights file %s into memory (%d bytes)\n", binPath, len(weightsBytes))
		} else {
			fmt.Printf("No separate weights file found at %s (using self-contained XML model)\n", binPath)
		}
	} else if ext == ".onnx" {
		// For ONNX models, weights are already embedded in the single ONNX file/buffer.
		// Set weightsBytes to nil.
		weightsBytes = nil
		fmt.Println("ONNX model: single file format (weightsBuffer set to nil)")
	}

	// Step 1: Create OpenVINO Core
	fmt.Println("\nCreating OpenVINO Core...")
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

	// Step 3: Read Model from memory buffers ([]byte)
	fmt.Println("\nLoading model from []byte memory buffer...")
	model, err := core.ReadModelFromBuffer(modelBytes, weightsBytes)
	if err != nil {
		log.Fatalf("Failed to read model from buffer: %v", err)
	}
	defer model.Close()
	fmt.Println("Model successfully parsed and loaded from memory!")

	// Step 4: Inspect Model I/O Information
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

	// Step 5: Compile model
	device := "CPU"
	if len(devices) > 0 {
		device = devices[0]
	}
	fmt.Printf("Compiling model for device: %s...\n", device)

	compiledModel, err := core.CompileModel(model, device,
		openvino.PerformanceHint(openvino.PerformanceModeThroughput),
		openvino.NumStreams(4),
	)
	if err != nil {
		log.Fatalf("Failed to compile model: %v", err)
	}
	defer compiledModel.Close()

	// Step 6: Create Infer Request & Run Inference
	fmt.Println("Creating inference request...")
	request, err := compiledModel.CreateInferRequest()
	if err != nil {
		log.Fatalf("Failed to create infer request: %v", err)
	}
	defer request.Close()

	var inputShape []int64
	var inputName string
	var inputDataType openvino.DataType

	if len(inputs) > 0 {
		inputName = inputs[0].Name
		inputDataType = inputs[0].DataType
		inputShape = make([]int64, len(inputs[0].Shape))
		for i, dim := range inputs[0].Shape {
			inputShape[i] = int64(dim)
		}
	} else {
		inputShape = []int64{1, 3, 224, 224}
		inputName = "input"
		inputDataType = openvino.DataTypeFloat32
	}

	inputSize := int64(1)
	for _, dim := range inputShape {
		inputSize *= dim
	}
	inputData := make([]float32, inputSize)
	for i := range inputData {
		inputData[i] = 0.5
	}

	fmt.Println("Setting input tensor...")
	err = request.SetInputTensor(inputName, inputData, inputShape, inputDataType)
	if err != nil {
		err = request.SetInputTensorByIndex(0, inputData, inputShape, inputDataType)
		if err != nil {
			log.Fatalf("Failed to set input tensor: %v", err)
		}
	}

	fmt.Println("Running inference...")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	err = request.InferWithContext(ctx)
	if err != nil {
		log.Fatalf("Failed to run inference: %v", err)
	}

	// Step 7: Get Output
	outputTensor, err := request.GetOutputTensorByIndex(0)
	if err != nil {
		log.Fatalf("Failed to get output tensor: %v", err)
	}
	defer outputTensor.Close()

	outputData, err := outputTensor.GetDataAsFloat32()
	if err != nil {
		log.Fatalf("Failed to get output data: %v", err)
	}

	outputShape, err := outputTensor.GetShape()
	if err != nil {
		log.Fatalf("Failed to get output shape: %v", err)
	}

	fmt.Printf("\nInference completed successfully!\n")
	fmt.Printf("Output shape: %v\n", outputShape)
	fmt.Printf("Output size: %d elements\n", len(outputData))

	fmt.Println("\nFirst 10 output values:")
	for i := 0; i < len(outputData) && i < 10; i++ {
		fmt.Printf("  [%d] = %f\n", i, outputData[i])
	}

	fmt.Println("\nHello World Bytes example completed successfully")
}
