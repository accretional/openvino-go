package openvino

import (
	"testing"
)

func TestInferRequest_SetOutputTensor(t *testing.T) {
	core := coreAvailable(t)
	defer core.Close()
	modelPath := getTestModelPath(t)
	if modelPath == "" {
		t.Skip("no test model path")
	}
	model, err := core.ReadModel(modelPath)
	if err != nil {
		t.Skipf("ReadModel failed: %v", err)
	}
	defer model.Close()

	compiled, err := core.CompileModel(model, "CPU")
	if err != nil {
		t.Skipf("CompileModel failed: %v", err)
	}
	defer compiled.Close()

	req, err := compiled.CreateInferRequest()
	if err != nil {
		t.Fatalf("CreateInferRequest failed: %v", err)
	}
	defer req.Close()

	inputs, _ := model.GetInputs()
	if len(inputs) == 0 {
		t.Skip("model has no inputs")
	}

	// Set input
	shape := inputs[0].Shape
	if len(shape) == 0 {
		t.Skip("first input has no shape")
	}
	size := int64(1)
	for _, d := range shape {
		size *= int64(d)
	}
	data := make([]float32, size)
	err = req.SetInputTensor(inputs[0].Name, data, shapeToInt64(shape), inputs[0].DataType)
	if err != nil {
		t.Fatalf("SetInputTensor failed: %v", err)
	}

	// Pre-allocate output tensor
	outputs, _ := model.GetOutputs()
	if len(outputs) == 0 {
		t.Skip("model has no outputs")
	}

	outputShape := outputs[0].Shape
	if len(outputShape) == 0 {
		t.Skip("first output has no shape")
	}

	outputTensor, err := NewTensor(outputs[0].DataType, shapeToInt64(outputShape))
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
	defer outputTensor.Close()

	err = req.SetOutputTensor(outputs[0].Name, outputTensor)
	if err != nil {
		t.Fatalf("SetOutputTensor failed: %v", err)
	}

	// Run inference
	err = req.Infer()
	if err != nil {
		t.Fatalf("Infer failed: %v", err)
	}

	// Verify output tensor has data
	result, err := outputTensor.GetDataAsFloat32()
	if err != nil {
		t.Fatalf("GetDataAsFloat32 failed: %v", err)
	}
	if len(result) == 0 {
		t.Error("Output tensor has no data")
	}
}

func TestInferRequest_SetOutputTensorByIndex(t *testing.T) {
	core := coreAvailable(t)
	defer core.Close()
	modelPath := getTestModelPath(t)
	if modelPath == "" {
		t.Skip("no test model path")
	}
	model, err := core.ReadModel(modelPath)
	if err != nil {
		t.Skipf("ReadModel failed: %v", err)
	}
	defer model.Close()

	compiled, err := core.CompileModel(model, "CPU")
	if err != nil {
		t.Skipf("CompileModel failed: %v", err)
	}
	defer compiled.Close()

	req, err := compiled.CreateInferRequest()
	if err != nil {
		t.Fatalf("CreateInferRequest failed: %v", err)
	}
	defer req.Close()

	inputs, _ := model.GetInputs()
	if len(inputs) > 0 {
		shape := inputs[0].Shape
		size := int64(1)
		for _, d := range shape {
			size *= int64(d)
		}
		data := make([]float32, size)
		_ = req.SetInputTensor(inputs[0].Name, data, shapeToInt64(shape), inputs[0].DataType)
	}

	outputs, _ := model.GetOutputs()
	if len(outputs) == 0 {
		t.Skip("model has no outputs")
	}

	outputShape := outputs[0].Shape
	outputTensor, err := NewTensor(outputs[0].DataType, shapeToInt64(outputShape))
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
	defer outputTensor.Close()

	err = req.SetOutputTensorByIndex(0, outputTensor)
	if err != nil {
		t.Fatalf("SetOutputTensorByIndex failed: %v", err)
	}

	err = req.Infer()
	if err != nil {
		t.Fatalf("Infer failed: %v", err)
	}

	_, err = outputTensor.GetDataAsFloat32()
	if err != nil {
		t.Fatalf("GetDataAsFloat32 failed: %v", err)
	}
}
