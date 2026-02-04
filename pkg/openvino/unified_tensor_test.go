package openvino

import (
	"testing"
)

func TestInferRequest_GetTensor(t *testing.T) {
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

	// Test GetTensor (unified API) - should work for inputs
	tensor, err := req.GetTensor(inputs[0].Name)
	if err != nil {
		t.Fatalf("GetTensor failed: %v", err)
	}
	if tensor == nil {
		t.Fatal("GetTensor returned nil tensor")
	}
	defer tensor.Close()

	shape, err := tensor.GetShape()
	if err != nil {
		t.Fatalf("GetShape failed: %v", err)
	}
	if len(shape) == 0 {
		t.Error("GetShape returned empty shape")
	}
}

func TestInferRequest_SetTensor(t *testing.T) {
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

	// Create tensor
	shape := shapeToInt64(inputs[0].Shape)
	size := int64(1)
	for _, d := range shape {
		size *= d
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i)
	}

	tensor, err := NewTensorWithData(inputs[0].DataType, shape, data)
	if err != nil {
		t.Fatalf("NewTensorWithData failed: %v", err)
	}
	defer tensor.Close()

	// Test SetTensor (unified API)
	err = req.SetTensor(inputs[0].Name, tensor)
	if err != nil {
		t.Fatalf("SetTensor failed: %v", err)
	}

	// Run inference to verify it works
	err = req.Infer()
	if err != nil {
		t.Fatalf("Infer failed: %v", err)
	}
}
