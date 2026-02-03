package openvino

import (
	"context"
	"testing"
)

func TestInferRequest_Close(t *testing.T) {
	ir := &InferRequest{}
	ir.Close()
}

func TestInferRequest_SetInputTensor(t *testing.T) {
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
	// Set input with valid shape for first input
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
}

func shapeToInt64(s []int32) []int64 {
	out := make([]int64, len(s))
	for i, v := range s {
		out[i] = int64(v)
	}
	return out
}

func TestInferRequest_Infer(t *testing.T) {
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

	err = req.Infer()
	if err != nil {
		t.Fatalf("Infer failed: %v", err)
	}
}

func TestInferRequest_InferWithContext(t *testing.T) {
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

	err = req.InferWithContext(context.Background())
	if err != nil {
		t.Fatalf("InferWithContext failed: %v", err)
	}
}

func TestInferRequest_InferWithContext_cancelled(t *testing.T) {
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

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err = req.InferWithContext(ctx)
	if err != context.Canceled {
		t.Errorf("InferWithContext with cancelled context: got %v, want context.Canceled", err)
	}
}

func TestInferRequest_StartAsync_Wait(t *testing.T) {
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

	// Start async inference
	err = req.StartAsync()
	if err != nil {
		t.Fatalf("StartAsync failed: %v", err)
	}

	// Wait for completion
	err = req.Wait()
	if err != nil {
		t.Fatalf("Wait failed: %v", err)
	}
}

func TestInferRequest_WaitFor(t *testing.T) {
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

	// Start async inference
	err = req.StartAsync()
	if err != nil {
		t.Fatalf("StartAsync failed: %v", err)
	}

	// Wait with timeout (should complete)
	completed, err := req.WaitFor(5000) // 5 seconds
	if err != nil {
		t.Fatalf("WaitFor failed: %v", err)
	}
	if !completed {
		t.Error("WaitFor returned false, expected true (inference should complete)")
	}
}

func TestInferRequest_InferAsync(t *testing.T) {
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

	// Use convenience method
	err = req.InferAsync()
	if err != nil {
		t.Fatalf("InferAsync failed: %v", err)
	}
}

func TestInferRequest_InferAsyncWithContext(t *testing.T) {
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

	// Use convenience method with context
	err = req.InferAsyncWithContext(context.Background())
	if err != nil {
		t.Fatalf("InferAsyncWithContext failed: %v", err)
	}
}

func TestInferRequest_GetInputTensor(t *testing.T) {
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

	// Set input first
	shape := inputs[0].Shape
	size := int64(1)
	for _, d := range shape {
		size *= int64(d)
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	err = req.SetInputTensor(inputs[0].Name, data, shapeToInt64(shape), inputs[0].DataType)
	if err != nil {
		t.Fatalf("SetInputTensor failed: %v", err)
	}

	// Get input tensor by name
	inputTensor, err := req.GetInputTensor(inputs[0].Name)
	if err != nil {
		t.Fatalf("GetInputTensor failed: %v", err)
	}
	defer inputTensor.Close()

	// Verify we can read the data
	retrievedData, err := inputTensor.GetDataAsFloat32()
	if err != nil {
		t.Fatalf("GetDataAsFloat32 failed: %v", err)
	}
	if len(retrievedData) != len(data) {
		t.Errorf("Retrieved data length mismatch: got %d, want %d", len(retrievedData), len(data))
	}

	// Get input tensor by index
	inputTensor2, err := req.GetInputTensorByIndex(0)
	if err != nil {
		t.Fatalf("GetInputTensorByIndex failed: %v", err)
	}
	defer inputTensor2.Close()

	retrievedData2, err := inputTensor2.GetDataAsFloat32()
	if err != nil {
		t.Fatalf("GetDataAsFloat32 failed: %v", err)
	}
	if len(retrievedData2) != len(data) {
		t.Errorf("Retrieved data length mismatch: got %d, want %d", len(retrievedData2), len(data))
	}
}
