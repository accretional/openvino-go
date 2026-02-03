package openvino

import "testing"

func TestTensor_Close(t *testing.T) {
	tensor := &Tensor{}
	tensor.Close()
}

func TestTensor_GetDataAsFloat32(t *testing.T) {
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
	if err := req.Infer(); err != nil {
		t.Skipf("Infer failed: %v", err)
	}

	outputs, _ := model.GetOutputs()
	if len(outputs) == 0 {
		t.Skip("model has no outputs")
	}
	tensor, err := req.GetOutputTensor(outputs[0].Name)
	if err != nil {
		t.Fatalf("GetOutputTensor failed: %v", err)
	}
	defer tensor.Close()
	_, err = tensor.GetDataAsFloat32()
	if err != nil {
		t.Fatalf("GetDataAsFloat32 failed: %v", err)
	}
}

func TestTensor_GetDataAsInt64(t *testing.T) {
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
	if err := req.Infer(); err != nil {
		t.Skipf("Infer failed: %v", err)
	}

	outputs, _ := model.GetOutputs()
	if len(outputs) == 0 {
		t.Skip("model has no outputs")
	}
	tensor, err := req.GetOutputTensorByIndex(0)
	if err != nil {
		t.Fatalf("GetOutputTensorByIndex failed: %v", err)
	}
	defer tensor.Close()
	_, err = tensor.GetDataAsInt64()
	if err != nil {
		t.Fatalf("GetDataAsInt64 failed: %v", err)
	}
}

func TestTensor_GetShape(t *testing.T) {
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
	if err := req.Infer(); err != nil {
		t.Skipf("Infer failed: %v", err)
	}

	tensor, err := req.GetOutputTensorByIndex(0)
	if err != nil {
		t.Fatalf("GetOutputTensorByIndex failed: %v", err)
	}
	defer tensor.Close()
	shape, err := tensor.GetShape()
	if err != nil {
		t.Fatalf("GetShape failed: %v", err)
	}
	if shape == nil {
		t.Fatal("GetShape returned nil slice")
	}
}
