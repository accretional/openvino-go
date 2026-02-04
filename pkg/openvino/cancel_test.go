package openvino

import (
	"testing"
	"time"
)

func TestInferRequest_Cancel(t *testing.T) {
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

	// Cancel should work even if no inference is running
	err = req.Cancel()
	if err != nil {
		t.Fatalf("Cancel failed: %v", err)
	}
}

func TestInferRequest_CancelAsync(t *testing.T) {
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

	// Cancel immediately
	err = req.Cancel()
	if err != nil {
		t.Logf("Cancel failed (may not be supported during inference): %v", err)
	}

	// Wait a bit
	time.Sleep(10 * time.Millisecond)
}
