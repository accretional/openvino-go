package openvino

import (
	"sync"
	"testing"
	"time"
)

func TestInferRequest_SetCallback(t *testing.T) {
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

	var wg sync.WaitGroup
	var callbackErr error
	var callbackCalled bool

	wg.Add(1)
	err = req.SetCallback(func(err error) {
		callbackErr = err
		callbackCalled = true
		wg.Done()
	})
	if err != nil {
		t.Fatalf("SetCallback failed: %v", err)
	}

	err = req.StartAsync()
	if err != nil {
		t.Fatalf("StartAsync failed: %v", err)
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		if !callbackCalled {
			t.Error("Callback was not called")
		}
		if callbackErr != nil {
			t.Logf("Callback received error: %v", callbackErr)
		}
	case <-time.After(5 * time.Second):
		t.Error("Callback timeout")
	}

	err = req.SetCallback(nil)
	if err != nil {
		t.Fatalf("SetCallback(nil) failed: %v", err)
	}
}
