package openvino

import (
	"testing"
)

func TestInferRequest_QueryState(t *testing.T) {
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

	// Query state (will return empty slice for non-stateful models)
	states, err := req.QueryState()
	if err != nil {
		t.Fatalf("QueryState failed: %v", err)
	}

	// Most test models won't have variable states, so empty slice is expected
	if len(states) > 0 {
		// If we have states, test the API
		state := states[0]
		defer state.Close()

		name, err := state.GetName()
		if err != nil {
			t.Fatalf("GetName failed: %v", err)
		}
		if name == "" {
			t.Error("GetName returned empty string")
		}

		stateTensor, err := state.GetState()
		if err != nil {
			t.Fatalf("GetState failed: %v", err)
		}
		if stateTensor != nil {
			defer stateTensor.Close()
		}
	}
}

func TestInferRequest_ResetState(t *testing.T) {
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

	// ResetState should work even for non-stateful models (no-op)
	err = req.ResetState()
	if err != nil {
		t.Fatalf("ResetState failed: %v", err)
	}
}

func TestVariableState_API(t *testing.T) {
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

	states, err := req.QueryState()
	if err != nil {
		t.Fatalf("QueryState failed: %v", err)
	}

	if len(states) == 0 {
		t.Skip("model has no variable states")
	}

	state := states[0]
	defer state.Close()

	// Test GetName
	name, err := state.GetName()
	if err != nil {
		t.Fatalf("GetName failed: %v", err)
	}
	if name == "" {
		t.Error("GetName returned empty string")
	}

	// Test GetState
	stateTensor, err := state.GetState()
	if err != nil {
		t.Fatalf("GetState failed: %v", err)
	}
	if stateTensor == nil {
		t.Skip("GetState returned nil tensor")
	}
	defer stateTensor.Close()

	// Test SetState with a new tensor
	shape, err := stateTensor.GetShape()
	if err != nil {
		t.Fatalf("GetShape failed: %v", err)
	}

	dataType, err := stateTensor.GetElementType()
	if err != nil {
		t.Fatalf("GetElementType failed: %v", err)
	}

	newTensor, err := NewTensor(dataType, shapeToInt64(shape))
	if err != nil {
		t.Fatalf("NewTensor failed: %v", err)
	}
	defer newTensor.Close()

	err = state.SetState(newTensor)
	if err != nil {
		t.Fatalf("SetState failed: %v", err)
	}

	// Test Reset
	err = state.Reset()
	if err != nil {
		t.Fatalf("Reset failed: %v", err)
	}
}
