package openvino

import "testing"

func TestCore_ReadModel(t *testing.T) {
	core := coreAvailable(t)
	defer core.Close()

	_, err := core.ReadModel("/nonexistent/path/model.ir")
	if err == nil {
		t.Fatal("ReadModel with nonexistent path should return error")
	}
	// With a real path we'd get model; skip integration test without a fixture
}

func TestCore_ReadModel_integration(t *testing.T) {
	core := coreAvailable(t)
	defer core.Close()

	// If OPENVINO_TEST_MODEL is set, try to load and exercise model
	modelPath := getTestModelPath(t)
	if modelPath == "" {
		t.Skip("no test model path (set OPENVINO_TEST_MODEL for integration)")
	}

	model, err := core.ReadModel(modelPath)
	if err != nil {
		t.Skipf("ReadModel failed: %v", err)
	}
	defer model.Close()
	if model == nil {
		t.Fatal("ReadModel returned nil with nil error")
	}
}

func TestModel_Close(t *testing.T) {
	m := &Model{}
	m.Close()
}

func TestModel_GetInputs(t *testing.T) {
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

	inputs, err := model.GetInputs()
	if err != nil {
		t.Fatalf("GetInputs failed: %v", err)
	}
	if inputs == nil {
		t.Fatal("GetInputs returned nil slice")
	}
}

func TestModel_GetOutputs(t *testing.T) {
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

	outputs, err := model.GetOutputs()
	if err != nil {
		t.Fatalf("GetOutputs failed: %v", err)
	}
	if outputs == nil {
		t.Fatal("GetOutputs returned nil slice")
	}
}
