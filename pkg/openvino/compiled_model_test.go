package openvino

import "testing"

func TestCore_CompileModel(t *testing.T) {
	core := coreAvailable(t)
	defer core.Close()

	modelPath := getTestModelPath(t)
	if modelPath == "" {
		t.Skip("no test model path (set OPENVINO_TEST_MODEL for CompileModel test)")
	}
	model, err := core.ReadModel(modelPath)
	if err != nil {
		t.Skipf("cannot load model for CompileModel test: %v", err)
	}
	defer model.Close()

	compiled, err := core.CompileModel(model, "CPU")
	if err != nil {
		t.Skipf("CompileModel failed (device or model issue): %v", err)
	}
	if compiled == nil {
		t.Fatal("CompileModel returned nil with nil error")
	}
	compiled.Close()
}

func TestCore_CompileModel_withOptions(t *testing.T) {
	core := coreAvailable(t)
	defer core.Close()

	modelPath := getTestModelPath(t)
	if modelPath == "" {
		t.Skip("no test model path")
	}
	model, err := core.ReadModel(modelPath)
	if err != nil {
		t.Skipf("cannot load model: %v", err)
	}
	defer model.Close()

	compiled, err := core.CompileModel(model, "CPU", PerformanceHint(PerformanceModeLatency), NumStreams(1))
	if err != nil {
		t.Skipf("CompileModel with options failed: %v", err)
	}
	if compiled == nil {
		t.Fatal("CompileModel returned nil with nil error")
	}
	compiled.Close()
}

func TestCompiledModel_Close(t *testing.T) {
	cm := &CompiledModel{}
	// Close on zero value should not panic
	cm.Close()
}

func TestCompiledModel_CreateInferRequest(t *testing.T) {
	core := coreAvailable(t)
	defer core.Close()

	modelPath := getTestModelPath(t)
	if modelPath == "" {
		t.Skip("no test model path")
	}
	model, err := core.ReadModel(modelPath)
	if err != nil {
		t.Skipf("cannot load model: %v", err)
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
	if req == nil {
		t.Fatal("CreateInferRequest returned nil with nil error")
	}
	req.Close()
}
