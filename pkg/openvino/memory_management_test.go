package openvino

import (
	"testing"
)

func TestCompiledModel_ReleaseMemory(t *testing.T) {
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

	// ReleaseMemory should work without errors
	err = compiled.ReleaseMemory()
	if err != nil {
		t.Fatalf("ReleaseMemory failed: %v", err)
	}

	// Should still be able to create infer requests after releasing memory
	req, err := compiled.CreateInferRequest()
	if err != nil {
		t.Fatalf("CreateInferRequest failed after ReleaseMemory: %v", err)
	}
	req.Close()
}
