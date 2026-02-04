package openvino

import (
	"testing"
)

func TestInferRequest_GetProfilingInfo(t *testing.T) {
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

	// Run inference first (profiling info may only be available after inference)
	err = req.Infer()
	if err != nil {
		t.Skipf("Infer failed: %v", err)
	}

	// Get profiling info
	profilingInfo, err := req.GetProfilingInfo()
	if err != nil {
		t.Fatalf("GetProfilingInfo failed: %v", err)
	}

	// Most plugins may not provide profiling info, so empty slice is acceptable
	if len(profilingInfo) > 0 {
		for i, info := range profilingInfo {
			if info.NodeName == "" {
				t.Errorf("ProfilingInfo[%d] has empty NodeName", i)
			}
			if info.RealTime < 0 {
				t.Errorf("ProfilingInfo[%d] has negative RealTime: %d", i, info.RealTime)
			}
			if info.CPUTime < 0 {
				t.Errorf("ProfilingInfo[%d] has negative CPUTime: %d", i, info.CPUTime)
			}
		}
	}
}
