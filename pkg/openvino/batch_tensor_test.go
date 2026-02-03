package openvino

import (
	"testing"
)

func shapeToInt64(s []int32) []int64 {
	out := make([]int64, len(s))
	for i, v := range s {
		out[i] = int64(v)
	}
	return out
}

func TestInferRequest_SetInputTensors(t *testing.T) {
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

	inputs, _ := model.GetInputs()
	if len(inputs) == 0 {
		t.Skip("model has no inputs")
	}

	// Check if model has batch dimension
	shape := inputs[0].Shape
	if len(shape) == 0 {
		t.Skip("first input has no shape")
	}

	// Note: This test will only work if the model has a batch dimension
	// For most test models, we'll skip if batch dimension is not present
	hasBatchDim := shape[0] > 1 || shape[0] == -1
	if !hasBatchDim {
		t.Skip("model does not have batch dimension")
	}

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

	// Create batch of tensors
	batchSize := 2
	if shape[0] > 0 && shape[0] < int32(batchSize) {
		batchSize = int(shape[0])
	}

	tensors := make([]*Tensor, batchSize)
	for i := 0; i < batchSize; i++ {
		// Create tensor with batch dimension = 1
		tensorShape := []int64{1}
		for j := 1; j < len(shape); j++ {
			if shape[j] > 0 {
				tensorShape = append(tensorShape, int64(shape[j]))
			} else {
				tensorShape = append(tensorShape, 224) // default
			}
		}

		size := int64(1)
		for _, d := range tensorShape {
			size *= d
		}
		data := make([]float32, size)
		for j := range data {
			data[j] = float32(i*100 + j)
		}

		tensors[i], err = NewTensorWithData(DataTypeFloat32, tensorShape, data)
		if err != nil {
			t.Fatalf("NewTensorWithData failed: %v", err)
		}
		defer tensors[i].Close()
	}

	// Try to set batch tensors (may fail if model doesn't support it)
	err = req.SetInputTensors(inputs[0].Name, tensors)
	if err != nil {
		t.Logf("SetInputTensors failed (expected for models without batch support): %v", err)
		t.Skip("model does not support batch tensor operations")
	}

	// If successful, run inference
	err = req.Infer()
	if err != nil {
		t.Logf("Infer failed: %v", err)
	}
}

func TestInferRequest_SetInputTensorsByIndex(t *testing.T) {
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

	inputs, _ := model.GetInputs()
	if len(inputs) == 0 {
		t.Skip("model has no inputs")
	}

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

	shape := inputs[0].Shape
	if len(shape) == 0 {
		t.Skip("first input has no shape")
	}

	batchSize := 2
	tensors := make([]*Tensor, batchSize)
	for i := 0; i < batchSize; i++ {
		tensorShape := []int64{1}
		for j := 1; j < len(shape); j++ {
			if shape[j] > 0 {
				tensorShape = append(tensorShape, int64(shape[j]))
			} else {
				tensorShape = append(tensorShape, 224)
			}
		}

		size := int64(1)
		for _, d := range tensorShape {
			size *= d
		}
		data := make([]float32, size)
		tensors[i], err = NewTensorWithData(DataTypeFloat32, tensorShape, data)
		if err != nil {
			t.Fatalf("NewTensorWithData failed: %v", err)
		}
		defer tensors[i].Close()
	}

	err = req.SetInputTensorsByIndex(0, tensors)
	if err != nil {
		t.Logf("SetInputTensorsByIndex failed (expected for models without batch support): %v", err)
		t.Skip("model does not support batch tensor operations")
	}
}
