package bench_test

import (
	"fmt"
	"os"
	"testing"

	openvino "github.com/accretional/openvino-go/pkg/openvino"
)

// Default model path relative to the benchmarks directory.
const defaultModelPath = "../models/all-MiniLM-L6-v2/onnx/model.onnx"

// Default sequence length to substitute for dynamic dimensions (-1).
const defaultSeqLen = 128

// modelInputInfo holds the resolved input metadata for the benchmark model.
type modelInputInfo struct {
	Name     string
	Shape    []int64
	DataType openvino.DataType
	Size     int64 // total number of elements
}

// modelOutputInfo holds the resolved output metadata.
type modelOutputInfo struct {
	Name     string
	Shape    []int64
	DataType openvino.DataType
	Size     int64
}

var (
	benchModelPath string
	benchInputs    []modelInputInfo
	benchOutputs   []modelOutputInfo
	// sharedCore is a shared OpenVINO Core for all benchmarks.
	// Creating/destroying Core repeatedly causes stability issues in OpenVINO.
	sharedCore *openvino.Core
)

func TestMain(m *testing.M) {
	benchModelPath = getModelPath()

	if _, err := os.Stat(benchModelPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "benchmark model not found at %s — skipping all benchmarks\n", benchModelPath)
		os.Exit(0)
	}

	// Use OpenVINO to introspect model I/O so both runtimes use the same shapes.
	core, err := openvino.NewCore()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create OpenVINO core: %v\n", err)
		os.Exit(1)
	}
	model, err := core.ReadModel(benchModelPath)
	if err != nil {
		core.Close()
		fmt.Fprintf(os.Stderr, "failed to read model %s: %v\n", benchModelPath, err)
		os.Exit(1)
	}

	inputs, _ := model.GetInputs()
	for _, inp := range inputs {
		shape := resolveShape(inp.Shape)
		benchInputs = append(benchInputs, modelInputInfo{
			Name:     inp.Name,
			Shape:    shape,
			DataType: inp.DataType,
			Size:     shapeSize(shape),
		})
	}

	outputs, _ := model.GetOutputs()
	for _, out := range outputs {
		shape := resolveShape(out.Shape)
		benchOutputs = append(benchOutputs, modelOutputInfo{
			Name:     out.Name,
			Shape:    shape,
			DataType: out.DataType,
			Size:     shapeSize(shape),
		})
	}

	model.Close()
	// Keep the core alive for benchmarks - destroying it causes issues
	sharedCore = core

	code := m.Run()
	
	// Clean up after all benchmarks
	if sharedCore != nil {
		sharedCore.Close()
	}
	
	os.Exit(code)
}

func getModelPath() string {
	// Check multiple environment variables for flexibility
	if p := os.Getenv("BENCH_MODEL"); p != "" {
		return p
	}
	if p := os.Getenv("MODEL_PATH"); p != "" {
		return p
	}
	return defaultModelPath
}

// resolveShape converts int32 port shapes to int64 and replaces dynamic
// dimensions (-1) with sensible defaults.
func resolveShape(s []int32) []int64 {
	out := make([]int64, len(s))
	for i, d := range s {
		if d <= 0 {
			switch i {
			case 0:
				out[i] = 1 // batch dimension
			default:
				out[i] = defaultSeqLen
			}
		} else {
			out[i] = int64(d)
		}
	}
	return out
}

func shapeSize(s []int64) int64 {
	n := int64(1)
	for _, d := range s {
		n *= d
	}
	return n
}

// makeDummyInt64 creates a zero-filled int64 slice of the given size.
func makeDummyInt64(size int64) []int64 {
	return make([]int64, size)
}

// makeDummyFloat32 creates a zero-filled float32 slice of the given size.
func makeDummyFloat32(size int64) []float32 {
	return make([]float32, size)
}
