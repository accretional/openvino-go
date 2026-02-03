package openvino

import (
	"os"
	"testing"
)

func shapeToInt64(s []int32) []int64 {
	out := make([]int64, len(s))
	for i, v := range s {
		out[i] = int64(v)
	}
	return out
}

func coreAvailable(t *testing.T) *Core {
	t.Helper()
	core, err := NewCore()
	if err != nil {
		t.Skipf("OpenVINO not available: %v", err)
	}
	return core
}

func getTestModelPath(t *testing.T) string {
	t.Helper()
	return os.Getenv("OPENVINO_TEST_MODEL")
}
