package openvino

import (
	"os"
	"path/filepath"
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
	path := os.Getenv("OPENVINO_TEST_MODEL")
	if path == "" {
		return ""
	}
	// Resolve relative paths against cwd first, then module root, so "models/test_model.onnx" works
	if !filepath.IsAbs(path) {
		cwd, _ := os.Getwd()
		for dir := cwd; dir != ""; dir = filepath.Dir(dir) {
			try := filepath.Join(dir, path)
			if _, err := os.Stat(try); err == nil {
				return try
			}
			if dir == filepath.Dir(dir) {
				break
			}
		}
		path = filepath.Join(cwd, path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("OPENVINO_TEST_MODEL: file not found %q (use absolute path or run tests from repo root)", path)
		return ""
	}
	return path
}
