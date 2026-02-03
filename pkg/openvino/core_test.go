package openvino

import (
	"os"
	"testing"
)

func coreAvailable(t *testing.T) *Core {
	t.Helper()
	core, err := NewCore()
	if err != nil {
		t.Skipf("OpenVINO not available: %v", err)
	}
	return core
}

func TestNewCore(t *testing.T) {
	core, err := NewCore()
	if err != nil {
		t.Skipf("OpenVINO not available: %v", err)
		return
	}
	if core == nil {
		t.Fatal("NewCore returned nil core with nil error")
	}
	core.Close()
}

func TestCore_Close(t *testing.T) {
	core := coreAvailable(t)
	core.Close()
}

func TestCore_GetAvailableDevices(t *testing.T) {
	core := coreAvailable(t)
	defer core.Close()

	devices, err := core.GetAvailableDevices()
	if err != nil {
		t.Fatalf("GetAvailableDevices failed: %v", err)
	}
	if devices == nil {
		t.Fatal("GetAvailableDevices returned nil slice")
	}
}

func getTestModelPath(t *testing.T) string {
	t.Helper()
	return os.Getenv("OPENVINO_TEST_MODEL")
}
