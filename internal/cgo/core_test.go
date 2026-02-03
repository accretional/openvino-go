package cgo

import "testing"

func coreAvailable(t *testing.T) *Core {
	t.Helper()
	core, err := CreateCore()
	if err != nil {
		t.Skipf("OpenVINO not available: %v", err)
	}
	return core
}

func TestCreateCore(t *testing.T) {
	core, err := CreateCore()
	if err != nil {
		t.Skipf("OpenVINO not available: %v", err)
		return
	}
	if core == nil {
		t.Fatal("CreateCore returned nil core with nil error")
	}
	core.Destroy()
}

func TestCore_Destroy(t *testing.T) {
	core := coreAvailable(t)
	core.Destroy()
}

func TestCore_Destroy_nil(t *testing.T) {
	var core *Core
	// Destroy on nil should not panic
	core.Destroy()
}

func TestCore_GetAvailableDevices(t *testing.T) {
	core := coreAvailable(t)
	defer core.Destroy()

	devices, err := core.GetAvailableDevices()
	if err != nil {
		t.Fatalf("GetAvailableDevices failed: %v", err)
	}
	if devices == nil {
		t.Fatal("GetAvailableDevices returned nil slice")
	}
	// At least CPU should typically be available
	if len(devices) == 0 {
		t.Log("no devices reported (unusual)")
	}
}

func TestIsAvailable(t *testing.T) {
	// IsAvailable should not panic and return true or false
	ok := IsAvailable()
	_ = ok
}
