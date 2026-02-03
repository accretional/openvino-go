package cgo

import "testing"

func TestError_Error(t *testing.T) {
	err := &Error{
		Code:    1,
		Message: "test error",
	}
	if err.Error() == "" {
		t.Error("Error() should return non-empty string")
	}
}
