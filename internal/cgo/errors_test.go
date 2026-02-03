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

func TestError_Error_format(t *testing.T) {
	err := &Error{
		Code:    42,
		Message: "something went wrong",
	}
	got := err.Error()
	want := "openvino error 42: something went wrong"
	if got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
}

func TestError_Error_zeroValues(t *testing.T) {
	err := &Error{}
	got := err.Error()
	if got == "" {
		t.Error("Error() with zero values should not return empty string")
	}
	if got != "openvino error 0: " {
		t.Errorf("Error() = %q, want %q", got, "openvino error 0: ")
	}
}
