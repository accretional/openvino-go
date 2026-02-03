package openvino

import (
	"errors"
	"testing"
)

func TestError_Error(t *testing.T) {
	err := &Error{
		Code:    1,
		Message: "test error",
	}
	if err.Error() == "" {
		t.Error("Error() should return non-empty string")
	}
}

func TestError_Unwrap(t *testing.T) {
	err := &Error{
		Code:    1,
		Message: "test error",
	}
	unwrapped := err.Unwrap()
	if unwrapped == nil {
		t.Error("Unwrap() should return non-nil error")
	}
}

func TestErrorVariables(t *testing.T) {
	tests := []struct {
		name string
		err  error
	}{
		{"ErrDeviceNotFound", ErrDeviceNotFound},
		{"ErrModelLoadFailed", ErrModelLoadFailed},
		{"ErrModelCompileFailed", ErrModelCompileFailed},
		{"ErrInferenceFailed", ErrInferenceFailed},
		{"ErrInvalidTensor", ErrInvalidTensor},
		{"ErrUnsupportedType", ErrUnsupportedType},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.err == nil {
				t.Errorf("%s should not be nil", tt.name)
			}
			if !errors.Is(tt.err, tt.err) {
				t.Errorf("%s should be comparable with errors.Is", tt.name)
			}
		})
	}
}
