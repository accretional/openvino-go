package openvino

import (
	"errors"
	"fmt"
)

var (
	ErrDeviceNotFound     = errors.New("openvino: device not found")
	ErrModelLoadFailed    = errors.New("openvino: failed to load model")
	ErrModelCompileFailed = errors.New("openvino: failed to compile model")
	ErrInferenceFailed    = errors.New("openvino: inference failed")
	ErrInvalidTensor      = errors.New("openvino: invalid tensor")
	ErrUnsupportedType    = errors.New("openvino: unsupported data type")
)

type Error struct {
	Code    int32
	Message string
}

func (e *Error) Error() string {
	return fmt.Sprintf("openvino error %d: %s", e.Code, e.Message)
}

func (e *Error) Unwrap() error {
	return fmt.Errorf("openvino error: %s", e.Message)
}
