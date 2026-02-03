package cgo

import "fmt"

type Error struct {
	Code    int32
	Message string
}

func (e *Error) Error() string {
	return fmt.Sprintf("openvino error %d: %s", e.Code, e.Message)
}
