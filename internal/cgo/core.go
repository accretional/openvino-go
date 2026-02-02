// Package cgo provides CGO bindings to the C wrapper
// This is the bridge between Go and the C++ wrapper

package cgo

/*
#cgo CFLAGS: -I${SRCDIR}/../cwrapper
#cgo LDFLAGS: -L${SRCDIR}/../cwrapper -lopenvino_wrapper -lov::runtime

#include "core_wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

// Error represents an OpenVINO error
type Error struct {
	Code    int32
	Message string
}

func (e *Error) Error() string {
	return fmt.Sprintf("openvino error %d: %s", e.Code, e.Message)
}

// Core represents an OpenVINO Core instance
type Core C.OpenVINOCore

// CreateCore creates a new OpenVINO Core instance
func CreateCore() (*Core, error) {
	var cErr C.OpenVINOError
	core := C.openvino_core_create(&cErr)
	
	if core == nil {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return nil, err
	}
	
	return (*Core)(core), nil
}

// Destroy releases the Core instance
func (c *Core) Destroy() {
	if c != nil {
		C.openvino_core_destroy(C.OpenVINOCore(c))
	}
}

// GetAvailableDevices returns a list of available devices
func (c *Core) GetAvailableDevices() ([]string, error) {
	var count C.int32_t
	var cErr C.OpenVINOError
	
	devices := C.openvino_core_get_available_devices(C.OpenVINOCore(c), &count, &cErr)
	
	if devices == nil {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return nil, err
	}
	
	defer C.openvino_core_free_device_list(devices, count)
	
	result := make([]string, int(count))
	for i := 0; i < int(count); i++ {
		ptr := (*C.char)(*(**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(devices)) + uintptr(i)*unsafe.Sizeof(devices))))
		result[i] = C.GoString(ptr)
	}
	
	return result, nil
}

// Model represents an OpenVINO Model
type Model C.OpenVINOModel

// ReadModel loads a model from a file path
func (c *Core) ReadModel(modelPath string) (*Model, error) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	
	var cErr C.OpenVINOError
	model := C.openvino_core_read_model(C.OpenVINOCore(c), cPath, &cErr)
	
	if model == nil {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return nil, err
	}
	
	return (*Model)(model), nil
}

// Destroy releases the Model instance
func (m *Model) Destroy() {
	if m != nil {
		C.openvino_model_destroy(C.OpenVINOModel(m))
	}
}

// CompiledModel represents a compiled OpenVINO model
type CompiledModel C.OpenVINOCompiledModel

// CompileModel compiles a model for a specific device
func (c *Core) CompileModel(model *Model, device string) (*CompiledModel, error) {
	cDevice := C.CString(device)
	defer C.free(unsafe.Pointer(cDevice))
	
	var cErr C.OpenVINOError
	compiled := C.openvino_core_compile_model(
		C.OpenVINOCore(c),
		C.OpenVINOModel(model),
		cDevice,
		&cErr,
	)
	
	if compiled == nil {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return nil, err
	}
	
	return (*CompiledModel)(compiled), nil
}

// Destroy releases the CompiledModel instance
func (cm *CompiledModel) Destroy() {
	if cm != nil {
		C.openvino_compiled_model_destroy(C.OpenVINOCompiledModel(cm))
	}
}

// InferRequest represents an inference request
type InferRequest C.OpenVINOInferRequest

// CreateInferRequest creates a new inference request
func (cm *CompiledModel) CreateInferRequest() (*InferRequest, error) {
	var cErr C.OpenVINOError
	request := C.openvino_compiled_model_create_infer_request(
		C.OpenVINOCompiledModel(cm),
		&cErr,
	)
	
	if request == nil {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return nil, err
	}
	
	return (*InferRequest)(request), nil
}

// Destroy releases the InferRequest instance
func (ir *InferRequest) Destroy() {
	if ir != nil {
		C.openvino_infer_request_destroy(C.OpenVINOInferRequest(ir))
	}
}

// Helper function to check if OpenVINO is available
func IsAvailable() bool {
	core, err := CreateCore()
	if err != nil {
		return false
	}
	defer core.Destroy()
	return true
}
