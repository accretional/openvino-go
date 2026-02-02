// Package cgo provides CGO bindings to the C wrapper
// This is the bridge between Go and the C++ wrapper

package cgo

/*
#cgo CFLAGS: -I${SRCDIR}/../cwrapper
#cgo LDFLAGS: -L${SRCDIR}/../cwrapper -Wl,-rpath,${SRCDIR}/../cwrapper -lopenvino_wrapper -lopenvino

#include "core_wrapper.h"
#include <stdlib.h>
#include <string.h>
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
type Core C.struct_openvino_core

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

	return (*Core)(unsafe.Pointer(core)), nil
}

// Destroy releases the Core instance
func (c *Core) Destroy() {
	if c != nil {
		C.openvino_core_destroy(C.OpenVINOCore(unsafe.Pointer(c)))
	}
}

// GetAvailableDevices returns a list of available devices
func (c *Core) GetAvailableDevices() ([]string, error) {
	var count C.int32_t
	var cErr C.OpenVINOError

	devices := C.openvino_core_get_available_devices(C.OpenVINOCore(unsafe.Pointer(c)), &count, &cErr)

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
type Model C.struct_openvino_model

// ReadModel loads a model from a file path
func (c *Core) ReadModel(modelPath string) (*Model, error) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	var cErr C.OpenVINOError
	model := C.openvino_core_read_model(C.OpenVINOCore(unsafe.Pointer(c)), cPath, &cErr)

	if model == nil {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return nil, err
	}

	return (*Model)(unsafe.Pointer(model)), nil
}

// Destroy releases the Model instance
func (m *Model) Destroy() {
	if m != nil {
		C.openvino_model_destroy(C.OpenVINOModel(unsafe.Pointer(m)))
	}
}

// CompiledModel represents a compiled OpenVINO model
type CompiledModel C.struct_openvino_compiled_model

// CompileModel compiles a model for a specific device
func (c *Core) CompileModel(model *Model, device string) (*CompiledModel, error) {
	cDevice := C.CString(device)
	defer C.free(unsafe.Pointer(cDevice))

	var cErr C.OpenVINOError
	compiled := C.openvino_core_compile_model(
		C.OpenVINOCore(unsafe.Pointer(c)),
		C.OpenVINOModel(unsafe.Pointer(model)),
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

	return (*CompiledModel)(unsafe.Pointer(compiled)), nil
}

// Destroy releases the CompiledModel instance
func (cm *CompiledModel) Destroy() {
	if cm != nil {
		C.openvino_compiled_model_destroy(C.OpenVINOCompiledModel(unsafe.Pointer(cm)))
	}
}

// InferRequest represents an inference request
type InferRequest C.struct_openvino_infer_request

// CreateInferRequest creates a new inference request
func (cm *CompiledModel) CreateInferRequest() (*InferRequest, error) {
	var cErr C.OpenVINOError
	request := C.openvino_compiled_model_create_infer_request(
		C.OpenVINOCompiledModel(unsafe.Pointer(cm)),
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

	return (*InferRequest)(unsafe.Pointer(request)), nil
}

// Destroy releases the InferRequest instance
func (ir *InferRequest) Destroy() {
	if ir != nil {
		C.openvino_infer_request_destroy(C.OpenVINOInferRequest(unsafe.Pointer(ir)))
	}
}

// DataType represents the data type of a tensor
type DataType int32

const (
	DataTypeFloat32 DataType = 0
	DataTypeInt64   DataType = 1
	DataTypeInt32   DataType = 2
	DataTypeUint8   DataType = 3
)

// SetInputTensor sets an input tensor by name
func (ir *InferRequest) SetInputTensor(name string, data interface{}, shape []int64, dataType DataType) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	cShape := make([]C.int32_t, len(shape))
	for i, s := range shape {
		cShape[i] = C.int32_t(s)
	}

	var cErr C.OpenVINOError
	var dataPtr unsafe.Pointer

	switch v := data.(type) {
	case []float32:
		dataPtr = unsafe.Pointer(&v[0])
	case []int64:
		dataPtr = unsafe.Pointer(&v[0])
	case []int32:
		dataPtr = unsafe.Pointer(&v[0])
	case []uint8:
		dataPtr = unsafe.Pointer(&v[0])
	default:
		return errors.New("unsupported data type")
	}

	result := C.openvino_infer_request_set_input_tensor(
		C.OpenVINOInferRequest(unsafe.Pointer(ir)),
		cName,
		dataPtr,
		&cShape[0],
		C.int32_t(len(shape)),
		C.int32_t(dataType),
		&cErr,
	)

	if result != 0 {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return err
	}

	return nil
}

// SetInputTensorByIndex sets an input tensor by index
func (ir *InferRequest) SetInputTensorByIndex(index int32, data interface{}, shape []int64, dataType DataType) error {
	cShape := make([]C.int32_t, len(shape))
	for i, s := range shape {
		cShape[i] = C.int32_t(s)
	}

	var cErr C.OpenVINOError
	var dataPtr unsafe.Pointer

	switch v := data.(type) {
	case []float32:
		dataPtr = unsafe.Pointer(&v[0])
	case []int64:
		dataPtr = unsafe.Pointer(&v[0])
	case []int32:
		dataPtr = unsafe.Pointer(&v[0])
	case []uint8:
		dataPtr = unsafe.Pointer(&v[0])
	default:
		return errors.New("unsupported data type")
	}

	result := C.openvino_infer_request_set_input_tensor_by_index(
		C.OpenVINOInferRequest(unsafe.Pointer(ir)),
		C.int32_t(index),
		dataPtr,
		&cShape[0],
		C.int32_t(len(shape)),
		C.int32_t(dataType),
		&cErr,
	)

	if result != 0 {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return err
	}

	return nil
}

// Infer runs synchronous inference
func (ir *InferRequest) Infer() error {
	var cErr C.OpenVINOError
	result := C.openvino_infer_request_infer(C.OpenVINOInferRequest(unsafe.Pointer(ir)), &cErr)

	if result != 0 {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return err
	}

	return nil
}

// Tensor represents an OpenVINO tensor
type Tensor C.struct_openvino_tensor

// GetOutputTensor gets an output tensor by name
func (ir *InferRequest) GetOutputTensor(name string) (*Tensor, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	var cErr C.OpenVINOError
	tensor := C.openvino_infer_request_get_output_tensor(
		C.OpenVINOInferRequest(unsafe.Pointer(ir)),
		cName,
		&cErr,
	)

	if tensor == nil {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return nil, err
	}

	return (*Tensor)(unsafe.Pointer(tensor)), nil
}

// GetOutputTensorByIndex gets an output tensor by index
func (ir *InferRequest) GetOutputTensorByIndex(index int32) (*Tensor, error) {
	var cErr C.OpenVINOError
	tensor := C.openvino_infer_request_get_output_tensor_by_index(
		C.OpenVINOInferRequest(unsafe.Pointer(ir)),
		C.int32_t(index),
		&cErr,
	)

	if tensor == nil {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return nil, err
	}

	return (*Tensor)(unsafe.Pointer(tensor)), nil
}

// Destroy releases the Tensor instance
func (t *Tensor) Destroy() {
	if t != nil {
		C.openvino_tensor_destroy(C.OpenVINOTensor(unsafe.Pointer(t)))
	}
}

// GetData returns the tensor data as a byte slice
func (t *Tensor) GetData() ([]byte, error) {
	var dataType C.int32_t
	var cErr C.OpenVINOError

	dataPtr := C.openvino_tensor_get_data(C.OpenVINOTensor(unsafe.Pointer(t)), &dataType, &cErr)
	if dataPtr == nil {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return nil, err
	}

	shape, err := t.GetShape()
	if err != nil {
		return nil, err
	}

	totalElements := int64(1)
	for _, dim := range shape {
		totalElements *= int64(dim)
	}

	var elementSize int
	switch DataType(dataType) {
	case DataTypeFloat32:
		elementSize = 4
	case DataTypeInt64:
		elementSize = 8
	case DataTypeInt32:
		elementSize = 4
	case DataTypeUint8:
		elementSize = 1
	default:
		elementSize = 4
	}

	dataSize := int(totalElements) * elementSize
	data := make([]byte, dataSize)
	C.memcpy(unsafe.Pointer(&data[0]), dataPtr, C.size_t(dataSize))

	return data, nil
}

// GetDataAsFloat32 returns the tensor data as float32 slice
func (t *Tensor) GetDataAsFloat32() ([]float32, error) {
	data, err := t.GetData()
	if err != nil {
		return nil, err
	}

	floats := make([]float32, len(data)/4)
	for i := range floats {
		floats[i] = *(*float32)(unsafe.Pointer(&data[i*4]))
	}

	return floats, nil
}

// GetDataAsInt64 returns the tensor data as int64 slice
func (t *Tensor) GetDataAsInt64() ([]int64, error) {
	data, err := t.GetData()
	if err != nil {
		return nil, err
	}

	ints := make([]int64, len(data)/8)
	for i := range ints {
		ints[i] = *(*int64)(unsafe.Pointer(&data[i*8]))
	}

	return ints, nil
}

// GetShape returns the tensor shape
func (t *Tensor) GetShape() ([]int32, error) {
	var shapeSize C.int32_t
	var cErr C.OpenVINOError

	shapePtr := C.openvino_tensor_get_shape(C.OpenVINOTensor(unsafe.Pointer(t)), &shapeSize, &cErr)
	if shapePtr == nil {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return nil, err
	}

	defer C.openvino_tensor_free_shape(shapePtr)

	shape := make([]int32, int(shapeSize))
	for i := 0; i < int(shapeSize); i++ {
		shape[i] = int32(*(*C.int32_t)(unsafe.Pointer(uintptr(unsafe.Pointer(shapePtr)) + uintptr(i)*unsafe.Sizeof(C.int32_t(0)))))
	}

	return shape, nil
}

// IsAvailable checks if OpenVINO is available
func IsAvailable() bool {
	core, err := CreateCore()
	if err != nil {
		return false
	}
	defer core.Destroy()
	return true
}
