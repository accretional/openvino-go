package cgo

/*
#cgo CFLAGS: -I${SRCDIR}/../cwrapper
#cgo LDFLAGS: -L${SRCDIR}/../cwrapper -Wl,-rpath,${SRCDIR}/../cwrapper -lopenvino_wrapper -lopenvino

#include "core_wrapper.h"
#include <stdlib.h>
#include <stdint.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

type InferRequest C.struct_openvino_infer_request

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

func (ir *InferRequest) Destroy() {
	if ir != nil {
		C.openvino_infer_request_destroy(C.OpenVINOInferRequest(unsafe.Pointer(ir)))
	}
}

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

func (ir *InferRequest) StartAsync() error {
	var cErr C.OpenVINOError
	result := C.openvino_infer_request_start_async(C.OpenVINOInferRequest(unsafe.Pointer(ir)), &cErr)

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

func (ir *InferRequest) Wait() error {
	var cErr C.OpenVINOError
	result := C.openvino_infer_request_wait(C.OpenVINOInferRequest(unsafe.Pointer(ir)), &cErr)

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

func (ir *InferRequest) WaitFor(timeoutMs int64) (bool, error) {
	var cErr C.OpenVINOError
	result := C.openvino_infer_request_wait_for(
		C.OpenVINOInferRequest(unsafe.Pointer(ir)),
		C.int64_t(timeoutMs),
		&cErr,
	)

	if result == -1 {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return false, err
	}

	// result: 0 = completed, 1 = timeout
	completed := result == 0
	return completed, nil
}

func (ir *InferRequest) GetInputTensor(name string) (*Tensor, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	var cErr C.OpenVINOError
	tensor := C.openvino_infer_request_get_input_tensor(
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

func (ir *InferRequest) GetInputTensorByIndex(index int32) (*Tensor, error) {
	var cErr C.OpenVINOError
	tensor := C.openvino_infer_request_get_input_tensor_by_index(
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
