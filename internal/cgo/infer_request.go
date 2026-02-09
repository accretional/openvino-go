package cgo

/*
#cgo CFLAGS: -I${SRCDIR}/../cwrapper
#cgo LDFLAGS: -L${SRCDIR}/../cwrapper/prebuilt -Wl,-rpath,${SRCDIR}/../cwrapper/prebuilt -lopenvino_wrapper -lopenvino

#include "core_wrapper.h"
#include <stdlib.h>
#include <stdint.h>

// Forward declaration for Go callback bridge
extern void openvinoGoCallbackBridge(void* user_data, int32_t has_error, const char* error_msg);
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
	case []float64:
		dataPtr = unsafe.Pointer(&v[0])
	case []int8:
		dataPtr = unsafe.Pointer(&v[0])
	case []uint16:
		dataPtr = unsafe.Pointer(&v[0])
	case []int16:
		dataPtr = unsafe.Pointer(&v[0])
	case []uint32:
		dataPtr = unsafe.Pointer(&v[0])
	case []uint64:
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
	case []float64:
		dataPtr = unsafe.Pointer(&v[0])
	case []int8:
		dataPtr = unsafe.Pointer(&v[0])
	case []uint16:
		dataPtr = unsafe.Pointer(&v[0])
	case []int16:
		dataPtr = unsafe.Pointer(&v[0])
	case []uint32:
		dataPtr = unsafe.Pointer(&v[0])
	case []uint64:
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

func (ir *InferRequest) Cancel() error {
	var cErr C.OpenVINOError
	result := C.openvino_infer_request_cancel(
		C.OpenVINOInferRequest(unsafe.Pointer(ir)),
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

func (ir *InferRequest) GetProfilingInfo() ([]ProfilingInfo, error) {
	var infoCount C.int32_t
	var infoPtr *C.OpenVINOProfilingInfo
	var cErr C.OpenVINOError

	result := C.openvino_infer_request_get_profiling_info(
		C.OpenVINOInferRequest(unsafe.Pointer(ir)),
		&infoPtr,
		&infoCount,
		&cErr,
	)

	if result != 0 {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return nil, err
	}

	if infoCount == 0 || infoPtr == nil {
		return []ProfilingInfo{}, nil
	}

	defer C.openvino_profiling_info_free(infoPtr, infoCount)

	infos := make([]ProfilingInfo, int(infoCount))
	infoSlice := unsafe.Slice(infoPtr, int(infoCount))

	for i := range infos {
		cInfo := infoSlice[i]
		infos[i] = ProfilingInfo{
			Status:   ProfilingInfoStatus(cInfo.status),
			RealTime: int64(cInfo.real_time_us),
			CPUTime:  int64(cInfo.cpu_time_us),
			NodeName: C.GoString(cInfo.node_name),
			ExecType: C.GoString(cInfo.exec_type),
			NodeType: C.GoString(cInfo.node_type),
		}
	}

	return infos, nil
}

type ProfilingInfoStatus int32

const (
	ProfilingInfoStatusNotRun       ProfilingInfoStatus = 0
	ProfilingInfoStatusOptimizedOut ProfilingInfoStatus = 1
	ProfilingInfoStatusExecuted     ProfilingInfoStatus = 2
)

type ProfilingInfo struct {
	Status   ProfilingInfoStatus
	RealTime int64
	CPUTime  int64
	NodeName string
	ExecType string
	NodeType string
}

func (ir *InferRequest) GetTensor(name string) (*Tensor, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	var cErr C.OpenVINOError
	tensor := C.openvino_infer_request_get_tensor(
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

func (ir *InferRequest) SetTensor(name string, tensor *Tensor) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	if tensor == nil {
		return errors.New("tensor cannot be nil")
	}

	var cErr C.OpenVINOError
	result := C.openvino_infer_request_set_tensor_unified(
		C.OpenVINOInferRequest(unsafe.Pointer(ir)),
		cName,
		C.OpenVINOTensor(unsafe.Pointer(tensor)),
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

func (ir *InferRequest) SetInputTensors(name string, tensors []*Tensor) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	if len(tensors) == 0 {
		return errors.New("tensor slice cannot be empty")
	}

	cTensors := make([]C.OpenVINOTensor, len(tensors))
	for i, t := range tensors {
		if t == nil {
			return errors.New("tensor cannot be nil")
		}
		cTensors[i] = C.OpenVINOTensor(unsafe.Pointer(t))
	}

	var cErr C.OpenVINOError
	result := C.openvino_infer_request_set_tensors(
		C.OpenVINOInferRequest(unsafe.Pointer(ir)),
		cName,
		&cTensors[0],
		C.int32_t(len(tensors)),
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

func (ir *InferRequest) SetInputTensorsByIndex(index int32, tensors []*Tensor) error {
	if len(tensors) == 0 {
		return errors.New("tensor slice cannot be empty")
	}

	cTensors := make([]C.OpenVINOTensor, len(tensors))
	for i, t := range tensors {
		if t == nil {
			return errors.New("tensor cannot be nil")
		}
		cTensors[i] = C.OpenVINOTensor(unsafe.Pointer(t))
	}

	var cErr C.OpenVINOError
	result := C.openvino_infer_request_set_tensors_by_index(
		C.OpenVINOInferRequest(unsafe.Pointer(ir)),
		C.int32_t(index),
		&cTensors[0],
		C.int32_t(len(tensors)),
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

func (ir *InferRequest) SetOutputTensor(name string, tensor *Tensor) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	if tensor == nil {
		return errors.New("tensor cannot be nil")
	}

	var cErr C.OpenVINOError
	result := C.openvino_infer_request_set_output_tensor(
		C.OpenVINOInferRequest(unsafe.Pointer(ir)),
		cName,
		C.OpenVINOTensor(unsafe.Pointer(tensor)),
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

func (ir *InferRequest) SetOutputTensorByIndex(index int32, tensor *Tensor) error {
	if tensor == nil {
		return errors.New("tensor cannot be nil")
	}

	var cErr C.OpenVINOError
	result := C.openvino_infer_request_set_output_tensor_by_index(
		C.OpenVINOInferRequest(unsafe.Pointer(ir)),
		C.int32_t(index),
		C.OpenVINOTensor(unsafe.Pointer(tensor)),
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
