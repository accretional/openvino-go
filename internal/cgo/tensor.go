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
	"unsafe"
)

type Tensor C.struct_openvino_tensor

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

func (t *Tensor) Destroy() {
	if t != nil {
		C.openvino_tensor_destroy(C.OpenVINOTensor(unsafe.Pointer(t)))
	}
}

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
