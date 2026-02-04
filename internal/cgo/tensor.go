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
	case DataTypeFloat64:
		elementSize = 8
	case DataTypeInt8:
		elementSize = 1
	case DataTypeUint16:
		elementSize = 2
	case DataTypeInt16:
		elementSize = 2
	case DataTypeUint32:
		elementSize = 4
	case DataTypeUint64:
		elementSize = 8
	case DataTypeFloat16:
		elementSize = 2
	case DataTypeBFloat16:
		elementSize = 2
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

func (t *Tensor) GetDataAsFloat64() ([]float64, error) {
	data, err := t.GetData()
	if err != nil {
		return nil, err
	}

	doubles := make([]float64, len(data)/8)
	for i := range doubles {
		doubles[i] = *(*float64)(unsafe.Pointer(&data[i*8]))
	}

	return doubles, nil
}

func (t *Tensor) GetDataAsInt32() ([]int32, error) {
	data, err := t.GetData()
	if err != nil {
		return nil, err
	}

	ints := make([]int32, len(data)/4)
	for i := range ints {
		ints[i] = *(*int32)(unsafe.Pointer(&data[i*4]))
	}

	return ints, nil
}

func (t *Tensor) GetDataAsUint8() ([]uint8, error) {
	data, err := t.GetData()
	if err != nil {
		return nil, err
	}

	return data, nil
}

func (t *Tensor) GetDataAsInt8() ([]int8, error) {
	data, err := t.GetData()
	if err != nil {
		return nil, err
	}

	ints := make([]int8, len(data))
	for i := range ints {
		ints[i] = int8(data[i])
	}

	return ints, nil
}

func (t *Tensor) GetDataAsUint16() ([]uint16, error) {
	data, err := t.GetData()
	if err != nil {
		return nil, err
	}

	uints := make([]uint16, len(data)/2)
	for i := range uints {
		uints[i] = *(*uint16)(unsafe.Pointer(&data[i*2]))
	}

	return uints, nil
}

func (t *Tensor) GetDataAsInt16() ([]int16, error) {
	data, err := t.GetData()
	if err != nil {
		return nil, err
	}

	ints := make([]int16, len(data)/2)
	for i := range ints {
		ints[i] = *(*int16)(unsafe.Pointer(&data[i*2]))
	}

	return ints, nil
}

func (t *Tensor) GetDataAsUint32() ([]uint32, error) {
	data, err := t.GetData()
	if err != nil {
		return nil, err
	}

	uints := make([]uint32, len(data)/4)
	for i := range uints {
		uints[i] = *(*uint32)(unsafe.Pointer(&data[i*4]))
	}

	return uints, nil
}

func (t *Tensor) GetDataAsUint64() ([]uint64, error) {
	data, err := t.GetData()
	if err != nil {
		return nil, err
	}

	uints := make([]uint64, len(data)/8)
	for i := range uints {
		uints[i] = *(*uint64)(unsafe.Pointer(&data[i*8]))
	}

	return uints, nil
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

func NewTensor(dataType DataType, shape []int64) (*Tensor, error) {
	cShape := make([]C.int32_t, len(shape))
	for i, s := range shape {
		cShape[i] = C.int32_t(s)
	}

	var cErr C.OpenVINOError
	tensor := C.openvino_tensor_new(
		C.int32_t(dataType),
		&cShape[0],
		C.int32_t(len(shape)),
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

func NewTensorWithData(dataType DataType, shape []int64, data interface{}) (*Tensor, error) {
	cShape := make([]C.int32_t, len(shape))
	for i, s := range shape {
		cShape[i] = C.int32_t(s)
	}

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
		return nil, errors.New("unsupported data type for tensor creation")
	}

	var cErr C.OpenVINOError
	tensor := C.openvino_tensor_new_with_data(
		C.int32_t(dataType),
		&cShape[0],
		C.int32_t(len(shape)),
		dataPtr,
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

func (t *Tensor) GetSize() (int64, error) {
	var cErr C.OpenVINOError
	size := C.openvino_tensor_get_size(C.OpenVINOTensor(unsafe.Pointer(t)), &cErr)
	if size < 0 {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return -1, err
	}
	return int64(size), nil
}

func (t *Tensor) GetByteSize() (int64, error) {
	var cErr C.OpenVINOError
	byteSize := C.openvino_tensor_get_byte_size(C.OpenVINOTensor(unsafe.Pointer(t)), &cErr)
	if byteSize < 0 {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return -1, err
	}
	return int64(byteSize), nil
}

func (t *Tensor) GetElementType() (DataType, error) {
	var cErr C.OpenVINOError
	dataType := C.openvino_tensor_get_element_type(C.OpenVINOTensor(unsafe.Pointer(t)), &cErr)
	if dataType < 0 {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return 0, err
	}
	return DataType(dataType), nil
}

func (t *Tensor) SetShape(shape []int64) error {
	cShape := make([]C.int32_t, len(shape))
	for i, s := range shape {
		cShape[i] = C.int32_t(s)
	}

	var cErr C.OpenVINOError
	result := C.openvino_tensor_set_shape(
		C.OpenVINOTensor(unsafe.Pointer(t)),
		&cShape[0],
		C.int32_t(len(shape)),
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
