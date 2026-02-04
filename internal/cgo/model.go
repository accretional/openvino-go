package cgo

/*
#cgo CXXFLAGS: -std=c++17 -O2 -Wall -Wno-deprecated-declarations
#cgo LDFLAGS: -lopenvino

#include "core_wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
)

type Model C.struct_openvino_model

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

func (m *Model) Destroy() {
	if m != nil {
		C.openvino_model_destroy(C.OpenVINOModel(unsafe.Pointer(m)))
	}
}

func (m *Model) GetInputs() ([]PortInfo, error) {
	var count C.int32_t
	var cErr C.OpenVINOError

	ports := C.openvino_model_get_inputs(C.OpenVINOModel(unsafe.Pointer(m)), &count, &cErr)
	if ports == nil {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return nil, err
	}

	defer C.openvino_model_free_port_info(ports, count)

	result := make([]PortInfo, int(count))
	for i := 0; i < int(count); i++ {
		port := (*C.OpenVINOPortInfo)(unsafe.Pointer(uintptr(unsafe.Pointer(ports)) + uintptr(i)*unsafe.Sizeof(C.OpenVINOPortInfo{})))
		result[i] = PortInfo{
			Name:     C.GoString(port.name),
			DataType: DataType(port.data_type),
		}
		result[i].Shape = make([]int32, int(port.shape_size))
		for j := 0; j < int(port.shape_size); j++ {
			result[i].Shape[j] = int32(*(*C.int32_t)(unsafe.Pointer(uintptr(unsafe.Pointer(port.shape)) + uintptr(j)*unsafe.Sizeof(C.int32_t(0)))))
		}
	}

	return result, nil
}

func (m *Model) GetOutputs() ([]PortInfo, error) {
	var count C.int32_t
	var cErr C.OpenVINOError

	ports := C.openvino_model_get_outputs(C.OpenVINOModel(unsafe.Pointer(m)), &count, &cErr)
	if ports == nil {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return nil, err
	}

	defer C.openvino_model_free_port_info(ports, count)

	result := make([]PortInfo, int(count))
	for i := 0; i < int(count); i++ {
		port := (*C.OpenVINOPortInfo)(unsafe.Pointer(uintptr(unsafe.Pointer(ports)) + uintptr(i)*unsafe.Sizeof(C.OpenVINOPortInfo{})))
		result[i] = PortInfo{
			Name:     C.GoString(port.name),
			DataType: DataType(port.data_type),
		}
		result[i].Shape = make([]int32, int(port.shape_size))
		for j := 0; j < int(port.shape_size); j++ {
			result[i].Shape[j] = int32(*(*C.int32_t)(unsafe.Pointer(uintptr(unsafe.Pointer(port.shape)) + uintptr(j)*unsafe.Sizeof(C.int32_t(0)))))
		}
	}

	return result, nil
}
