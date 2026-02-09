package cgo

/*
#cgo CFLAGS: -I${SRCDIR}/../cwrapper
#cgo LDFLAGS: -L${SRCDIR}/../cwrapper/prebuilt -Wl,-rpath,${SRCDIR}/../cwrapper/prebuilt -lopenvino_wrapper -lopenvino

#include "core_wrapper.h"
#include <stdlib.h>
#include <stdint.h>
*/
import "C"
import (
	"unsafe"
)

type VariableState C.struct_openvino_variable_state

func (ir *InferRequest) QueryState() ([]*VariableState, error) {
	var stateCount C.int32_t
	var cErr C.OpenVINOError
	var statesPtr *C.OpenVINOVariableState

	result := C.openvino_infer_request_query_state(
		C.OpenVINOInferRequest(unsafe.Pointer(ir)),
		&statesPtr,
		&stateCount,
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

	if stateCount == 0 || statesPtr == nil {
		return []*VariableState{}, nil
	}

	// statesPtr points to the first element of an array of OpenVINOVariableState (pointers)
	statesSlice := unsafe.Slice(statesPtr, int(stateCount))
	states := make([]*VariableState, int(stateCount))
	for i := range states {
		states[i] = (*VariableState)(unsafe.Pointer(statesSlice[i]))
	}

	return states, nil
}

func (ir *InferRequest) ResetState() error {
	var cErr C.OpenVINOError
	result := C.openvino_infer_request_reset_state(
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

func (vs *VariableState) Destroy() {
	if vs != nil {
		C.openvino_variable_state_destroy(C.OpenVINOVariableState(unsafe.Pointer(vs)))
	}
}

func (vs *VariableState) GetName() (string, error) {
	var cErr C.OpenVINOError
	namePtr := C.openvino_variable_state_get_name(
		C.OpenVINOVariableState(unsafe.Pointer(vs)),
		&cErr,
	)

	if namePtr == nil {
		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return "", err
	}

	name := C.GoString(namePtr)
	C.openvino_variable_state_free_name(namePtr)
	return name, nil
}

func (vs *VariableState) GetState() (*Tensor, error) {
	var cErr C.OpenVINOError
	tensor := C.openvino_variable_state_get_state(
		C.OpenVINOVariableState(unsafe.Pointer(vs)),
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

func (vs *VariableState) SetState(tensor *Tensor) error {
	var cErr C.OpenVINOError
	result := C.openvino_variable_state_set_state(
		C.OpenVINOVariableState(unsafe.Pointer(vs)),
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

func (vs *VariableState) Reset() error {
	var cErr C.OpenVINOError
	result := C.openvino_variable_state_reset(
		C.OpenVINOVariableState(unsafe.Pointer(vs)),
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
