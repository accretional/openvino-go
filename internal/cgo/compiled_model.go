package cgo

/*
#cgo CFLAGS: -I${SRCDIR}/../cwrapper
#cgo LDFLAGS: -L${SRCDIR}/../cwrapper -Wl,-rpath,${SRCDIR}/../cwrapper -lopenvino_wrapper -lopenvino

#include "core_wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"strings"
	"unsafe"
)

type CompiledModel C.struct_openvino_compiled_model

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

func (c *Core) CompileModelWithProperties(model *Model, device string, properties map[string]string) (*CompiledModel, error) {
	cDevice := C.CString(device)
	defer C.free(unsafe.Pointer(cDevice))

	var keysBuilder, valuesBuilder strings.Builder
	first := true
	for k, v := range properties {
		if !first {
			keysBuilder.WriteString(",")
			valuesBuilder.WriteString(",")
		}
		keysBuilder.WriteString(k)
		valuesBuilder.WriteString(v)
		first = false
	}

	cKeys := C.CString(keysBuilder.String())
	defer C.free(unsafe.Pointer(cKeys))
	cValues := C.CString(valuesBuilder.String())
	defer C.free(unsafe.Pointer(cValues))

	var cErr C.OpenVINOError
	compiled := C.openvino_core_compile_model_with_properties(
		C.OpenVINOCore(unsafe.Pointer(c)),
		C.OpenVINOModel(unsafe.Pointer(model)),
		cDevice,
		cKeys,
		cValues,
		C.int32_t(len(properties)),
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

func (cm *CompiledModel) Destroy() {
	if cm != nil {
		C.openvino_compiled_model_destroy(C.OpenVINOCompiledModel(unsafe.Pointer(cm)))
	}
}

func (cm *CompiledModel) ReleaseMemory() error {
	var cErr C.OpenVINOError
	result := C.openvino_compiled_model_release_memory(
		C.OpenVINOCompiledModel(unsafe.Pointer(cm)),
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
