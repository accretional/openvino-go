package cgo

/*
#cgo CXXFLAGS: -std=c++17 -O2 -Wall -Wno-deprecated-declarations
#cgo LDFLAGS: -lopenvino

#include "core_wrapper.h"
#include <stdlib.h>
#include <string.h>
*/
import "C"
import (
	"unsafe"
)

type Core C.struct_openvino_core

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

func (c *Core) Destroy() {
	if c != nil {
		C.openvino_core_destroy(C.OpenVINOCore(unsafe.Pointer(c)))
	}
}

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

func IsAvailable() bool {
	core, err := CreateCore()
	if err != nil {
		return false
	}
	defer core.Destroy()
	return true
}
