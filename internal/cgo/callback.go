package cgo

/*
#cgo CFLAGS: -I${SRCDIR}/../cwrapper
#cgo LDFLAGS: -L${SRCDIR}/../cwrapper -Wl,-rpath,${SRCDIR}/../cwrapper -lopenvino_wrapper -lopenvino

#include "core_wrapper.h"
#include <stdlib.h>

extern void openvinoGoCallbackBridge(void* user_data, int32_t has_error, const char* error_msg);
*/
import "C"
import (
	"errors"
	"sync"
	"unsafe"
)

var (
	callbackRegistry = make(map[unsafe.Pointer]chan error)
	callbackMutex    sync.RWMutex
)

//export openvinoGoCallbackBridge
func openvinoGoCallbackBridge(userData unsafe.Pointer, hasError C.int32_t, errorMsg *C.char) {
	if userData == nil {
		return
	}

	callbackMutex.RLock()
	callbackChan, exists := callbackRegistry[userData]
	callbackMutex.RUnlock()

	if !exists {
		return
	}

	var err error
	if hasError != 0 && errorMsg != nil {
		err = errors.New(C.GoString(errorMsg))
	}

	select {
	case callbackChan <- err:
	default:
	}
}

func (ir *InferRequest) SetCallback(callback func(error)) error {
	if callback == nil {
		C.openvino_infer_request_clear_callback(C.OpenVINOInferRequest(unsafe.Pointer(ir)))
		return nil
	}

	callbackChan := make(chan error, 1)
	requestPtr := unsafe.Pointer(ir)

	callbackMutex.Lock()
	if oldChan, exists := callbackRegistry[requestPtr]; exists {
		close(oldChan)
		delete(callbackRegistry, requestPtr)
	}
	callbackRegistry[requestPtr] = callbackChan
	callbackMutex.Unlock()

	go func() {
		err := <-callbackChan
		callback(err)
		callbackMutex.Lock()
		delete(callbackRegistry, requestPtr)
		callbackMutex.Unlock()
	}()

	cCallback := C.OpenVINOCallback(C.openvinoGoCallbackBridge)
	userData := requestPtr

	var cErr C.OpenVINOError
	result := C.openvino_infer_request_set_callback(
		C.OpenVINOInferRequest(unsafe.Pointer(ir)),
		cCallback,
		userData,
		&cErr,
	)

	if result != 0 {
		callbackMutex.Lock()
		if ch, exists := callbackRegistry[requestPtr]; exists {
			close(ch)
			delete(callbackRegistry, requestPtr)
		}
		callbackMutex.Unlock()

		err := &Error{
			Code:    int32(cErr.code),
			Message: C.GoString(cErr.message),
		}
		C.openvino_error_free(&cErr)
		return err
	}

	return nil
}
